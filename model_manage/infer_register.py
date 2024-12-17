## This file encapsulates the inference code for each model
## Uses Python's registration mechanism to implement
## Does not return history (i.e., multi-turn dialogue is not currently supported)

import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import pandas as pd 
from tqdm import tqdm
import httpx
import torch.nn.functional as F 
from openai import OpenAI
from model_inform import (
    MODEL_ROOT_DIR,
    NAME_REPATH_TYPE_DICT,
    MODEL_TYPE,
    EVAL_PATH_DICT
)
import json 

# Load environment variables from the .env file
load_dotenv()

class ModelInfer:
    model_root_dir = MODEL_ROOT_DIR
    ## Add models by modifying this dictionary with model name and relative path to the model root directory
    name_repath_type_dict = NAME_REPATH_TYPE_DICT
    model_type=MODEL_TYPE
    physic_eval_csv = EVAL_PATH_DICT['physic_eval_csv']
    normal_eval_csv = EVAL_PATH_DICT['normal_eval_csv']
    topo_eval_csv = EVAL_PATH_DICT['topo_eval_csv']

      
    def __init__(self, *args, **kwargs):
        self._infer_dict = {}
        self.model =None
        self.tokenizer = None 
        self.current_infer_func =None
        self.current_model_name = None

    def __call__(self, target):
        return self.infer_register(target)
   
    def infer_register(self, target):
        def add_infer(model_type, infer_func):
            if model_type not in self.model_type:
                raise ValueError("Before registration, please first modify the corresponding content in `model_type`.")
            if not callable(infer_func):
                raise Exception(f"Error:{infer_func} must be callable!")
            if model_type in self._infer_dict:
                print(f"\033[31mWarning:\033[0m {infer_func.__name__} already exists and will be overwritten!")
            self._infer_dict[model_type] = infer_func
            return infer_func
         ## The registration name should be passed her
        return lambda x : add_infer(target, x)  
    
    def load_model(self,model_name:str,dtype=torch.float16,device="auto",adapter_name:str=None,adapter_path:str=None,specific_path:str =None):
        """
        Loads a specified model with given precision and onto a specific device.

        Parameters:
        - model_name (str): The name of the model to be loaded. This name should correspond to a predefined or custom model available in the environment.
        - adapter_name (str, optional): The name of the adapter to be used with the model. Adapters allow for model customization and task-specific tuning without altering the original model weights.
        - adapter_path (str, optional): The path to the adapter's location. This is required if an adapter is to be used with the model,\ indicating where the adapter configuration and weights are stored.
        - dtype (torch.dtype): The data type for the model's inputs and outputs, determining the precision of computation. Common values include torch.float32 for single precision and torch.float16 for reduced precision to save memory and accelerate inference.
        - device (str or torch.device): The target device for model deployment. This can be 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc., depending on the hardware availability and requirements, or 'auto' to automatically select the most appropriate device.

        This function loads the specified model along with an optional adapter. If the 'device' parameter is set to 'auto', the function automatically selects the most appropriate device (GPU if available, otherwise CPU) for model deployment. The model is set to evaluation mode after loading.
         
        """
        if bool(adapter_name) != bool(adapter_path):
            raise ValueError("Loading an adapter requires both 'adapter_name' and 'adapter_path' to be provided. Missing either name or path.")

        if model_name == "gpt3.5":
         ## The registration name should be passed her
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.Client(
                    base_url=base_url,
                    follow_redirects=True,
                ),
            )
            self.model = client  
            self.current_model_name = "gpt3.5"
            self.current_infer_func = self._infer_dict["gpt3.5"]
            print("Loaded GPT-3.5 model for inference.")

        elif model_name == "gpt4":
            api_key = os.getenv("OPENAI_API_KEY_4")
            base_url = os.getenv("BASE_URL_4")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.Client(
                    base_url=base_url,
                    follow_redirects=True,
                ),
            )
            self.model = client  
            self.current_model_name = "gpt4"
            self.current_infer_func = self._infer_dict["gpt4"]
            print("Loaded GPT-4 model for inference.")      
        else:
            # Check if specific_path and specific_type are provided
            if specific_path:
                # Load model from provided specific path and type
                print(f"Loading fully fine-tuned model from specific path: {specific_path}")
                model_path = specific_path
                model_type = self.name_repath_type_dict[model_name]['type']
                self.current_infer_func = self._infer_dict[model_type]
            
             # If specific_path and specific_type are not provided, load model from dictionary
            elif model_name in self.name_repath_type_dict:
                self.current_model_name = model_name
                model_path = os.path.join(self.model_root_dir, self.name_repath_type_dict[model_name]["path"])
                model_type = self.name_repath_type_dict[model_name]['type']
                self.current_infer_func = self._infer_dict[model_type]
            
            else:
                # If the model name is unrecognized, raise an exception
                raise ValueError(
                    f"Model name '{model_name}' is not recognized. Please check the name and try again. "
                    f"You can print available model names by checking ModelInference.name_repath_type_dict."
                )
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.model_root_dir, self.name_repath_type_dict[model_name]["path"])
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
            
            # Load the model to the appropriate device
            if device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).to(device).eval()
            
            print(f"Loaded model from {model_path} successfully, model is on device {self.model.device}")

             # If adapter_name and adapter_path are specified, load the adapter
            if adapter_name and adapter_path:
                self.model = PeftModel.from_pretrained(self.model, model_id=adapter_path, adapter_name=adapter_name)
                self.model = self.model.merge_and_unload()
                print(f"Loaded adapter '{adapter_name}' for model successfully from {adapter_path}\n")

        
            

    def model_infer(self, input_data,**kwargs):
        
        """
        Performs inference using the loaded model with the given input and sampling parameters.
        Parameters:
        - input: The input data to be processed by the model. This can be in various formats such as a string, a list of tokens, or any other model-specific input format.
        - **kwargs: A dictionary of keyword arguments that specify the sampling parameters and other options for the model inference. These parameters can include settings like temperature, \
            max_new_tokens,max_length, top_k, top_p, etc., depending on the model's capabilities and the desired behavior of the inference.
        """
        
         # Check if the model is loaded, except for gpt3.5 and gpt4
        if self.current_model_name != "gpt3.5" and self.current_model_name != "gpt4" and (not self.model or not self.tokenizer):
            raise ValueError("Model not loaded. Please load a model first.")

        response= self.current_infer_func(self.model, self.tokenizer, input_data,**kwargs)
        return response
   
    def switch_model(self, model_name:str, dtype=torch.float16, device="auto", adapter_name=None, adapter_path=None, release_resources=True):
        """
        In this method, by releasing the resources of the old model and loading a new one,
        we ensure smooth transition and effective memory utilization.
        """
        
        if release_resources and self.model is not None:
            self.release_resources()
        self.load_model(model_name, dtype=dtype, device=device, adapter_name=adapter_name, adapter_path=adapter_path)
        self.current_model_name = model_name

    def release_resources(self):
        
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() 
        self.model = None
        self.tokenizer = None
        self.current_infer_func = None
        print(f"Resources for model {self.current_model_name} released.")
        self.current_model_name = None
    
    def save_model(self,output_path):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("save model and tokenizer successfully")
        
    def do_eval(self,custom_name,normal_eval=True,physic_eval=True,topo_eval=True,sys_content = None,save_format='json',save_path=''):
        """
        Evaluate the model from three domains: general, physics, and topology, 
        and save the results to the corresponding CSV files
        """

        if normal_eval:
            self.get_eval_result(custom_name,self.normal_eval_csv,sys_content,save_format,save_path)
        if physic_eval:
            self.get_eval_result(custom_name,self.physic_eval_csv,sys_content,save_format,save_path)
        if topo_eval:
            self.get_eval_result(custom_name,self.topo_eval_csv,sys_content,save_format,save_path)

    def get_eval_result(self,custom_name,csv_path,sys_content,save_format='json',save_path=''):
        ## Note that custom_name is the user-defined name
        eval_pd = pd.read_csv(csv_path)
        ## Iterate over each question in the CSV
        question_list = []
        answer_list =[]
        for index, row in tqdm(eval_pd.iterrows(), total=eval_pd.shape[0]):
            question = row['question']
            question_list.append(question)
            curr_answer= self.model_infer(question,sys_content= sys_content,max_new_tokens=512)
            answer_list.append(curr_answer)
        if save_format == 'json':
            data_inform = [{"question": q, "answer": a} for q, a in zip(question_list, answer_list)]
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(data_inform, f, indent=4, ensure_ascii=False) 
            else:
                default_path = os.path.join(os.path.dirname(self.physic_eval_csv),custom_name)
                with open(default_path, 'w') as f:
                    json.dump(data_inform, f, indent=4, ensure_ascii=False) 
                  
        elif save_format == 'csv':    
            eval_pd[custom_name] = answer_list
            eval_pd.to_csv(csv_path,index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_format}") 
        
model_manager = ModelInfer()

## Call the instance's __call__ method to directly register the corresponding inference function
## The registration name must match the one defined in model_type
@model_manager("chatglm")
def chatglm_infer(model,tokenizer, input_data, **kwargs):
    response, history = model.chat(tokenizer, input_data,history=[],**kwargs)
    return response

@model_manager("qwen1.5")
def qwen_1half_infer(model, tokenizer, input_data, **kwargs):
    # 准备当前的系统和用户消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_data}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        **kwargs  
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

@model_manager("phi2")
def phi_2_infer(model, tokenizer, input_data, **kwargs):
    inputs = tokenizer(input_data, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs,**kwargs)
    response = tokenizer.batch_decode(outputs)[0]
    return response 

@model_manager("gpt3.5")
def gpt3_5_infer(model, tokenizer, input_data, **kwargs):
    completion = model.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_data}
        ]
    )
    response = completion.choices[0].message.content  
    return response

@model_manager("gpt4")
def gpt4_infer(model, tokenizer, input_data, **kwargs):
    completion = model.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_data}
        ]
    )
    response = completion.choices[0].message.content  
    return response

@model_manager("yi")
def yi_infer(model, tokenizer, input_data, **kwargs):
    messages = [
        {"role": "user", "content": input_data}
    ]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to(model.device))
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

@model_manager("llama2")
def llama2_infer(model, tokenizer, input_data, sys_content = None,eval_mode=False,**kwargs):

    if sys_content:
        messages = [
            {
                "role": "system",
                "content": sys_content,
            },
            {"role": "user", "content": input_data},
        ]
    else:
        messages = [{"role": "user", "content": input_data}]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    
    if eval_mode:
        # Generate output and get scores
        generation_output = model.generate(input_ids.to(model.device), output_logits=True, output_scores=True, 
                            return_dict_in_generate = True, **kwargs)

        # Extract generated output ids and scores
        output_ids = generation_output.sequences
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        logits = generation_output.scores[0][0]

        return response, logits
    
    else:
         # Generate output without getting scores
        output_ids = model.generate(input_ids.to(model.device),**kwargs)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response


@model_manager("llama3")
def llama3_infer(model, tokenizer, input_data, sys_content = None,eval_mode=False, **kwargs):
    if sys_content:
        messages = [
            {
                "role": "system",
                "content": sys_content,
            },
            {"role": "user", "content": input_data},
        ]
    else:
        messages = [
            {"role": "user", "content": input_data},
        ]
    
    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
    ).to(model.device)

    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    
    if eval_mode:


        outputs = model.generate(
            input_ids.to(model.device),
            output_logits=True,
            output_scores=True,
            return_dict_in_generate = True,
            eos_token_id=terminators,
            **kwargs
        )
        logits= outputs.logits[0][0]
        response = tokenizer.decode(outputs[0][0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response,logits 
    
    else:
        output_ids = model.generate(input_ids.to(model.device), eos_token_id=terminators,**kwargs)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

@model_manager("chatglm4")
def chatglm4_infer(model,tokenizer, input_data, **kwargs):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": input_data}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response 



