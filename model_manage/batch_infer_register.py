"""
This file encapsulates the inference logic for multiple models.
It leverages Python's instance-level registration mechanism to 
dynamically register inference functions. Models can be loaded, 
switched, and used seamlessly for inference. The inference functions 
are registered for each model type using a decorator, enabling 
flexible management and execution of model-specific inference tasks.
"""

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import PeftModel
import torch
import torch.nn.functional as F
import ray
import time
from model_inform import (
    MODEL_ROOT_DIR,
    NAME_REPATH_TYPE_DICT_CL,
)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA current device:", torch.cuda.current_device())



class ModelInference:
    model_root_dir = MODEL_ROOT_DIR
    name_repath_type_dict =NAME_REPATH_TYPE_DICT_CL 

    def __init__(self):
        self._infer_dict = {}
        self.model = None
        self.tokenizer = None
        self.current_infer_func = None
        self.current_model_name = None

    def __call__(self, target):
        return self.infer_register(target)
    
    def infer_register(self, target):
        def add_infer(infer_name, infer_func):
            if not callable(infer_func):
                raise Exception(f"Error:{infer_func} must be callable!")
            if infer_name in self._infer_dict:
                print(f"\033[31mWarning:\033[0m {infer_func.__name__} already exists and will be overwritten!")
            self._infer_dict[infer_name] = infer_func
            return infer_func
        return lambda x: add_infer(target, x)  

    def load_model(self, infer_type: str, model_name: str, dtype=torch.float16, device="auto", adapter_name: str = None, adapter_path: str = None,gpu_memory_utilization=0.8):
        if infer_type not in ["vllm", "hf"]:
            raise ValueError("Invalid infer_type. Allowed values are 'vllm' or 'hf'.")
        
        if model_name not in self.name_repath_type_dict:
            raise ValueError(f"Model name '{model_name}' is not recognized.")
        
        model_path = os.path.join(self.model_root_dir, self.name_repath_type_dict[model_name]["path"])
        model_type = self.name_repath_type_dict[model_name]['type']           
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if infer_type == "hf":
            self.current_infer_func = self._infer_dict["hf_" + model_type]
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
            print(f"Loaded {model_name} successfully, model on {self.model.device}")

            if adapter_name and adapter_path:
                self.model = PeftModel.from_pretrained(self.model, model_id=adapter_path, adapter_name=adapter_name)
                self.model = self.model.merge_and_unload()
                print(f"Loaded {adapter_name} for {model_name} successfully")

        else:
            self.current_infer_func = self._infer_dict["vllm_" + model_type]
            if device != "auto":
                device_index = device.split(":")[1]
                #os.environ["CUDA_VISIBLE_DEVICES"] = device_index

            if adapter_name and adapter_path:
                if model_name == "llama3_70b":  
                    ray.init(num_cpus=4, num_gpus=2) 
                    self.model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=2,enable_lora=True)
                    print(f"Loaded {adapter_name} for {model_name} successfully")
                else:
                    self.model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=1,enable_lora=True)
                    print(f"Loaded {adapter_name} for {model_name} successfully")
            else:
                if model_name == "llama3_70b": 
                    ray.init(num_cpus=4, num_gpus=2) 
                    self.model = LLM(model=model_path,block_size=8, gpu_memory_utilization=gpu_memory_utilization,tensor_parallel_size=2)
                else:
                    self.model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization,tensor_parallel_size=1,trust_remote_code=True)

    def model_infer(self, input_data: list, **kwargs):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Please load a model first.")

        response = self.current_infer_func(self.model, self.tokenizer, input_data, **kwargs)
        return response 

    def get_max_token(self, input_data):
        tokens = [self.tokenizer.encode(data, add_special_tokens=True) for data in input_data]
        max_tokens = max(len(t) for t in tokens)
        return max_tokens

    def switch_model(self, infer_type: str, model_name: str, dtype=torch.float16, device="auto",
                     adapter_name: str = None, adapter_path: str = None, release_resources=True):
        if release_resources and self.model is not None:
            self.release_resources()
        self.load_model(infer_type, model_name, dtype, device, adapter_name, adapter_path)
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

batch_manager = ModelInference()

@batch_manager("hf_llama2")
def hf_llama2_infer(model, tokenizer, input_data,**kwargs):
    input_tensors = []
    attention_tensors = []
    tensors_len = []
    for k in range(len(input_data)):
        messages = [{"role": "user", "content":input_data[k]}]
        input_tensor = tokenizer.apply_chat_template(conversation=messages, tokenize=True, 
                                                     add_generation_prompt=True, return_tensors='pt')
        input_tensors.append(input_tensor)
        tensors_len.append(input_tensor.shape[1])

    max_len = max(tensors_len)
    pad_num = [max_len - l for l in tensors_len]
    for k in range(len(input_tensors)):
        input_tensor = input_tensors[k]
        padded_tensor = F.pad(input_tensor, (pad_num[k], 0, 0, 0), value=tokenizer.bos_token_id)
        input_tensors[k] = padded_tensor

        attention = [0]*pad_num[k]
        attention.extend([1]*tensors_len[k])
        attention = torch.Tensor(attention).unsqueeze(0)

        attention_tensors.append(attention)

    last_input = torch.cat(input_tensors,dim=0)
    attention_all = torch.cat(attention_tensors,dim=0)

    output_ids = model.generate(last_input.to(model.device), pad_token_id=tokenizer.bos_token_id, 
                                attention_mask=attention_all.to(model.device),**kwargs) #do_sample=False)
    # response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    responses = tokenizer.batch_decode(output_ids[:,last_input.shape[1]:], skip_special_tokens=True)
    return responses

@batch_manager("hf_llama3")
def hf_llama3_infer(model, tokenizer, input_data,**kwargs):
    # Start tracking total time
    total_start_time = time.time()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Prepare inputs for HF
    messages = [[{"role": "user", "content": content}] for content in input_data]
    texts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(texts, padding="longest", return_tensors="pt")
    inputs = {key: val.cuda() for key, val in inputs.items()}

    gen_tokens = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        **kwargs
    )
    # Collect HF results
    responses  = tokenizer.batch_decode(gen_tokens[::,len(inputs['input_ids'][0]):], skip_special_tokens=True)
        # Calculate total time
    total_end_time = time.time()
    total_time_spent = total_end_time - total_start_time
    print(f"Total inference time: {total_time_spent:.2f} seconds")
    return responses 

@batch_manager("hf_chatglm4")
def hf_llama3_infer(model, tokenizer, input_data,**kwargs):
    # Start tracking total time
    total_start_time = time.time()
    tokenizer.pad_token_id = tokenizer.eos_token_id

    format_messages = ["[gMASK]<sop><|user|>\n{}<|assistant|>".format(text) for text in input_data]
    inputs = tokenizer(format_messages,padding="longest", return_tensors="pt")
    
    gen_tokens = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs
    )
   
    # Collect HF results
    responses  = tokenizer.batch_decode(gen_tokens[::,len(inputs['input_ids'][0]):], skip_special_tokens=True)
    
    # Calculate total time
    total_end_time = time.time()
    total_time_spent = total_end_time - total_start_time
    print(f"Total inference time: {total_time_spent:.2f} seconds")
    return responses 

@batch_manager("vllm_llama2")
def vllm_llama2_infer(model, tokenizer, input_data, **kwargs):
    sampling_params = SamplingParams(**{k: v for k, v in kwargs.items() if k != 'adapter_path'})
    adapter_path = kwargs.get('adapter_path')
    format_input_data = [f"[INST] {input} [/INST]" for input in input_data]

    if not adapter_path:
        outputs = model.generate(format_input_data, sampling_params)
    else:
        outputs = model.generate(format_input_data, sampling_params, LoRARequest("adapter", 1, adapter_path))
    answer_list = [output.outputs[0].text for output in outputs]
    return answer_list

@batch_manager("vllm_yi")
def vllm_yi_infer(model, tokenizer, input_data, **kwargs):
    sampling_params = SamplingParams(**{k: v for k, v in kwargs.items() if k != 'adapter_path'})
    adapter_path = kwargs.get('adapter_path')
    format_input_data = [f"user\n{input} \n" for input in input_data]

    if not adapter_path:
        outputs = model.generate(format_input_data, sampling_params)
    else:
        outputs = model.generate(format_input_data, sampling_params, LoRARequest("adapter", 1, adapter_path))
    answer_list = [output.outputs[0].text for output in outputs]
    return answer_list

@batch_manager("vllm_chatglm4")
def vllm_chatglm4_infer(model, tokenizer, input_data, **kwargs):
    sampling_params = SamplingParams(**{k: v for k, v in kwargs.items() if k != 'adapter_path'})
    adapter_path = kwargs.get('adapter_path')
    format_input_data = input_data

    if not adapter_path:
        outputs = model.generate(format_input_data, sampling_params)
    else:
        outputs = model.generate(format_input_data, sampling_params, LoRARequest("adapter", 1, adapter_path))
    answer_list = [output.outputs[0].text for output in outputs]
    return answer_list


@batch_manager("vllm_llama3")
def vllm_llama3_infer(model, tokenizer, input_data,**kwargs):
   # Set up sampling parameters, excluding 'lora_request' from kwargs
    star_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    kwargs['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(**{k: v for k, v in kwargs.items() if k != 'adapter_path'})

    # Check if lora_request exists in kwargs and set it if available
    adapter_path = kwargs.get('adapter_path')
  
    format_input_data = [f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>' for input in input_data]
    if not adapter_path:
    # Generate outputs using the model with the specified sampling parameters
        outputs = model.generate(format_input_data, sampling_params)
    else:
        print(f"load adapter from {adapter_path} successfully")
        outputs = model.generate(format_input_data, sampling_params,LoRARequest("adapter", 1, \
        adapter_path))
    # Collect and return the generated texts
    answer_list = []
    for output in outputs:
        generated_text = output.outputs[0].text
        if generated_text.startswith(star_string):
            generated_text = generated_text[len(star_string):]
        answer_list.append(generated_text)
    
    return answer_list



