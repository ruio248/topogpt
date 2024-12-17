"""
This file encapsulates the inference code used by various models.
It utilizes Python's class-level registration mechanism to allow 
dynamic inference function registration. By using a class method 
`infer_register`, the inference functions are registered for 
different models. The models can be easily loaded, switched, and 
used for inference in a flexible manner, with batch processing 
capabilities for handling multiple inputs at once. The decorator 
approach ensures that the inference functions are dynamically 
added and overwritten when necessary, based on the model type, 
while enabling efficient batch inference across different model 
architectures.
"""


import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from peft import PeftModel
from batch_infer_register import ModelInference 
import torch
import torch.nn.functional as F
from model_inform import (
    MODEL_ROOT_DIR,
    NAME_REPATH_TYPE_DICT_CL,
)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA current device:", torch.cuda.current_device())



class ModelInfer_Cl(ModelInference ):
    model_root_dir = MODEL_ROOT_DIR
    name_repath_type_dict =NAME_REPATH_TYPE_DICT_CL 

    def __init__(self):
        super()
        self._infer_dict = {}
        self.model = None
        self.tokenizer = None
        self.current_infer_func = None
        self.current_model_name = None

    @classmethod
    def infer_register(cls, model_type):
        def decorator(infer_func):
            if model_type not in cls.model_type:
                raise ValueError(f"Model type '{model_type}' not recognized.")
            if model_type in cls._infer_dict:
                print(f"Warning: Overwriting existing infer method for '{model_type}'.")
            cls._infer_dict[model_type] = infer_func
            return infer_func
        return decorator       
    
@ModelInfer_Cl.infer_register("hf_llama2")
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

## 这里写成model_name 可能不合理
@ModelInfer_Cl.infer_register("vllm_llama2")
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

@ModelInfer_Cl.infer_register("vllm_yi")
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

@ModelInfer_Cl.infer_register("vllm_llama3")
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

