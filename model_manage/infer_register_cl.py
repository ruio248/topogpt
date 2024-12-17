## Register inference methods at the class level

import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import pandas as pd 
from tqdm import tqdm
import httpx
from openai import OpenAI
from infer_register import ModelInfer
from model_inform import (
    MODEL_ROOT_DIR,
    NAME_REPATH_TYPE_DICT,
    MODEL_TYPE,
    EVAL_PATH_DICT
)
class ModelInfer_Cl(ModelInfer):
    model_root_dir = MODEL_ROOT_DIR
    name_repath_type_dict = NAME_REPATH_TYPE_DICT
    model_type=MODEL_TYPE
    physic_eval_csv = EVAL_PATH_DICT['physic_eval_csv']
    normal_eval_csv = EVAL_PATH_DICT['normal_eval_csv']
    topo_eval_csv = EVAL_PATH_DICT['topo_eval_csv']
    _infer_dict = {}  

    def __init__(self, *args, **kwargs):
        self.model =None
        self.tokenizer = None 
        self.current_infer_func =None
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
   

@ModelInfer_Cl.infer_register("chatglm")
def chatglm_infer(model,tokenizer, input_data, **kwargs):
    response, history = model.chat(tokenizer, input_data,history=[],**kwargs)
    return response

@ModelInfer_Cl.infer_register("qwen1.5")
def qwen_1half_infer(model, tokenizer, input_data, **kwargs):
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

@ModelInfer_Cl.infer_register("phi2")
def phi_2_infer(model, tokenizer, input_data, **kwargs):
    inputs = tokenizer(input_data, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs,**kwargs)
    response = tokenizer.batch_decode(outputs)[0]
    return response 

@ModelInfer_Cl.infer_register("gpt3.5")
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

@ModelInfer_Cl.infer_register("gpt4")
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

@ModelInfer_Cl.infer_register("yi")
def yi_infer(model, tokenizer, input_data, **kwargs):
    messages = [
        {"role": "user", "content": input_data}
    ]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to(model.device))
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

@ModelInfer_Cl.infer_register("llama2")
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
        output_ids = model.generate(input_ids.to(model.device),**kwargs)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response

@ModelInfer_Cl.infer_register("llama3")
def llama3_infer(model, tokenizer, input_data, sys_content = None, **kwargs):
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

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        **kwargs
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

