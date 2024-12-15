### 这个文件定义我们怎样得到训练所需要的process_func，也就是怎么得到每个样本input_ids,attention_mask，和labels
from instruct_template import STANDARD_ALPHACHA,NORMAL_TEMPL

class Process_Func:
    
    @classmethod
    def sft_llama2(cls, example, tokenizer,template_type,max_length =4096):
        ##  template_type 类型为alphacha,normal 
        if template_type == 'alphacha':
            prompt_template =STANDARD_ALPHACHA
        else:
            prompt_template =NORMAL_TEMPL
        
        ## 如果有input这个键且不为空 
        if example["input"]:
            input_text = prompt_template['with_input'].format(Instruction=example['instruction'], input=example['input'])

        else:
            input_text = prompt_template['without_input'].format(Instruction=example['instruction'])
    
        ## 调用build_chat_input方法得到模型的输入格式并进行token话
        messages = [{"role": "user", "content": input_text}]
        tokenized_input = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        response = tokenizer(example["output"],add_special_tokens=False)
       
        ## 得到input_dis，并根据相应的长度，对labels进行设置
        input_ids = tokenized_input["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        labels = [-100] * len(tokenized_input["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

        ## 得到attention_mask
        attention_mask = tokenized_input["attention_mask"] + response["attention_mask"] + [1]
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    

    @classmethod
    def sft_chatglm3(cls, example, tokenizer,template_type,max_length =4096):
        if template_type == 'alphacha':
            prompt_template =STANDARD_ALPHACHA
        else:
            prompt_template =NORMAL_TEMPL
        
        ## 如果有input这个键且不为空 
        if example["input"]:
            input_text = prompt_template['with_input'].format(Instruction=example['instruction'], input=example['input'])

        else:
            input_text = prompt_template['without_input'].format(Instruction=example['instruction'])
        
        tokenized_input=tokenizer.build_chat_input(input_text, history=[], role="user")
        response = tokenizer(example["output"],add_special_tokens=False)
        
        input_ids =tokenized_input["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = tokenized_input["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]
        labels = [-100] * len(tokenized_input["input_ids"][0].numpy().tolist()) + response["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    @classmethod
    def sft_llama3(cls, example, tokenizer,template_type,max_length =8192,sys_content=None):
        ##  template_type 类型为alphacha,normal 
        if template_type == 'alphacha':
            prompt_template =STANDARD_ALPHACHA
        else:
            prompt_template =NORMAL_TEMPL
        
        ## 如果有input这个键且不为空 
        if example["input"]:
            input_text = prompt_template['with_input'].format(Instruction=example['instruction'], input=example['input'])

        else:
            input_text = prompt_template['without_input'].format(Instruction=example['instruction'])
    
        ## 调用build_chat_input方法得到模型的输入格式并进行token话
        if sys_content:
            input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
       
        else:
            input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenized_input = tokenizer(input)
        response = tokenizer(example["output"],add_special_tokens=False)
       
        ## 得到input_dis，并根据相应的长度，对labels进行设置
        input_ids = tokenized_input["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        labels = [-100] * len(tokenized_input["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

        ## 得到attention_mask
        attention_mask = tokenized_input["attention_mask"] + response["attention_mask"] + [1]
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    @classmethod
    def sft_qwen(cls, example, tokenizer,template_type,max_length =4096):
        pass

## TODO：加入qwen1.5的token函数，修改llama2的微调代码，如[INST] tell a story about physics [/INST] 
## messages = [{"role": "user", "content": input_data}]
## input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
 

'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in the field of topology. Please answer the questions posed by users from a professional perspective.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nwhat is Bi2Se3<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nwhat is Bi2Se3<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

#chatglm4 
'[gMASK] <sop> <|user|> \n你是谁? <|assistant|>'
'[gMASK] <sop> <|system|> \n你是拓扑领域的专家 <|user|> \n你是谁 <|assistant|>'

## 有无system prompt 应该怎么改

