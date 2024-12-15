## 此脚本提供指令微调lora版本 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import random 
import numpy as np 
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from process_func import Process_Func
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
#from .callback.custom_callback import CustomTrainerCallback
def set_seed(seed: int):
    # Python 随机数种子
    random.seed(seed)
    
    # Numpy 随机数种子
    np.random.seed(seed)
    
    # PyTorch 随机数种子
    torch.manual_seed(seed)
    
    # cuda 随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        
    # 确保CUDNN库操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在代码开始时设置一个固定的种子
set_seed(42)

@dataclass
class SftConfig:
    input_directory: str = field(metadata={"help": "The directory of the specific data"})
    model_path: str = field(metadata={"help": "The model path for training"})
    model_type: str = field(metadata={"help": "The model type training"})
    train_config: str = field(metadata={"help": "The filepath of training config"})
    lora_config: str = field(default=None,metadata={"help": "The filepath of lora config"})
    template_type: str = field(default="normal", metadata={"help": "The device to load the model"})
    need_parallel: bool = field(default=False, metadata={"help": "Whether you need to parallel the training process"})


class SftPipeline:
    def __init__(self, config: SftConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_args = None

    def preprocess_function(self, examples):
        if self.config.model_type == "chatglm3":
            return Process_Func.sft_chatglm3(examples, self.tokenizer, self.config.template_type)
        elif self.config.model_type == "llama2":
            return Process_Func.sft_llama2(examples, self.tokenizer, self.config.template_type)
        elif self.config.model_type == "mistral":
            return Process_Func.sft_llama2(examples, self.tokenizer, self.config.template_type)
        elif self.config.model_type == "yi":
            return Process_Func.sft_llama2(examples, self.tokenizer, self.config.template_type)
        elif self.config.model_type == "qwen":
            return Process_Func.sft_qwen(examples, self.tokenizer, self.config.template_type)
        elif self.config.model_type == "llama3":
            return Process_Func.sft_llama3(examples, self.tokenizer, self.config.template_type)
        else:
            raise ValueError("Input model type error: The argument passed is incorrect.")

    def make_supervised_data_module(self, tokenized_ds):
        data_collator = self.DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        return dict(train_dataset=tokenized_ds, eval_dataset=None, data_collator=data_collator)

    @dataclass
    class DataCollatorForSupervisedDataset:
        tokenizer: AutoTokenizer

        def __call__(self, examples):
            input_ids, labels = tuple([example[key] for example in examples] for key in ("input_ids", "labels"))
            input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
            labels = [torch.tensor(lbs, dtype=torch.long) for lbs in labels]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

    def load_datasets(self):
        ## 关于streaming 是否为True的问题
        instruct_datasets = load_dataset('json', data_dir=self.config.input_directory, split="train",streaming=False)
        return instruct_datasets

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)

        if self.config.model_type == 'llama2':
            self.tokenizer.pad_token_id = 2
        elif self.config.model_type == 'llama3':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, low_cpu_mem_usage=True, trust_remote_code=True, torch_dtype=torch.half, device_map="cuda:0"
        )
        
        if self.config.lora_config:
            with open(self.config.lora_config, "r") as config_file:
                lora_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            lora_config = LoraConfig(**lora_config_content)
            self.model = get_peft_model(self.model, lora_config)

    def setup_training_args(self):
        with open(self.config.train_config, "r") as config_file:
            train_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            if self.config.need_parallel and 'deepspeed' not in train_config_content:
                raise ValueError("If parallelism is required, please specify the configuration path of deepspeed_config in training_config.")
        train_config_content['learning_rate'] = float(train_config_content['learning_rate'])
        train_config_content['adam_epsilon'] = float(train_config_content['adam_epsilon'])

        self.training_args = TrainingArguments(**train_config_content)

    def train(self):
        instruct_datasets = self.load_datasets()
        ## 取2000 个样本做测试
        #instruct_datasets = instruct_datasets.select(range(2000))
        tokenized_datasets = instruct_datasets.map(
            self.preprocess_function, remove_columns=instruct_datasets.column_names 
        )
        data_module = self.make_supervised_data_module(tokenized_datasets)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            **data_module,
        )
        
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=self.training_args.output_dir)

    def run(self):
        self.load_model_and_tokenizer()
        self.setup_training_args()
        self.train()

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Script for LoRA SFT training")
    parser.add_argument("--input_directory", type=str, required=True, help="The directory of the specific data")
    parser.add_argument("--model_path", type=str, required=True, help="The model path for training")
    parser.add_argument("--model_type", type=str, required=True, choices=["llama2", "llama3", "chatglm3", "qwen", "mistral", "yi"], help="The model type")
    parser.add_argument("--train_config", type=str, required=True, help="The filepath of training config")
    parser.add_argument("--lora_config", type=str, required=False, help="The filepath of lora config")
    parser.add_argument("--template_type", type=str, default="normal", help="The device to load the model")
    parser.add_argument("--need_parallel", action='store_true', help="Whether you need to parallel the training process")


    args = parser.parse_args()

    config = SftConfig(
        input_directory=args.input_directory,
        model_path=args.model_path,
        model_type=args.model_type,
        train_config=args.train_config,
        lora_config=args.lora_config,
        template_type=args.template_type,
        need_parallel=args.need_parallel,
    )

    pipeline = SftPipeline(config)
    pipeline.run()
    """

    config = SftConfig(
        input_directory="/data/work/ruioliao/topo_agent/data/topo_qa_test/format_tune_a_2",
        model_path="/data/work/HUGGINGFACE/model/Llama3-8B-Instruct-HF",
        model_type="llama3",
        need_parallel=False,
        train_config="/data/work/ruioliao/topo_agent/train/sft/train_config/format_tune_a/llama3_train_config.yaml",
        lora_config = "/data/work/ruioliao/topo_agent/train/sft/train_config/lora_llama3_config.yaml" 
    )
    
    pipeline = SftPipeline(config)
    pipeline.run()
    

## 在tensorboard 加入测试指标
##  这里是不是没加eval 进行评估的步骤

# /work/ruioliao/topo_agent/data/api_qa_demo/regression_data/q_8_a_5
## 参数量太大的话会抱以下错误
"""
[2024-09-21 14:30:55,673] [ERROR] [launch.py:322:sigkill_handler] ['/home/ruioliao/anaconda3/envs/vllm/bin/python', '-u', '/work/ruioliao/topo_agent/train/sft/sft.py', '--local_rank=4'] exits with return code = 1
Error in atexit._run_exitfuncs:
Traceback (most recent call last):
  File "/home/ruioliao/anaconda3/envs/vllm/lib/python3.9/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 96, in put
    with open(self.file_path + ".tmp", 'wb') as handle:
OSError: [Errno 122] Disk quota exceeded: '/home/ruioliao/.triton/autotune/Fp16Matmul_2d_kernel.pickle.tmp'
Error in atexit._run_exitfuncs:
Traceback (most recent call last):
  File "/home/ruioliao/anaconda3/envs/vllm/lib/python3.9/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 96, in put
    with open(self.file_path + ".tmp", 'wb') as handle:
OSError: [Errno 122] Disk quota exceeded: '/home/ruioliao/.triton/autotune/Fp16Matmul_2d_kernel.pickle.tmp'

"""
## todo: 加入ptuning的步骤, 测评, 回调函数(领域增强否)
## 