import os
import torch
import random
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import is_main_process


# 固定随机数种子
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


@dataclass
class DPOConfig:
    input_directory: str = field(metadata={"help": "The directory of the specific data"})
    model_path: str = field(metadata={"help": "The model path for training"})
    ref_model_path: str = field(metadata={"help": "The reference model path for DPO"})  # 新增字段
    model_type: str = field(metadata={"help": "The model type for training"})
    dpo_config: str = field(metadata={"help": "The filepath of training config using DPO method"})
    lora_config: Optional[str] = field(default=None, metadata={"help": "The filepath of lora config if using LoRA"})
    need_parallel: bool = field(default=False, metadata={"help": "Whether to enable model parallelism"})


class DPOTrainerPipeline:
    def __init__(self, config: DPOConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.ref_model = None  # 新增字段用于存储参考模型
        self.training_args = None

    # 数据集加载
    def load_datasets(self):
        print("Loading dataset...")
        dataset = load_dataset('json', data_dir=self.config.input_directory, split="train", streaming=False)
        return dataset

    
    # 使用 map 函数格式化每个样本，生成 prompt、chosen 和 rejected 字段
    def format_sample(self,sample):
        single_data = {
            "prompt": sample["prompt"],  # 生成 Prompt
            "chosen": sample["response_choose"],  # 选择的 (更好) 的回答
            "rejected": sample["response_reject"]  # 被拒绝的 (较差) 的回答
        }
        return single_data

    # 加载模型与 tokenizer
    def load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")

        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        if self.config.model_type == 'llama2':
            self.tokenizer.pad_token_id = 2
        elif self.config.model_type == 'llama3':
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型 (model)
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if not self.config.need_parallel else "cuda:0"
        )

        # 加载参考模型 (ref_model)
        print("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.ref_model_path,  # 使用参考模型路径
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if not self.config.need_parallel else "cuda:0"
        )

        # 加载 LoRA 配置，如果存在
        if self.config.lora_config:
            print("Applying LoRA configuration...")
            with open(self.config.lora_config, "r") as config_file:
                lora_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            lora_config = LoraConfig(**lora_config_content)
            self.model = get_peft_model(self.model, lora_config)
            self.ref_model = get_peft_model(self.ref_model, lora_config)  # 对参考模型应用相同的 LoRA 配置

    # 设置训练参数
    def setup_training_args(self):
        print("Setting up training arguments...")
        with open(self.config.dpo_config, "r") as config_file:
            train_config_content = yaml.load(config_file, Loader=yaml.FullLoader)

        if self.config.need_parallel and 'deepspeed' not in train_config_content:
            raise ValueError("If parallelism is required, please include the deepspeed configuration in training_config.")
        
        # 确保某些数值参数的正确格式
        train_config_content['learning_rate'] = float(train_config_content['learning_rate'])
        train_config_content['adam_epsilon'] = float(train_config_content['adam_epsilon'])

        self.training_args = TrainingArguments(**train_config_content)

    # DPO 训练流程
    def train(self):
        print("Starting training...")
        dataset = self.load_datasets()
        original_columns = dataset.column_names
        dataset.map(
            self.format_sample,
            batched=True,
            remove_columns=original_columns
        )

        # 使用 DPOTrainer 进行训练
        trainer = DPOTrainer(
            model=self.model,  # 主模型
            model_ref=self.ref_model,  # 参考模型
            beta=0.1, # DPO 的温度超参
            args=self.training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )
      
        trainer.train()

        # 保存训练状态与模型
        trainer.save_state()
        trainer.save_model(output_dir=self.training_args.output_dir)
        print("Training completed and model saved.")

    # 启动训练流程
    def run(self):
        self.load_model_and_tokenizer()
        self.setup_training_args()
        self.train()


# 入口函数，用于启动 DPO 训练
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Pipeline")
    parser.add_argument("--input_directory", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--ref_model_path", type=str, required=True, help="Path to the reference model")  # 新增参数
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model (e.g., llama2, llama3)")
    parser.add_argument("--dpo_config", type=str, required=True, help="Path to the DPO training configuration file")
    parser.add_argument("--lora_config", type=str, default=None, help="Path to the LoRA configuration file")
    parser.add_argument("--need_parallel", action="store_true", help="Flag to enable parallel training")

    args = parser.parse_args()

    config = DPOConfig(
        input_directory=args.input_directory,
        model_path=args.model_path,
        ref_model_path=args.ref_model_path,  # 新增字段
        model_type=args.model_type,
        dpo_config=args.dpo_config,
        lora_config=args.lora_config,
        need_parallel=args.need_parallel
    )

    pipeline = DPOTrainerPipeline(config)
    pipeline.run()
