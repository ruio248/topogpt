import os
import torch
import evaluate
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import yaml
from topo_agent.train.rlhf.train.reward_collator import RewardDataCollatorWithPadding

## TODO:完善这一部分，类型提示
@dataclass
class RewardTrainingConfig:
    input_directory: str = field(metadata={"help": "The directory of the reward data"})
    model_path: str = field(metadata={"help": "The model path for training"})
    model_type: str = field(metadata={"help": "The model type training"})
    train_config: str = field(metadata={"help": "The filepath of training config"})
    lora_config: str = field(metadata={"help": "The filepath of lora config"})
    need_parallel: bool = field(default=False, metadata={"help": "Whether you need to parallel the training process"})
    split_ratio: float = field(default=0.8, metadata={"help": "The ratio to split dataset"})

class RewardTrainingPipeline:
    def __init__(self, config: RewardTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_args = None

    def preprocess_function(self, examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for question, answer_1, answer_2,choice in zip(examples["instruction"], examples["answer_1"],
                                                    examples["answer_2"],examples["choice"]):
            if choice  == "answer_1":
                response_j = answer_1
                response_k = answer_2
            else:
                response_j = answer_2
                response_k = answer_2
                
            tokenized_j = self.tokenizer(question + response_j, truncation=True)
            tokenized_k = self.tokenizer(question + response_k, truncation=True)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

        return new_examples

    def make_supervised_data_module(self, tokenized_ds):
        data_collator = RewardDataCollatorWithPadding(tokenizer=self.tokenizer, max_length=self.config.max_new_tokens)
        split_datasets = tokenized_ds.train_test_split(train_size=self.config.split_ratio)
        train_dataset = split_datasets['train']
        test_dataset = split_datasets['test']
        return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)

    def compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, _ = eval_pred
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
            rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss

    def load_datasets(self):
        reward_datasets = load_dataset('json', data_dir=self.config.input_directory, split="train")
        return reward_datasets

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)

        if self.config.model_type == 'llama2':
            self.tokenizer.pad_token_id = 2
        elif self.config.model_type == 'llama3':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path, num_labels=1, torch_dtype=torch.bfloat16
        )

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
        reward_datasets = self.load_datasets()
        tokenized_datasets = reward_datasets.map(
            self.preprocess_function, batched=True, num_proc=24, remove_columns=reward_datasets.column_names
        )
        data_module = self.make_supervised_data_module(tokenized_datasets)

        trainer = self.RewardTrainer(
            model=self.model,
            args=self.training_args,
            **data_module,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=self.training_args.output_dir)
        
        print("Saving last checkpoint of the model")
        self.model.save_pretrained(self.training_args.output_dir + "peft_last_checkpoint")

    def run(self):
        self.load_model_and_tokenizer()
        self.setup_training_args()
        self.train()

if __name__ == "__main__":
    config = RewardTrainingConfig(
        input_directory='/path/to/input_directory',
        model_path='/path/to/model_path',
        model_type='llama3',
        train_config='/path/to/train_config',
        lora_config='/path/to/lora_config',
        need_parallel=False,
        split_ratio=0.8
    )
    pipeline = RewardTrainingPipeline(config)
    pipeline.run()
