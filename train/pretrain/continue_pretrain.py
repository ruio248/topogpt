import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import evaluate
import torch
from datasets import load_from_disk, concatenate_datasets
import transformers
from peft import LoraConfig, get_peft_model
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.testing_utils import CaptureLogger
from torch.nn.utils.rnn import pad_sequence

@dataclass
class PretrainConfig:
    specific_data_dir: str = field(metadata={"help": "The directory of the specific data"})
    normal_data_dir: str = field(metadata={"help": "The directory of the normal data"})
    model_path: str = field(metadata={"help": "The initial model path for training"})
    log_directory: str = field(metadata={"help": "Directory for log information"})
    training_config: str = field(metadata={"help": "The filepath of training config"})
    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "The initial tokenizer path"})
    model_type: Optional[str] = field(default=None, metadata={"help": "Specify if model is llama2"})
    device_map: str = field(default="cuda:0", metadata={"help": "Where to load the model"})
    streaming: bool = field(default=False, metadata={"help": "Whether to use streaming technology"})
    chunk_size: int = field(default=2048, metadata={"help": "The chunk size of input"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to perform evaluation"})
    split_ratio: float = field(default=0.9, metadata={"help": "split ratio between train and eval"})
    num_proc: int = field(default=4, metadata={"help": "The number of processes when loading data"})
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA for fine-tuning"})
    lora_config: Optional[str] = field(default=None, metadata={"help": "The filepath of LoRA config"})
    need_parallel: bool = field(default=False, metadata={"help": "Whether to parallelize the training process"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Max train samples for evaluation"})

class PretrainPipeline:
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.logger = self.setup_logger(config.log_directory)

    def setup_logger(self, log_directory, log_level=logging.INFO):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(f"{log_directory}/training.log")
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
 
    def save_length_distribution_plot(self, tokenized_lengths, save_path):
        # 取出长度信息
        lengths = tokenized_lengths["text"]

        # 计算最大值和平均值
        max_length = max(lengths)
        avg_length = np.mean(lengths)

        # 绘制长度分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, color='blue', edgecolor='black')
        plt.title("Text Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 保存图像到指定路径
        plt.savefig(os.path.join(save_path, "length_distribution.png"))
        plt.close()

        # 返回最大值和平均值
        return max_length

    def load_data(self):
        raw_specific_datasets = load_from_disk(self.config.specific_data_dir)
        raw_normal_datasets =  load_from_disk(self.config.normal_data_dir)
        # 查看 text 列的最大长度
        max_text_length = self.save_length_distribution_plot(raw_normal_datasets, self.tokenizer)
        print(f"The maximum length and average length of text in tokens is: {max_text_length}")
        raw_datasets = concatenate_datasets([raw_specific_datasets, raw_normal_datasets])
        shuffled_datasets = raw_datasets.shuffle(seed=42)  # 打乱数据
        shuffled_datasets = shuffled_datasets.select(range(100))  # 选择100个样本用于测试
        self.logger.info("Data loaded successfully")
        return shuffled_datasets

    def load_model_and_tokenizer(self):
        if self.config.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                torch_dtype=torch.half, device_map="cuda:0"
            )
            self.logger.info("Model loaded successfully")
        else:
            self.logger.error("Model path not specified")
            raise ValueError("Please input a path related to the model.")

        if self.config.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self.logger.info("Tokenizer loaded successfully")

        if self.config.model_type == 'llama2':
            self.tokenizer.pad_token_id = 2
        elif self.config.model_type == 'llama3':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"Embedding size: {embedding_size}")

        if self.config.lora_config:
            with open(self.config.lora_config, "r") as config_file:
                lora_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            lora_config = LoraConfig(**lora_config_content)
            self.model = get_peft_model(self.model, lora_config)

    def tokenize_function(self, examples):
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        with CaptureLogger(tok_logger) as cl:
            output = self.tokenizer(examples["text"], padding=False, truncation=True,max_length=2048)  # 禁用自动padding
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    def prepare_datasets(self, raw_datasets):
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            #num_proc=self.config.num_proc,
            remove_columns=raw_datasets.column_names,
            desc="Running tokenizer on dataset",
        )
        return tokenized_datasets

    @dataclass
    class DataCollatorForPretrainDataset:
        """自定义的DataCollator，用于处理填充逻辑。"""

        tokenizer: transformers.PreTrainedTokenizer
        padding_value: int = -100  # 忽略计算损失的填充值

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
            labels = [torch.tensor(instance["input_ids"]) for instance in instances]

            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.padding_value)

            return dict(
                input_ids=input_ids_padded,
                labels=labels_padded,
                attention_mask=input_ids_padded.ne(self.tokenizer.pad_token_id),  # 创建attention mask
            )

    def make_pretrain_data_module(self, raw_datasets):
        """构建训练数据集和自定义collator。"""
        tokenized_ds = self.prepare_datasets(raw_datasets)
        data_collator = self.DataCollatorForPretrainDataset(tokenizer=self.tokenizer)
        if self.config.do_eval:
            split_datasets = tokenized_ds.train_test_split(train_size=self.config.split_ratio)
            train_dataset = split_datasets['train']
            eval_dataset = split_datasets['test']
        else:
            train_dataset = tokenized_ds
            eval_dataset = None
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    def setup_training_args(self):
        with open(self.config.training_config, "r") as config_file:
            train_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            if self.config.need_parallel and 'deepspeed' not in train_config_content:
                raise ValueError("If parallelism is required, please specify the configuration path of deepspeed_config in training_config.")
        train_config_content['learning_rate'] = float(train_config_content['learning_rate'])
        train_config_content['adam_epsilon'] = float(train_config_content['adam_epsilon'])
        self.training_args = TrainingArguments(**train_config_content)

    def train(self):
        self.load_model_and_tokenizer()
        shuffled_datasets = self.load_data() 
        data_module = self.make_pretrain_data_module(shuffled_datasets)
        self.setup_training_args()

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            **data_module,
        )

        try:
            train_result = trainer.train()
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            raise

        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = (
            self.config.max_train_samples if self.config.max_train_samples is not None else len(data_module["train_dataset"])
        )
        metrics["train_samples"] = min(max_train_samples, len(data_module["train_dataset"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def run(self):
        self.train()

if __name__ == "__main__":
    config = PretrainConfig(
        specific_data_dir="/work/ruioliao/topo_agent/data/pretrain_data_demo/train_data/specific",
        normal_data_dir="/work/ruioliao/topo_agent/data/pretrain_data_demo/train_data/general",
        model_path="/work/HUGGINGFACE/model/Llama3-8B-HF",
        tokenizer_path="/work/HUGGINGFACE/model/Llama3-8B-HF",
        model_type="llama3",
        device_map="cuda:6",
        streaming=False,
        chunk_size=2048,
        do_eval=False,
        num_proc=4,
        use_lora=True,
        log_directory="/work/ruioliao/topo_agent/train/pretrain/log",
        training_config="/work/ruioliao/topo_agent/train/pretrain/train_config/llama3_8b_train.yaml",
        lora_config="/work/ruioliao/topo_agent/train/pretrain/train_config/llama3_8b_lora.yaml",
        need_parallel=False,
    )

    pipeline = PretrainPipeline(config)
    pipeline.run()



##FIXME: 多进程map不动 
##TODO： 分组长度采样器LengthGroupedSampler
##TODO： 还是得打成一个一个的chunk，要不然单卡lora 都训练不动,多看看预训练

## 查看最大长度是多少，并行训练配置等

