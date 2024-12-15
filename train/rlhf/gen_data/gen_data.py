## 此脚本用来生成强化学习所需要标注的样本
## 同时加载两个模型，采用不同的参数
import os
import json
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from torch.utils.data import DataLoader
import sys 
import torch
import random
from datasets import load_from_disk
sys.path.append('/work/ruioliao/topo_agent/model_manage')
import batch_infer_register
from batch_infer_register import batch_manager
from typing import Optional
from dataclasses import dataclass, field
import pdb 

@dataclass
class InferenceConfig:
    background_dir: str
    out_answer_dir:str
    model_name: Optional[str] = field(default=None, metadata={"help": "模型名称"})
    batch_size: int = field(default=10, metadata={"help": "Batch size for data loading"})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "对于hf推理产生的最大新token数"})
    infer_type: str = field(default='hf', metadata={"help": "推理类型选择", "choices": ['vllm', 'hf']})
    device: str = field(default="cuda:0", metadata={"help": "Device for computation"})
    disk_split: bool = field(default=False, metadata={"help": "Whether to split the dataset on disk"})
    num_shards: Optional[int] = field(default=None, metadata={"help": "Number of shards if splitting the dataset"})
    shards_index: Optional[int] = field(default=None, metadata={"help": "Index of the shard if splitting the dataset"})
    gpu_memory_utilization: float = field(default=0.9, metadata={"help": "GPU memory utilization fraction"})

class InferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.batch_manager = batch_manager
    
    def read_json_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"{file_path} not found, creating a new file.")
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump([], json_file)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)

    def write_json_file(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    def update_json_data(self, data, indexes,instructions,texts,answers):
        ## 如果data 不为空
        if data:
            for idx,text,instruction, (ans_1, ans_2) in zip(indexes,texts,instructions, answers):
                if idx >= len(data):
                    print(f"Index {idx} out of range for data with length {len(data)}")
                    
                    data.append({"index":idx,
                            "instruction":instruction,
                            "text":text,
                            "answer_1":ans_1,
                            "answer_2":ans_2})
                else:
                    data[idx]['answer_1'] = ans_1
                    data[idx]['answer_2'] = ans_2
        else:
            for idx,instruction,text, (ans_1, ans_2) in zip(indexes,instructions,texts, answers):
                data.append({"index":idx,
                            "instruction":instruction,
                            "text":text,
                            "answer_1":ans_1,
                            "answer_2":ans_2})
            

    def write_results(self, batch_file_paths, batch_indexes,batch_instructions,batch_texts, answer_list_1, answer_list_2):
        last_output_file_path = os.path.join(self.config.out_answer_dir,os.path.basename(batch_file_paths[0]))
        current_answer_list = []
        current_index_list = []
        current_text_list = [] 
        current_instruction_list = [] 

        for file_path, index,instruction,text, answer_1, answer_2 in zip(batch_file_paths, batch_indexes,batch_instructions,batch_texts, answer_list_1, answer_list_2):
            output_file_path = os.path.join(self.config.out_answer_dir,os.path.basename(file_path))
            if output_file_path != last_output_file_path:
                if current_answer_list:
                    """
                    FIXME:有些文件写出来是空列表存在BUG 
                    if last_output_file_path =='/work/ruioliao/topo_agent/train/rlhf/gen_data/reward_data/physrevlett_105_136401.json':
                        pdb.set_trace()
                    """
                    
                    data = self.read_json_file(last_output_file_path)
                    self.update_json_data(data, current_index_list,current_instruction_list, current_text_list,current_answer_list)
                    self.write_json_file(last_output_file_path, data)
                    
                    current_answer_list = []
                    current_index_list = []
                    current_text_list = []
                    current_instruction_list = [] 

                last_output_file_path = output_file_path
            
            current_answer_list.append((answer_1, answer_2))
            current_index_list.append(index)
            current_text_list.append(text)
            current_instruction_list.append(instruction)

        if current_answer_list:
            data = self.read_json_file(last_output_file_path)
            self.update_json_data(data, current_index_list,current_instruction_list,current_text_list, current_answer_list)
            self.write_json_file(last_output_file_path, data)

    def random_gen_parm(self):
        top_p = random.uniform(0.8, 1.0)
        top_k = random.randint(40, 100)
        temperature = random.uniform(0.7, 1.3)
        return top_p, top_k, temperature

    def main(self):
        total_time = 0
        total_files = 0

        dataset = load_from_disk(self.config.background_dir)
        if self.config.disk_split:
            dataset = dataset.shard(num_shards=self.config.num_shards, index=self.config.shards_index, contiguous=True)
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        if self.config.infer_type == "vllm":
            self.batch_manager.load_model("vllm", self.config.model_name, dtype=torch.float16, device=self.config.device, gpu_memory_utilization=self.config.gpu_memory_utilization)
        else:
            self.batch_manager.load_model("hf", self.config.model_name, dtype=torch.float16, device=self.config.device)

        for batch_data in dataloader:
            top_p_1, top_k_1, temperature_1 = self.random_gen_parm()
            top_p_2, top_k_2, temperature_2 = self.random_gen_parm()

            start_time = time.time()
            answer_list_1 = self.batch_manager.model_infer(batch_data['instructions'],
                                                           temperature=temperature_1,
                                                           top_p=top_p_1,
                                                           top_k=top_k_1,
                                                           max_tokens=self.config.max_new_tokens,
                                                           repetition_penalty=1.2)
            
            answer_list_2 = self.batch_manager.model_infer(batch_data['instructions'],
                                                           temperature=temperature_2,
                                                           top_p=top_p_2,
                                                           top_k=top_k_2,
                                                           max_tokens=self.config.max_new_tokens,
                                                           repetition_penalty=1.2)
            
            self.write_results(batch_data['file_paths'], batch_data['indexes'].tolist(),batch_data['instructions'],batch_data["texts"],answer_list_1, answer_list_2)
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            total_files += len(batch_data['file_paths'])

        if total_files > 0:
            average_time_per_file = total_time / total_files
            print(f"Average processing time per file: {average_time_per_file:.2f} seconds")
        else:
            print("No files processed.")

if __name__ == "__main__":
    start_time = time.time()
    config = InferenceConfig(
        background_dir='/work/ruioliao/topo_agent/train/rlhf/gen_data/topo_background',
        out_answer_dir = '/work/ruioliao/topo_agent/train/rlhf/gen_data/reward_data',
        model_name='llama3_8b',
        batch_size=20,
        max_new_tokens=1024,
        infer_type='vllm',
        device='cuda:7',
        disk_split=False,
        num_shards=None,
        shards_index=None,
        gpu_memory_utilization=0.9
    )
    pipeline = InferencePipeline(config)
    pipeline.main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

## 得加Instruction 方便后续调优
## "choice": "answer_1"