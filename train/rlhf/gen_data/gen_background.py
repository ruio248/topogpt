## 此脚本用来产生背景信息
## 这里以self-q 为例,这里均是存储为hfdataset 形式
from datasets import Dataset as HFDataset
from datasets import Features, Value, load_from_disk
from datasets import load_dataset
from tqdm import tqdm
import pdb
import os
import sys
sys.path.append("/work/ruioliao/topo_agent/data/self_qa_demo")
from template import INSTRUCT_GENERATE_Prompt
import json

class Gen_inform:
    def __init__(self,input_dir,output_dir):
        self.input_dir = input_dir
        self.input_data =None
        self.data_dict = {}
        self.get_data_dict()      
        self.output_dir = output_dir
        
    def get_data_dict(self):
        
        file_path_list = []
        index_list = []
        texts_list = []
        instruction_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    for data in data_list:
                        file_path_list.append(file_path)
                        index_list.append(data.get('index'))  
                        texts_list.append(data.get('text')) 
                        instruction = INSTRUCT_GENERATE_Prompt.format(background_knowledge=data.get('text')) 
                        instruction_list.append(instruction)
        
        self.data_dict = {
            'file_paths': file_path_list,
            'indexes': index_list,
            'instructions':instruction_list,
            'texts': texts_list
        }

    def save_to_disk(self):
        features = Features({
        'file_paths': Value(dtype='string'),
        'indexes': Value(dtype='int32'),
        'instructions': Value(dtype='string'),
        'texts': Value(dtype='string')
        })
        hf_dataset = HFDataset.from_dict(self.data_dict, features=features)
        hf_dataset.save_to_disk(self.output_dir)
        
if __name__ == "__main__":
    input_dir = "/work/HUGGINGFACE/llm_train_data/pretrain_data/from_paper/llm_topo_chunk_final/split"
    output_dir = "/work/ruioliao/topo_agent/train/rlhf/gen_data/topo_background"
    gen_inform = Gen_inform(input_dir,output_dir)
    gen_inform.save_to_disk()
    

"""
        data_dict = {
            'file_path': [item['file_path'] for item in dataset],
            'index': [item['index'] for item in dataset],
            'prompts': [item['prompts'] for item in dataset],
            'texts': [item['texts'] for item in dataset]
        }

"""