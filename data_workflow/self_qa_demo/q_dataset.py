import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from datasets import Features, Value, load_from_disk
from template import INSTRUCT_GENERATE_Prompt
from tqdm import tqdm

class SELF_Q(Dataset):
    def __init__(self, input_dir, start_subdir_index, end_subdir_index):
        self.input_dir = input_dir  # Directory containing the data
        self.start_subdir_index = start_subdir_index  # Starting index of subdirectory for processing
        self.end_subdir_index = end_subdir_index  # Ending index of subdirectory for processing
        self.data = []  # List to store the data
        self.get_data()  # Method call to process and load data
        
    def read_inform(self, file_path) -> list:
        # Read data from a JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            inform_data = json.load(file)
        return inform_data
 
    def get_prompt(self, data) -> list:
        # Generate prompts based on data
        prompt_list = []
        for single_data in data:
            prompt = INSTRUCT_GENERATE_Prompt.format(background_knowledge=single_data['text'])
            prompt_list.append(prompt)
        return prompt_list
        
    def summarize_inform(self, file_path):
        # Summarize information from file and generate prompts
        overall_data = []
        data = self.read_inform(file_path)
        prompts_list = self.get_prompt(data)
        data_length = len(prompts_list)
        index_list = range(data_length)
        file_path_list = [file_path] * data_length
        for i in range(data_length):
            overall_data.append((file_path_list[i], index_list[i], prompts_list[i], data[i]['text']))
        return overall_data

    def get_data(self):
        # Process files and gather data from directories within specified index range
        subdir_name_list = sorted(os.listdir(self.input_dir))  # Ensure the file list is sorted
        process_subdir_name = subdir_name_list[self.start_subdir_index:self.end_subdir_index]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for subdir_name in process_subdir_name:
                subdir_path = os.path.join(self.input_dir, subdir_name)
                file_name_list = os.listdir(subdir_path)
                file_path_list = [os.path.join(subdir_path, file_name) for file_name in file_name_list]
                for file_path in file_path_list: 
                    futures.append(executor.submit(self.summarize_inform, file_path))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                result = future.result()
                if result:
                    self.data.extend(result)  # Append the entire dictionary to self.data

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return data at specific index
        file_path, index, prompts, texts = self.data[idx]
        return {'file_path': file_path, 'index': index, 'prompts': prompts, 'texts': texts}

# Usage example
if __name__ == "__main__":
    dataset = SELF_Q('/path/to/data/llm_topo_chunk_final/split', 0, 8)
    
    # Convert dataset to a Dataset object of the 'datasets' library
    data_dict = {
        'file_path': [item['file_path'] for item in dataset],
        'index': [item['index'] for item in dataset],
        'prompts': [item['prompts'] for item in dataset],
        'texts': [item['texts'] for item in dataset]
    }
    
    features = Features({
        'file_path': Value(dtype='string'),
        'index': Value(dtype='int32'),
        'prompts': Value(dtype='string'),
        'texts': Value(dtype='string')
    })
    
    hf_dataset = HFDataset.from_dict(data_dict, features=features)
    
    # Save dataset to directory
    output_dir = '/path/to/output/dataset'
    hf_dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
