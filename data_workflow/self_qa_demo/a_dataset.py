import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from datasets import Features, Value, load_from_disk
from template import Reading_Comprehension_Prompt
from tqdm import tqdm

class SELF_A(Dataset):
    def __init__(self, q_dir, start_subdir_index, end_subdir_index):
        self.input_dir = q_dir  # Directory containing question data
        self.start_subdir_index = start_subdir_index  # Start index for processing subdirectories
        self.end_subdir_index = end_subdir_index  # End index for processing subdirectories
        self.data = []  # Data container
        self.get_data()  # Populate data from directories
        
    def read_inform(self, file_path) -> list:
        # Load information from JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            inform_data = json.load(file)
        return inform_data
 
    def get_prompt(self, data) -> list:
        # Generate prompts from the provided data
        prompt_list = []
        for single_data in data:
            prompt = Reading_Comprehension_Prompt.format(background_knowledge=single_data['text'], response=single_data["question"])
            prompt_list.append(prompt)
        return prompt_list
        
    def summarize_inform(self, file_path):
        # Summarize information from file path and generate prompts
        overall_data = []
        data = self.read_inform(file_path)
        prompts_list = self.get_prompt(data)
        data_length = len(prompts_list)
        index_list = range(data_length)
        file_path_list = [file_path] * data_length
        for i in range(data_length):
            overall_data.append((file_path_list[i], index_list[i], prompts_list[i]))
        return overall_data

    def get_data(self):
        # Load and process data from subdirectories
        subdir_name_list = sorted(os.listdir(self.input_dir))  # Ensure directories are sorted
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
                    self.data.extend(result)

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch an item by index
        file_path, index, prompts = self.data[idx]
        return {'file_path': file_path, 'index': index, 'prompts': prompts}

# Example usage
if __name__ == "__main__":
    dataset = SELF_A('/path/to/question/data', 0, 1)
    
    # Convert dataset to a 'datasets' library Dataset object
    data_dict = {
        'file_path': [item['file_path'] for item in dataset],
        'index': [item['index'] for item in dataset],
        'prompts': [item['prompts'] for item in dataset]
    }
    
    features = Features({
        'file_path': Value(dtype='string'),
        'index': Value(dtype='int32'),
        'prompts': Value(dtype='string'),
    })
    
    hf_dataset = HFDataset.from_dict(data_dict, features=features)
    
    # Save dataset to directory
    output_dir = '/path/to/output/dataset'
    hf_dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
