from numpy import single
from q_dataset import *  # Assuming necessary imports and classes are defined here
from template import Quality_Template_1  # Assuming this template is used for formatting prompts
from datasets import load_from_disk
from datasets import Dataset as HFDataset
from datasets import Features, Value
from tqdm import tqdm

class SCORE(SELF_Q):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the parent class

    def get_prompt(self, data) -> list:
        """
        Overrides the method to format prompts that include both questions and answers.
        """
        prompt_list = []
        for single_data in data:
            prompt = Quality_Template_1.format(
                background_knowledge=single_data['text'],
                question=single_data["Question"],
                answer=single_data["Answer"]
            )
            prompt_list.append(prompt)
        return prompt_list
    
    def summarize_inform(self, file_path):
        """
        Overrides to process entries containing both 'Question' and 'Answer'.
        Includes entries only if both fields are non-empty.
        """
        overall_data = []
        data = self.read_inform(file_path)
        for single_data in data:
            question = single_data.get('Question', "")
            answer = single_data.get('Answer', "")
            if question and answer:
                prompts_list = self.get_prompt([single_data])
                file_path_list = [file_path]
                index_list = [len(overall_data)]
                overall_data.append((
                    file_path_list[0],
                    index_list[0],
                    prompts_list[0],
                    single_data['text'],
                    question,
                    answer
                ))
        return overall_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches an item by index and includes additional question and answer fields.
        """
        file_path, index, prompts, texts, question, answer = self.data[idx]
        return {
            'file_path': file_path,
            'index': index,
            'prompts': prompts,
            'texts': texts,
            'question': question,
            'answer': answer
        }

if __name__ == "__main__": 
    # Create a dataset instance with a specified directory and range
    dataset = SCORE('/path/to/question/data', 0, 2)  # Adjust path and range appropriately

    # Convert the dataset into a format compatible with Hugging Face's `datasets` library
    data_dict = {
        'file_path': [item['file_path'] for item in dataset],
        'index': [item['index'] for item in dataset],
        'prompts': [item['prompts'] for item in dataset],
        'texts': [item['texts'] for item in dataset],
        'question': [item['question'] for item in dataset],
        'answer': [item['answer'] for item in dataset]
    }
    
    features = Features({
        'file_path': Value(dtype='string'),
        'index': Value(dtype='int32'),
        'prompts': Value(dtype='string'),
        'texts': Value(dtype='string'),
        'question': Value(dtype='string'),
        'answer': Value(dtype='string')
    })
    
    hf_dataset = HFDataset.from_dict(data_dict, features=features)
    
    output_dir = '/path/to/output/dataset'
    hf_dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
