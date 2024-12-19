from q_dataset import*
from template import QA_Generation_Template,QA_COT_Template
from datasets import load_from_disk

class SELF_QA(SELF_Q):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_prompt(self, data) -> list:
        prompt_list = []
        for single_data in data:
            prompt = QA_COT_Template.format(background_knowledge=single_data['text'])
            prompt_list.append(prompt)
        return prompt_list
        
if __name__ == "__main__": 
    
    dataset = SELF_QA('/data/work/HUGGINGFACE/llm_train_data/pretrain_data/from_paper/re_washed_text_final_bert_chunk_topo',9,17)
    
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
    
    output_dir = '/data/work/HUGGINGFACE/qa_backinform_cot/chunk_9_17'
    hf_dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
    
    
