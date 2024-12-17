## This script defines the batch inference logic for BERT
## Each time a directory is passed
import os
import torch
import time
from text_dataset import TextDataset
from torch.utils.data import DataLoader  # Import DataLoader
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

"""
Define the labels used by the data
"""
ID2LABEL = {0: "High-quality data, rich in information.", 
            1: "Low-quality data, lacking in information"}

LABEL2ID = {"High-quality data, rich in information.": 0, 
            "Low-quality data, lacking in information": 1}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example of argument parsing")
    # Use action='append' to collect multiple directory arguments, or pass them as a comma-separated string
    parser.add_argument("--dir_path", type=str, required=True, default="llm_train_data/bert_sam_topo")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Model address")
    parser.add_argument("--tokenizer_path", type=str, default="model/bert_base_uncased", help="Tokenizer address")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--dir_path_star", type=int, required=True)
    parser.add_argument("--dir_path_end", type=int, required=True)
    args = parser.parse_args()
    return args

## CUDA number defined in the script
def main(dataloader, model, tokenizer, out_dir, device):  
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient tracking for inference
        for batch in dataloader:
            #print(batch['file_path'], batch['index'], batch['text'])
            batch_text_tokenize = tokenizer.batch_encode_plus(
                                    batch_text_or_text_pairs=batch['text'],
                                    truncation=True,
                                    padding=True,  # No padding here
                                    add_special_tokens=True,
                                    max_length=512,
                                    return_tensors='pt',
                                    return_token_type_ids=False,
                                    return_attention_mask=True,
                                    return_special_tokens_mask=False
                                    )
            b_input_ids, b_input_mask = batch_text_tokenize['input_ids'], batch_text_tokenize['attention_mask']
            
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            #start_time = time.time()
            results = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            )
            logits = results.logits.detach().cpu()
            score_list = softmax(logits, axis=1)[:, 1].tolist()

            ## score_list returns [(tex1, text2),....] type of scores
            #end_time = time.time()
            #elapsed_time = end_time - start_time
            #print(f"Time taken to generate result: {elapsed_time:.2f} seconds")
            batch['index'] = batch['index'].tolist()
            write_results(batch['file_path'], batch['index'], score_list, out_dir)

## Logic for writing the results
def write_results(batch_file_paths, batch_indexs, score_list, out_dir): 
    
    inform_list = []
    out_file_path = ""  # Initialize out_file_path outside the loop

    for file_path, index, score in zip(batch_file_paths, batch_indexs, score_list):
        curr_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
        sub_dir_name = os.path.basename(os.path.dirname(file_path))
        out_sub_dir_path = os.path.join(out_dir, sub_dir_name)

        # Ensure directory exists
        if not os.path.exists(out_sub_dir_path):
            os.makedirs(out_sub_dir_path)

        new_out_file_path = os.path.join(out_sub_dir_path, curr_file_name)
        
        # Check if a new file is being processed, and write previous file results
        if new_out_file_path != out_file_path and inform_list:
            with open(out_file_path, 'w') as f:
                for item in inform_list:
                    f.write(f"{item[0]}  {item[1]}  {item[2]}\n")
            inform_list = []  # Reset inform_list for the new file

        # Update current file path and add current line to inform_list
        out_file_path = new_out_file_path
        inform_list.append([file_path, index, score])

    # Process any remaining data after loop
    if inform_list:
        with open(out_file_path, 'w') as f:
            for item in inform_list:
                f.write(f"{item[0]}  {item[1]}  {item[2]}\n")

if __name__ == "__main__":
    # Use parsed arguments
    args = parse_arguments()
    print("Directory list:", args.dir_path)
    print("Model path:", args.model_path)

    ## Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = BertForSequenceClassification.from_pretrained(
                                                      args.model_path,
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False,
                                                      id2label=ID2LABEL,
                                                      label2id=LABEL2ID,
                                                      )
    model.to(device)
    model.eval()
    listdir = sorted(os.listdir(args.dir_path))
    ## Change here for 100 directories, processing 100 at a time
    for dir_name in listdir[args.dir_path_star:args.dir_path_end]:
        full_dir_path = os.path.join(args.dir_path, dir_name)
        print(f"Processing directory: {full_dir_path}")
        dataset = TextDataset(dir_path=full_dir_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        main(dataloader, model, tokenizer, args.out_dir, device)
