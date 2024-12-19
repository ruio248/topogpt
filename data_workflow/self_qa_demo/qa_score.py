## 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from self_qa_q import *

class Score_Pipeline(Self_Q_Pipeline):
    
    def write_results(self, batch_file_paths, batch_indexes, batch_texts, batch_prompts,
                     batch_question, batch_answer, output_list):
        """
        Write results to JSON files. Each batch of results is written to a separate JSON file.
        """
        single_json_file = []  # List to hold the results for a single file
        batch_dir_names = [os.path.basename(os.path.dirname(file_path)) for file_path in batch_file_paths]
        batch_files_names = [os.path.basename(file_path) for file_path in batch_file_paths]

        last_file_name = batch_files_names[0]  # Track the current file name for comparison
        
        for dir_name, file_name, index, text, prompt, question, answer, output in zip(batch_dir_names, 
                                                                                    batch_files_names,
                                                                                    batch_indexes, 
                                                                                    batch_texts,
                                                                                    batch_prompts,
                                                                                    batch_question,
                                                                                    batch_answer,
                                                                                    output_list):

            # Construct the subdirectory path
            out_subdir_path = os.path.join(self.output_dir, dir_name)
            if not os.path.exists(out_subdir_path):
                os.makedirs(out_subdir_path)

            # When the file name changes, save the current data to a file
            if file_name != last_file_name:
                if single_json_file:
                    # Write out the previous batch to a file
                    out_file_path = os.path.join(out_subdir_path, last_file_name)
                    if os.path.exists(out_file_path):
                        try:
                            # If the file exists, read the existing data and append new results
                            with open(out_file_path, 'r', encoding='utf-8') as existing_file:
                                existing_json_file = json.load(existing_file)
                                single_json_file = existing_json_file + single_json_file
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to decode JSON from {out_file_path}. Proceeding with new data.")
                    
                    # Write the updated JSON to the file
                    with open(out_file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(single_json_file, json_file, ensure_ascii=False, indent=4)
                    single_json_file = []  # Reset for the next batch

                last_file_name = file_name  # Update to the new file name

            # Create the dictionary for a single entry
            single_json_data = {
                "id": index,
                "text": text,
                "prompt": prompt,
                "Question": question,  # Add the Question
                "Answer": answer,  # Add the Answer
                "raw_score": output  # Assuming output is the score
            }

            # Append the single entry to the batch list
            single_json_file.append(single_json_data)

        # After the loop, write the last batch to the file
        if single_json_file:
            out_subdir_path = os.path.join(self.output_dir, dir_name)
            out_file_path = os.path.join(out_subdir_path, last_file_name)
            if os.path.exists(out_file_path):
                try:
                    # If the file exists, read the existing data and append new results
                    with open(out_file_path, 'r', encoding='utf-8') as existing_file:
                        existing_json_file = json.load(existing_file)
                        single_json_file = existing_json_file + single_json_file
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode JSON from {out_file_path}. Proceeding with new data.")
            
            # Write the updated JSON to the file
            with open(out_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(single_json_file, json_file, ensure_ascii=False, indent=4)


    def run_inference(self):
        """
        Execute batch inference with dynamic parameter adjustment based on inference type (vLLM or HuggingFace).
        """
        print("Starting inference...")
        total_batches = len(self.dataloader)

        # Dynamically set the token parameter name based on the inference type
        # 'max_tokens' for vLLM, 'max_new_tokens' for HuggingFace models
        token_param = "max_tokens" if self.infer_type == "vllm" else "max_new_tokens"

        for i, batch_data in enumerate(self.dataloader):
            print(f"Processing batch {i + 1}/{total_batches}")
            
            # Perform model inference using the dynamically set token parameter
            score_list = batch_manager.model_infer(
                batch_data['prompts'],          # List of input prompts
                temperature=self.temperature,   # Sampling temperature
                repetition_penalty=1.2,         # Repetition penalty for diverse outputs
                **{token_param: self.max_new_tokens}  # Dynamically pass token parameter
            )

            # Write the inference results to the output directory
            self.write_results(
                batch_data['file_path'],        # List of input file paths
                batch_data['index'].tolist(),   # List of sample indices
                batch_data['texts'],            # List of input texts
                batch_data['prompts'],          # List of input prompts
                batch_data['question'],
                batch_data['answer'],
                score_list                
            )

        print("Inference completed successfully.")
    
    def run(self):
        self.load_dataset()
        self.load_model()
        self.run_inference()

if __name__ == "__main__":
    # If you want to manually define parameters, initialize Namespace directly
    args = argparse.Namespace(
        input_dir="/path/to/input/data",  # Path to input data
        output_dir="/path/to/output/results",  # Path to save output results
        model_name="chatglm4_9b",  # Model name
        batch_size=50,  # Batch size
        temperature=0.85,  # Sampling temperature
        max_new_tokens=512,  # Maximum number of generated tokens
        star_subdir_index=None,  # Starting subdirectory index
        end_subdir_index=None,  # Ending subdirectory index
        infer_type="vllm",  # Inference type: 'vllm' or 'hf'
        load_from_disk=True,  # Whether to load data from disk
        disk_split=True,  # Whether to split dataset into shards
        num_shards=8,  # Total number of shards
        shards_index=0,  # Index of the current shard
        gpu_memory_utilization=0.9  # GPU memory utilization
    )

pipeline =  Score_Pipeline(args)
pipeline.run()



## score