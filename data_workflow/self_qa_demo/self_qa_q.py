import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
from tqdm import tqdm
import argparse
from q_dataset import SELF_Q
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import torch
import sys 
sys.path.append('/topogpt/model_manage')
sys.path.append('/topogpt//data/utils')
from batch_infer_register import batch_manager
from batch_infer_write import *

class Self_Q_Pipeline(BasePipeline):
    """
    InferencePipeline extends BasePipeline to provide implementation for:
    - Dataset loading (custom or HuggingFace datasets).
    - Running inference using vLLM or HuggingFace models.
    - Writing the inference results into JSON files.
    """

    def load_dataset(self):
        """Load dataset from disk or custom input directory."""
        print("Loading dataset...")
        if self.load_from_disk:
            dataset = load_from_disk(self.input_dir)
            if self.disk_split:
                dataset = dataset.shard(num_shards=self.num_shards, index=self.shards_index, contiguous=True)
        else:
            dataset = SELF_Q(self.input_dir, self.star_subdir_index, self.end_subdir_index)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        print("Dataset loaded successfully.")

    def run_inference(self,output_key):
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
            question_list = batch_manager.model_infer(
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
                output_key,                     # Key name for output results
                question_list                   # List of generated results
            )

        print("Inference completed successfully.")

    def run(self,output_key):
        self.load_dataset()
        self.load_model()
        self.run_inference(output_key)

if __name__ == "__main__":
    # If you want to manually define parameters, initialize Namespace directly
    args = argparse.Namespace(
        input_dir="/path/to/input/data",  # Path to input data
        output_dir="/path/to/output/results",  # Path to save output results
        model_name="llama3_8b",  # Model name
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

    # Uncomment the following to pass parameters via command line
    # parser = argparse.ArgumentParser(description="Batch Inference Pipeline")
    # parser.add_argument("--input_dir", type=str, required=True, help="Input directory with data")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    # parser.add_argument("--model_name", type=str, default=None, help="Model name to use")
    # parser.add_argument("--batch_size", type=int, default=58)
    # parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    # parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for HF inference")
    # parser.add_argument("--infer_type", type=str, choices=['vllm', 'hf'], default='hf', help="Inference type")
    # parser.add_argument("--load_from_disk", action="store_true", help="Load dataset from disk")
    # args = parser.parse_args()

    # Initialize and run the inference pipeline
    pipeline = Self_Q_Pipeline(args)
    pipeline.run(output_key='question')
