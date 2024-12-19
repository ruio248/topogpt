from self_qa_q import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class Self_QA_Pipeline(Self_Q_Pipeline):
    """
    Self_QA_Pipeline is a subclass of Self_Q_Pipeline.
    It reuses the same logic but fixes the output_key parameter to 'answer'.
    """

    def run_inference(self,output_key):
        """
        Call parent class's method with fixed output_key
        """
        super().run_inference(output_key=output_key)  

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
        disk_split=False,  # Whether to split dataset into shards
        num_shards=0,  # Total number of shards
        shards_index=0,  # Index of the current shard
        gpu_memory_utilization=0.9  # GPU memory utilization
    )

    # Initialize and run the inference pipeline
    pipeline = Self_QA_Pipeline(args)
    pipeline.run(output_key='raw_qa')


