#!/bin/bash -l
#SBATCH -J model_deployment            # Job name for deployment
#SBATCH -p gpu_partition               # GPU partition name (replace with the correct one)
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --nodelist=gpu_node01          # Specific GPU node (replace with generic name or comment out)
#SBATCH -N 1                           # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node
#SBATCH -o ./slurm_output/deployment_output.txt  # Output log file
#SBATCH -A project_name                # Project account (replace with your project name)

echo "===== Deployment Job: Model Deployment ====="

# Environment configuration
export CUDA_VISIBLE_DEVICES=0

# Job Information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Load user environment
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deployment_env  # Replace 'deployment_env' with your conda environment name

# Deploy model using vLLM
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model_directory/Model-Name \
    --served-model-name Model-Name \
    --trust-remote-code \
    --max-model-len 4096

# Monitoring Output Log (uncomment to tail output in real-time)
# tail -f ./slurm_output/deployment_output.txt
