#!/bin/bash -l
#SBATCH -J format_tune_a
#SBATCH -p h800
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpuh8001 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o /work/data/ruioliao/slurm_output/format_tune_a
#SBATCH -A hmt03

# 记录开始时间
start_time=$(date +%s)

export CUDA_VISIBLE_DEVICES= 0,1,2,3
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"

source /home/ruioliao/.bashrc
source /home/ruioliao/anaconda3/etc/profile.d/conda.sh
source activate vllm
conda=/home/liaoruihao/anaconda3/condabin/conda

# 运行 Python 脚本
deepspeed --num_gpus=4 /data/ruioliao/topo_agent/train/sft/sft.py \
          

# 记录结束时间
end_time=$(date +%s)

# 计算运行时长
runtime=$((end_time - start_time))

# 输出运行时间（秒）
echo "Job runtime: $runtime seconds"

# 如果你想以小时、分钟、秒的形式输出运行时间，可以使用以下代码
echo "Job runtime: $(($runtime / 3600)) hours, $((($runtime % 3600) / 60)) minutes, $(($runtime % 60)) seconds"