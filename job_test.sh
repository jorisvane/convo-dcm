#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000
#SBATCH --gres=gpu:turing:1



module use /opt/insy/modulefiles

module load cuda/11.1
module load cudnn/11.1-8.0.5.39

previous=$(nvidia-smi --query-accounted-apps=gpu_utilization,mem_utilization,max_memory_usage,time --format=csv | tail -n +2)
srun python3 train_eval_ResNet50_cluster.py $*
nvidia-smi --query-accounted-apps=gpu_utilization,mem_utilization,max_memory_usage,time --format=csv | grep -v -F "$previous"