#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:pascal:2



module use /opt/insy/modulefiles

module load cuda/11.1
module load cudnn/11.1-8.0.5.39

srun python3 train_eval_ResNet50_cluster.py $*
