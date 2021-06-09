#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1



module use /opt/insy/modulefiles

module load cuda/11.1
module load cudnn/11.1-8.0.5.39

python3 cluster_test.py $*
