#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000
#SBATCH --gres=gpu:1



module use /opt/insy/modulefiles

module load miniconda/3.8
module load cuda/11.1
module load cudnn/11.1-8.0.5.39

module load matlab/R2019b
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun matlab < CNDCM_v2.m

srun miniconda < train_cluster.py
