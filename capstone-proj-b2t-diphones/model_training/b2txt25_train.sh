#!/bin/bash
#SBATCH --job-name=b2txt25_train_diphones
#SBATCH --account=connorng-2
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

#SBATCH --output=b2txt25_train_diphones_%j.out
#SBATCH --error=b2txt25_train_diphones_%j.err

module load Miniconda3/25.11.1-1

source activate ~/.conda/envs/b2txt25_d

python3 train_model.py