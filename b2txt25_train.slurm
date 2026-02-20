#!/bin/bash
#SBATCH --job-name=b2txt25_train
#SBATCH --account=connorng
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=b2txt25_train_%j.out
#SBATCH --error=b2txt25_train_%j.err

module reset
module load cuda
module load anaconda3

conda activate b2txt25

cd /home/connorng/b2txt25/model_training

python train_model.py