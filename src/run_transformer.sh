#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load anaconda3

conda activate dl_proj

python src/train_eval.py \
    --train \
    --eval \
    --epochs 1 \
    --batch_size 32 \
    --lr 0.0001 \