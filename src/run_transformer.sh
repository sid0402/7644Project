#!/bin/bash
#SBATCH --job-name=transformer
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --gres=gpu:1

module load anaconda3

conda activate dl_proj

python src/train_eval.py \
    --train \
    --epochs 1 \
    --batch_size 32 \
    --lr 0.0001 \

python train_decoder.py \
    --train \
    --epochs 20 \
    --seq_length 512 \
    --batch_size 32 \
    --max_skip_prob 0.1 \
    --d_model 512 \
    --num_layers 12 \
    --d_ff 2048 
