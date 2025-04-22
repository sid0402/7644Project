#!/bin/bash
#SBATCH --job-name=install-env
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=install.log
#SBATCH --mem=32G

module load anaconda3
conda create -n layerskip_gpu_env_clean python=3.10 -y
source activate layerskip_gpu_env_clean

conda config --env --add channels conda-forge
conda config --env --add channels nvidia
conda config --env --add channels pytorch
conda config --env --set channel_priority strict

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -y
