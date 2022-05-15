#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=23:59:00
#SBATCH --job-name=FNet_1
#SBATCH --output=1.out

# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4

# Run the Python script that uses the GPU
python3 -u main.py
