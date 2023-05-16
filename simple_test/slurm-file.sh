#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=test
##SBATCH --cpus-per-task=cpu
#SBATCH --gpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=debug
##SBATCH --gres=gpu:2
#SBATCH --error=file.err
#SBATCH --output=file.out

ml cuda

#nvidia-smi

python3 main.py
