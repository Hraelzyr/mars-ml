#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=simple_ml_model
##SBATCH --cpus-per-task=cpu
#SBATCH --gpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --error=file.err
#SBATCH --output=file.out

ml cuda

#nvidia-smi
cp -r ../data /dev/shm/simpleml-%J
python3 main.py
rm -r /dev/shm/simpleml-%J