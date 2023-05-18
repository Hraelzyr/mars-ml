#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=test
##SBATCH --cpus-per-task=cpu
#SBATCH --gpus-per-task=2
##SBATCH --time=0-00:10:00
##SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --error=file.err
#SBATCH --output=file.out

ml cuda

#nvidia-smi
cp -r ../data /dev/shm/simple-$J
srun torchrun --standalone --nproc_per_node=gpu main.py
rm -r /dev/shm/simple-$J