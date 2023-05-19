#!/bin/bash

#SBATCH --nodes=4
#SBATCH --gpus=8
#SBATCH --job-name=multinode-ml
##SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --error=file.err
#SBATCH --output=file.out

ml cuda

#nvidia-smi
export STOR_LOC=/tmp
srun cp -r ../data $STOR_LOC/
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export OMP_NUM_THREADS=32
srun torchrun --nnodes 4 --nproc_per_node gpu --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $master_addr:29500 main.py

srun rm -r $STOR_LOC/data
