#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gpus=4
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
#cp -r ../data /dev/shm/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export OMP_NUM_THREADS=32
echo Node IP: $master_addr
srun torchrun \
--nnodes 2 \
--nproc_per_node gpu \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $master_addr:29500 \
main.py

#rm -r /dev/shm/data
