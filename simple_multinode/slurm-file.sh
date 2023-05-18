#!/bin/bash

#SBATCH --nodes=4
#SBATCH --job-name=multinode-ml
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
##SBATCH --time=0-00:10:00
##SBATCH --partition=debug
##SBATCH --gres=gpu:2
#SBATCH --error=file.err
#SBATCH --output=file.out

ml cuda

#nvidia-smi
cp -r ../data /dev/shm/simple-$J
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
srun torchrun \
--nnodes 4 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
main.py
rm -r /dev/shm/simple-$J