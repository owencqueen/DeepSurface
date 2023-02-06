#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/DeepSurface/esm/log/trial35m.o
#SBATCH -e /lustre/isaac/scratch/oqueen/DeepSurface/esm/log/trial35m.e
#SBATCH -J esm_35m

conda activate /lustre/isaac/scratch/oqueen/codonbert
python3 /lustre/isaac/scratch/oqueen/DeepSurface/esm/train_esm.py --num_params 35m --epochs 200 --batch_size 16 --lr 0.0001