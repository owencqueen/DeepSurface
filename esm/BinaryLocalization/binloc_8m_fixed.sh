#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/DeepSurface/esm/BinaryLocalization/out/bl_8m_fixed.o
#SBATCH -e /lustre/isaac/scratch/oqueen/DeepSurface/esm/BinaryLocalization/out/bl_8m_fixed.e
#SBATCH -J bl_8m_fixed

base="/lustre/isaac/scratch/oqueen/DeepSurface/esm/BinaryLocalization"
cd $base

conda activate /lustre/isaac/scratch/oqueen/codonbert
python3 ESM-binloc.py --nparams "8m" --frozen