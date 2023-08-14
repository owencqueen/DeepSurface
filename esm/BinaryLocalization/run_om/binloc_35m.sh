#!/bin/bash
#SBATCH -J bl35m
#SBATCH -o /om2/user/oqueen/DeepSurface/esm/BinaryLocalization/out/%x.%j.out
#SBATCH -e /om2/user/oqueen/DeepSurface/esm/BinaryLocalization/out/%x.%j.err
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30G

base="/om2/user/oqueen/DeepSurface/esm/BinaryLocalization"
cd $base

conda activate PLM
python3 ESM-binloc.py --nparams "35m"