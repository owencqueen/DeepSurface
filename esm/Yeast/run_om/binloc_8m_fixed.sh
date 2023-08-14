#!/bin/bash
#SBATCH -J bl8mf
#SBATCH -o /om2/user/oqueen/DeepSurface/esm/Yeast/out/%x.%j.out
#SBATCH -e /om2/user/oqueen/DeepSurface/esm/Yeast/out/%x.%j.err
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30G

base="/om2/user/oqueen/DeepSurface/esm/Yeast"
cd $base

conda activate PLM
python3 ESM-yeast.py --nparams "8m" --frozen