#!/bin/bash
#SBATCH -J bert
#SBATCH -o /om2/user/oqueen/DeepSurface/baselines/out/%x.%j.out
#SBATCH -e /om2/user/oqueen/DeepSurface/baselines/out/%x.%j.err
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G

base="/om2/user/oqueen/DeepSurface"
cd $base/baselines

python3 $base/baselines/ProtBERT-ds.py