#!/bin/bash
#SBATCH --gres=gpu:1 # request GPU "generic resource" 
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. 
#SBATCH --mem=10000M # memory per
#SBATCH --time=0-07:00 # time (DD-HH:MM) 
#SBATCH --output=logs/%j.out
#SBATCH --account=def-bengioy


# @CEM: you might need to change this
source ~/torch/bin/activate 
cd ~/contlearn
python "$@"
