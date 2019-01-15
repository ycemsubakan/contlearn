#!/bin/bash
#SBATCH --gres=gpu:1 # request GPU "generic resource" 
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. 
#SBATCH --mem=10000M # memory per
#SBATCH --time=0-07:00 # time (DD-HH:MM) 
#SBATCH --output=logs/%j.out
#SBATCH --account=rpp-bengioy


source ~/torch/bin/activate 
cd ~/multitask_things
python main_mnist.py --dataset_name mnist_plus_fmnist --prior vampprior_short --replay_size increase --use_mixingw_correction 0 --add_cap 1 --number_components 50 --number_components_init 50 --replay_type replay --notes 'testfmnist' --use_vampmixingw 1 --separate_means 0 --restart_means 1 --use_classifier 1  --use_replaycostcorrection 0 --use_visdom 1 --dynamic_binarization 0 --permindex 0 --semi_sup 0 --classifier_EP 75 --use_entrmax 1
