import numpy as np
import torch
import pdb
import datetime 
import os
import socket
import sys

timestamp = str(datetime.datetime.now()).replace(' ','')
hostname = socket.gethostname()

#dataset = 'mnist_plus_fmnist_m1'
dataset = sys.argv[1]
if dataset in ['mnist', 'fashion_mnist']:
    T = 10 
elif dataset in ['mnist_plus_fmnist_m1', 'mnist_plus_fmnist_m2']:
    T = 20
elif dataset == 'omniglot':
    T = 50
elif dataset == 'omniglot_char':
    T = 100

Nperms = 1000

seed = 2
torch.manual_seed(seed)
#np.random.seed(seed)

arangemat = torch.zeros(Nperms, T)
if (dataset != 'mnist_plus_fmnist_m1'):
    for n in range(Nperms):
        arangemat[n] = torch.randperm(T)
else:
    for n in range(Nperms): 
        mat1 = torch.randperm(10) 
        mat2 = torch.from_numpy(np.random.permutation(np.arange(10, 20)))
        arangemat[n] = torch.cat([mat1, mat2], dim=0)
arangemat[0] = torch.arange(T)

print(arangemat[:10])

torch.save(arangemat, dataset + 'permutations_seed' + str(seed) + '_' + timestamp + '.t')
