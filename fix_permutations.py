import numpy as np
import torch
import pdb
import datetime 
import os
import socket

timestamp = str(datetime.datetime.now()).replace(' ','')
hostname = socket.gethostname()

dataset = 'omniglot'
if dataset == 'mnist':
    T = 10 
elif dataset == 'omniglot':
    T = 50
Nperms = 1000

seed = 2
torch.manual_seed(seed)
#np.random.seed(seed)

arangemat = torch.zeros(Nperms, T)

for n in range(Nperms):
    arangemat[n] = torch.randperm(T)
arangemat[0] = torch.arange(T)

print(arangemat[:10])

torch.save(arangemat, dataset + 'permutations_seed' + str(seed) + '_' + hostname + '_' + timestamp + '.t')
