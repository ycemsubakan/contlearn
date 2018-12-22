import numpy as np
import torch
import visdom
import utils as ut
import pdb
import argparse
import models 
import copy
import os

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                    use_incoming_socket=False)
assert vis.check_connection()

parser = argparse.ArgumentParser(description='Multitask experiments')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

arguments.data = 'mnist'
arguments.input_type = 'autoenc'
arguments.batch_size = 1000

loader, _ = ut.get_loaders(300, arguments=arguments)
loader = ut.filter_mnist_loader([0, 1], loader, arguments)

mdl = models.VAE(784, 784, [20, 600], 28, outlin='sigmoid', use_gates=True)
mdl = mdl.cuda()
mdl.VAE_trainer(EP=2000, cuda=True, vis=vis, train_loader=loader)

