import numpy as np
import torch
import visdom
from utils.load_data import load_dataset
import pdb
import argparse
import models 
import copy
import os

from utils.optimizer import AdamNormGrad
from utils.evaluation import evaluate_vae as evaluate
from utils.training import experiment_vae 

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                    use_incoming_socket=False)
assert vis.check_connection()

parser = argparse.ArgumentParser(description='Multitask experiments')

parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')
parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')
# model: latent size, input_size, so on
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=500, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')
parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='vae', metavar='MN',
                    help='model name: vae, hvae_2level, convhvae_2level, pixelhvae_2level')
parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior')
parser.add_argument('--cov_type', type=str, default='diag', metavar='P',
                    help='cov_type: diag, full')
parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='dynamic_mnist', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10, celeba')
parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

model_name = arguments.dataset_name + '_' + arguments.model_name + '_' + arguments.prior + '_K' + str(arguments.number_components)  + '_wu' + str(arguments.warmup) + '_z1_' + str(arguments.z1_size) + '_z2_' + str(arguments.z2_size)

# DIRECTORY FOR SAVING
files_path = 'model_files/'
results_path = 'results_files/'
dr = files_path + model_name 

# create the folder if it doesn't exist
if not os.path.exists(files_path):
    os.mkdir(files_path)

if not os.path.exists(results_path):
    os.mkdir(results_path)

# LOAD DATA
print('load data')
train_loader, val_loader, test_loader, arguments = load_dataset(arguments)

dt = next(iter(train_loader))
vis.images(dt[0], win='data')

# importing model
if arguments.model_name == 'vae':
    if arguments.dataset_name == 'celeba':
        from models.VAE import conv_vae as VAE
    else:
        from models.VAE import VAE
elif arguments.model_name == 'hvae_2level':
    from models.HVAE_2level import VAE
elif arguments.model_name == 'convhvae_2level':
    from models.convHVAE_2level import VAE
elif arguments.model_name == 'pixelhvae_2level':
    from models.PixelHVAE_2level import VAE
else:
    raise Exception('Wrong name of the model!')

cwd = os.getcwd() + '/'

model_path = cwd + files_path + model_name + '.model'
if 1 & os.path.exists(model_path):
    print('loading model,,,')

    model = VAE(arguments).cuda()
    model.load_state_dict(torch.load(model_path))
else:
    print('training model')
    model = VAE(arguments)

    model = model.cuda()
    optimizer = AdamNormGrad(model.parameters(), lr=arguments.lr)
    experiment_vae(arguments, train_loader, val_loader, test_loader, model, 
                   optimizer, dr, arguments.model_name) 
 
# create the old model out of the first one
model2 = copy.deepcopy(model)

# load the second dataset
arguments.dataset_name = 'svhn'
train_loader, val_loader, test_loader, arguments = load_dataset(arguments)

# add the new head
#model.add_head([3, 32, 32])

model2.args = arguments
model2.add_head(args.inputs_size)
model2 = model2.cuda()

# generate data to see if everythin aight 
xgen = model2.generate_x(25).reshape(-1, 1, 28, 28)
vis.images(xgen, win='xgen')

# run the experiment for the second dataset
model_name = arguments.dataset_name + '_' + arguments.model_name + '_' + arguments.prior + '_K' + str(arguments.number_components)  + '_wu' + str(arguments.warmup) + '_z1_' + str(arguments.z1_size) + '_z2_' + str(arguments.z2_size) + '_secondtask'
dr2 = files_path + model_name

optimizer = AdamNormGrad(model2.parameters(), lr=arguments.lr)
experiment_vae(arguments, train_loader, val_loader, test_loader, model2, 
               optimizer, dr2, arguments.model_name, prev_model=model) 


#path = 'results/' 
#if 1: 
#    results = evaluate(arguments, model, train_loader, test_loader, 9999, dr, mode='test')
#    pickle.dump(results, open(path + model_name + '_vamp_K_{}'.format(Kmog) + '.pk', 'wb'))



#dt1 = iter(loader1).next()
#dt2 = iter(loader2).next()

#N = 64
#vis.images(dt1[1][:N], win='dt1')
#vis.images(dt2[1][:N], win='dt2')

#mdl = models.VAE(784, 784, [20, 600], 28, outlin='sigmoid', use_gates=True)
#mdl = mdl.cuda()
#
#for dg in range(3): 
#
#    path1 = 'model{}.t'.format(dg)
#    path2 = 'model{}.t'.format(dg+1)
#
#    print('Continual learning step, digit {}'.format(dg))
#    loader1, loader2, _ = ut.form_mixtures(dg, dg+1, loader, arguments)
#
#    # when in first task
#    if dg == 0:
#        if os.path.exists(path1):
#            mdl.load_state_dict(torch.load(path1))
#        else:
#            mdl.VAE_trainer(EP=2000, cuda=True, vis=vis, train_loader=loader1, 
#                            config_num=dg)
#            torch.save(mdl.state_dict(), path1)
#
#    # auxiliary model 
#    aux_mdl = copy.deepcopy(mdl)
#
#    if os.path.exists(path2): 
#        mdl.load_state_dict(torch.load(path2))
#    else:
#        mdl.VAE_trainer(EP=2000, cuda=True, vis=vis, train_loader=loader2, 
#                        replay_gen=aux_mdl, config_num=dg+1)
#        torch.save(mdl.state_dict(), path2)
#


