import numpy as np
import torch
import visdom
from utils.load_data import load_dataset
import pdb
import argparse
import models 
import copy
import os
import utilities as ut
import pickle

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
parser.add_argument('--number_components_init', type=int, default=50, metavar='NC',
                    help='number of pseudo-inputs initial number')

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

# replay parameters
parser.add_argument('--replay_size', type=str, default='constant', help='constant, increase')
parser.add_argument('--replay_type', type=str, default='replay', help='replay, prototype') 
parser.add_argument('--add_cap', type=int, default=0, help='0, 1')
parser.add_argument('--notes', type=str, default='', help='comments on the experiment')


arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

# DIRECTORY FOR SAVING
files_path = 'model_files/'
results_path = 'results_files/'

# create the folder if it doesn't exist
if not os.path.exists(files_path):
    os.mkdir(files_path)

if not os.path.exists(results_path):
    os.mkdir(results_path)

# LOAD DATA
print('load data')
if not os.path.exists('mnist_files'):
    train_loader, val_loader, test_loader, arguments = load_dataset(arguments)
    ut.separate_mnist(train_loader, 'train')
    ut.separate_mnist(val_loader, 'validation')
    ut.separate_mnist(test_loader, 'test')

#dt = next(iter(train_loader))
#vis.images(dt[0].reshape(-1, 1, 28, 28), win='data')

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
    raise Exception('Wrong model name!')

# start training
cwd = os.getcwd() + '/'
all_results = []
results_name = arguments.dataset_name + '_' + arguments.model_name + '_' + arguments.prior + '_K' + str(arguments.number_components)  + '_wu' + str(arguments.warmup) + '_z1_' + str(arguments.z1_size) + '_z2_' + str(arguments.z2_size) + 'replay_size_'+ str(arguments.replay_size) + arguments.replay_type + '_add_cap_' + str(arguments.add_cap) + arguments.notes

for dg in range(0, 10):
    train_loader = ut.get_mnist_loaders([dg], 'train', arguments)
    val_loader = ut.get_mnist_loaders(list(range(dg+1)), 'validation', arguments)
    test_loader = ut.get_mnist_loaders(list(range(dg+1)), 'test', arguments)

    model_name = arguments.dataset_name + str(dg) + '_' + arguments.model_name + '_' + arguments.prior + '_K' + str(arguments.number_components)  + '_wu' + str(arguments.warmup) + '_z1_' + str(arguments.z1_size) + '_z2_' + str(arguments.z2_size) + 'replay_size_' + str(arguments.replay_size) + arguments.replay_type + '_add_cap_' + str(arguments.add_cap)
    dr = files_path + model_name 

    model_path = cwd + files_path + model_name + '.model'

    if dg == 0:
        if 1 & os.path.exists(model_path):
            print('loading model... for digit {}'.format(dg))

            model = VAE(arguments).cuda()
            model.load_state_dict(torch.load(model_path))
        else:
            print('training model... for digit {}'.format(dg))
            model = VAE(arguments).cuda()

            optimizer = AdamNormGrad(model.parameters(), lr=arguments.lr)
            experiment_vae(arguments, train_loader, val_loader, test_loader, model, 
                           optimizer, dr, arguments.model_name) 
    else:
        if 1 & os.path.exists(model_path):
            print('loading model... for digit {}'.format(dg))

            #model = VAE(arguments).cuda()
            model.load_state_dict(torch.load(model_path))
        else:
            print('training model... for digit {}'.format(dg))
            #model = VAE(arguments)

            model = model.cuda()
            optimizer = AdamNormGrad(model.parameters(), lr=arguments.lr)
            experiment_vae(arguments, train_loader, val_loader, test_loader, model, 
                           optimizer, dr, arguments.model_name, prev_model=prev_model, dg=dg) 
        
    print('evaluating the model...')
    results = evaluate(arguments, model, train_loader, test_loader, 0, results_path, 'test')
    results['digit'] = dg
    all_results.append(results)
    pickle.dump(all_results, open(results_path + results_name + '.pk', 'wb')) 
    
    if arguments.replay_type == 'replay': 
        prev_model = copy.deepcopy(model)
    elif arguments.replay_type == 'prototype':
        X = model.means(model.idle_input)

        z_p_mean = model.q_z_mean(X)
        model.prototypes, _ = model.p_x(z_p_mean)
        model.prototypes = model.prototypes.data
        prev_model = None

    if arguments.add_cap and (dg < 9):
        model.add_latent_cap(dg)

