import numpy as np
import torch
import visdom
from utils.load_data import load_dataset
import pdb
import argparse
import models 
import copy
import os
#import utilities as ut
import pickle

from utils.optimizer import AdamNormGrad
import utils.evaluation as ev
import utils.training as tr 
from models.VAE import classifier as cls

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2',
                            use_incoming_socket=False)
assert vis.check_connection()

parser = argparse.ArgumentParser(description='Multitask experiments')

parser.add_argument('--use_visdom', type=int, default=1, 
                            help='use/not use visdom, {0, 1}')
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
parser.add_argument('--number_components_init', type=int, default=500, metavar='NC',
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
parser.add_argument('--prior', type=str, default='vampprior_short', metavar='P',
                            help='prior: standard, vampprior, vampprior_short')
parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                            help='type of the input: binary, gray, continuous')
parser.add_argument('--use_vampmixingw', type=int, default=1, help='Whether or not to use mixing weights in vamp prior, acceptable inputs: 0 1')
parser.add_argument('--separate_means', type=int, default=0, help='whether or not to separate the cluster means in the latent space, in {0, 1}')
parser.add_argument('--replay_type', type=str, default='replay', help='replay, prototype') 
parser.add_argument('--use_mixingw_correction', type=int, default=0, help='whether or not to use mixing weight correction, {0, 1}')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                            help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                            help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='dynamic_mnist', metavar='DN',
                            help='name of the dataset: static_mnist, dynamic_mnist, omniglot, fashion_mnist ,caltech101silhouettes, histopathologyGray, freyfaces, cifar10, celeba')
parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                            help='allow dynamic binarization')
parser.add_argument('--use_entrmax', type=int, default=0, help='whether or not to use entropy maximization, {0, 1}')
# semi supervise
parser.add_argument('--semi_sup', type=int, default=0, help='whether or not to do semi-supervised learning')
parser.add_argument('--Lambda', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--debug', action='store_true', 
                    help='debugging mode skips stuff')

# notes
parser.add_argument('--notes', type=str, default='', help='comments on the experiment')

arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

# directories FOR SAVING
files_path = 'model_files/'
results_path = 'results_files/'

# create the folder if it doesn't exist
if not os.path.exists(files_path):
    os.mkdir(files_path)

if not os.path.exists(results_path):
    os.mkdir(results_path)

# LOAD DATA
print('load data')
#train_loader, val_loader, test_loader, arguments = load_dataset(arguments)

if arguments.dataset_name == 'dynamic_mnist':
    Lclass = 10
    datapath = 'mnist_files/'
elif arguments.dataset_name == 'omniglot':
    Lclass = 50
    datapath = 'omniglot_files/'
elif arguments.dataset_name == 'fashion_mnist': 
    Lclass = 10
    datapath = 'fashion_mnist_files'
elif arguments.dataset_name == 'mnist_plus_fmnist': 
    Lclass = 20
    datapath = 'fashion_mnist_files'

train_loader, val_loader, test_loader, arguments = load_dataset(arguments)
    
#train_loader = ut.get_mnist_loaders([9], 'train', arguments, path=datapath)
#val_loader = ut.get_mnist_loaders(list(range(10)), 'validation', arguments, path=datapath)
#test_loader = ut.get_mnist_loaders(list(range(10)), 'test', arguments, path=datapath)
#
#
dt = next(iter(val_loader))
vis.images(dt[0].reshape(-1, 1, 28, 28))
##

arguments.dynamic_binarization = False

from models.VAE import VAE

exp_details = 'batch_learning' + arguments.model_name + '_' + arguments.prior + '_K' + str(arguments.number_components)  + '_wu' + str(arguments.warmup) + '_z1_' + str(arguments.z1_size) + '_z2_' + str(arguments.z2_size) + arguments.notes
model_name = results_name = arguments.dataset_name + '_' + exp_details
dr = files_path + model_name
model_path = dr + '.model'


### classifier
arguments.classifier_EP = 100
classifier = cls(arguments, 100, 784, Lclass=Lclass, architecture='conv')

if arguments.cuda:
    classifier = classifier.cuda()

optimizer_cls = AdamNormGrad(classifier.parameters(), lr=arguments.lr)

tr.train_classifier(arguments, train_loader, classifier=classifier, 
                    optimizer_cls=optimizer_cls)

acc, all_preds = ev.evaluate_classifier(arguments, classifier, test_loader)        

print('accuracy {}'.format(acc.item()))
pdb.set_trace()


### generator
model = VAE(arguments)
if arguments.cuda:
    model = model.cuda()

if 0 & os.path.exists(model_path):
    print('loading model...')

    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
else:
    print('training model...')
    optimizer = AdamNormGrad(model.parameters(), lr=arguments.lr)
    tr.experiment_vae(arguments, train_loader, val_loader, test_loader, model, 
                      optimizer, dr, arguments.model_name) 
                      
results = ev.evaluate_vae(arguments, model, train_loader, test_loader, 0, results_path, 'test')
pickle.dump(results, open(results_path + results_name + '.pk', 'wb')) 


