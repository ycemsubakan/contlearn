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
import time

from utils.optimizer import AdamNormGrad
import utils.evaluation as ev
import utils.training as tr 
from models.VAE import classifier as cls

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                    use_incoming_socket=False)

parser = argparse.ArgumentParser(description='Multitask experiments')

parser.add_argument('--use_visdom', type=int, default=0, 
                    help='use/not use visdom, {0, 1}')
parser.add_argument('--debug', action='store_true', 
                    help='debugging mode skips stuff')
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
                    help='number of epochs for warm-up')

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

parser.add_argument('--number_components', type=int, default=50, metavar='NC',
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
parser.add_argument('--prior', type=str, default='vampprior_short', metavar='P',
                    help='prior: standard, vampprior, vampprior_short')
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
parser.add_argument('--permindex', type=int, default=1, 
                    help='permutation index, integer in [0, 1000)' )
parser.add_argument('--dynamic_binarization', type=int, default=1,
                    help='allow dynamic binarization, {0, 1}')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# replay parameters
parser.add_argument('--replay_size', type=str, default='constant', help='constant, increase')

parser.add_argument('--replay_type', type=str, default='replay', help='replay, prototype') 
parser.add_argument('--add_cap', type=int, default=0, help='0, 1')
parser.add_argument('--classifier_EP', type=int, default='75', help='number of iterations for classifier training')

parser.add_argument('--use_vampmixingw', type=int, default=1, help='Whether or not to use mixing weights in vamp prior, acceptable inputs: 0 1')
parser.add_argument('--separate_means', type=int, default=0, help='whether or not to separate the cluster means in the latent space, in {0, 1}')
parser.add_argument('--restart_means', type=int, default=1, help='whether or not to re-initialize the the cluster means in the latent space, in {0, 1}')
parser.add_argument('--use_classifier', type=int, default=0, help='whether or not to use a classifier to balance the classes, in {0, 1}')
parser.add_argument('--use_mixingw_correction', type=int, default=0, help='whether or not to use mixing weight correction, {0, 1}')
parser.add_argument('--use_replaycostcorrection', type=int, default=0, help='whether or not to use a constant for replay cost correction, {0, 1}')

# semi supervise
parser.add_argument('--semi_sup', type=int, default=0, help='whether or not to do semi-supervised learning')
parser.add_argument('--Lambda', type=float, default=1, help='weight of the classification loss')

parser.add_argument('--notes', type=str, default='', help='comments on the experiment')


# things to add: balancing via classifier, adding constants to the loss function for replay balancing
# goals: primarily trying to show the benefits of replay balancing 
# dataset ideas: sketches, mnist combinations, celeba, permutations on mnist order 
# continually training without replay

# for mnist, need to repeat the experiments with bernoulli cost  

arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()
arguments.number_components = copy.deepcopy(arguments.number_components_init)

if arguments.use_visdom:
    assert vis.check_connection()
assert arguments.semi_sup + arguments.use_classifier < 2

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

# Dataset preparation
if arguments.dataset_name == 'dynamic_mnist':
    if not os.path.exists('mnist_files'):
        train_loader, val_loader, test_loader, arguments = load_dataset(arguments)
        ut.separate_mnist(train_loader, 'train')
        ut.separate_mnist(val_loader, 'validation')
        ut.separate_mnist(test_loader, 'test')
elif arguments.dataset_name == 'omniglot':
    if not os.path.exists('omniglot_files'):
        train_loader, val_loader, test_loader, arguments = load_dataset(arguments)
        ut.separate_omniglot(train_loader, 'train')
        ut.separate_omniglot(val_loader, 'validation')
        ut.separate_omniglot(test_loader, 'test')

#train_loader = ut.get_mnist_loaders([2], 'train', arguments, path='omniglot_files/')

#dt = next(iter(train_loader))
#vis.images(dt[0].reshape(-1, 1, 28, 28))

#pdb.set_trace()

# importing model
if arguments.model_name == 'vae':
    if arguments.semi_sup:
        from models.SSVAE import SSVAE as VAE
    else:
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

exp_details = 'permutation_' + str(arguments.permindex) + \
              'db_' + str(arguments.dynamic_binarization) + \
              arguments.model_name + \
              '_' + arguments.prior + \
              '_K' + str(arguments.number_components) + \
              '_wu' + str(arguments.warmup) + \
              '_z1_' + str(arguments.z1_size) + \
              '_z2_' + str(arguments.z2_size) + \
              '_replay_size_'+ str(arguments.replay_size) + \
              arguments.replay_type + \
              '_add_cap_' + str(arguments.add_cap) + \
              '_usevampmixingw_' +  str(arguments.use_vampmixingw) + \
              '_separate_means_' + str(arguments.separate_means) + \
              '_useclassifier_' + str(arguments.use_classifier) + \
              '_semi_sup_' +str(arguments.semi_sup) + \
              '_Lambda_' + str(arguments.Lambda) + \
              '_use_mixingw_correction_' + str(arguments.use_mixingw_correction) + \
              '_use_replaycostcorrection_' + str(arguments.use_replaycostcorrection) + \
              arguments.notes

results_name = arguments.dataset_name + '_' + exp_details
print(results_name)

# load the permutations
permutations = torch.load('mnistpermutations_seed2_cdr305.int.cedar.computecanada.ca_2019-01-0613:31:06.234041.t')
perm = permutations[arguments.permindex]

model = VAE(arguments).cuda()
if arguments.use_classifier:
    classifier = cls(arguments, 100, 784).cuda()
else: 
    classifier = None

# implement proper sampling with vamp, learn the weights too.  
for dg in range(0, 10):

    print('\n________________________')
    print('starting task {}'.format(dg))
    print('________________________\n')


    train_loader = ut.get_mnist_loaders([int(perm[dg].item())], 'train', arguments)
    val_loader = ut.get_mnist_loaders(list(perm[list(range(dg+1))].numpy().astype('int')), 'validation', arguments)
    test_loader = ut.get_mnist_loaders(list(perm[list(range(dg+1))].numpy().astype('int')), 'test', arguments)
    
    model_name = arguments.dataset_name + str(dg) + '_' + exp_details
    dr = files_path + model_name 

    model_path = cwd + files_path + model_name + '.model'
    if dg == 0:
        prev_model = None
        prev_classifier = None

    if arguments.use_classifier:
        optimizer_cls = AdamNormGrad(classifier.parameters(), lr=arguments.lr)

        tr.train_classifier(arguments, train_loader, classifier=classifier, 
                            prev_classifier=prev_classifier,
                            prev_model=prev_model,
                            optimizer_cls=optimizer_cls, dg=dg, perm=perm)

        acc, all_preds = ev.evaluate_classifier(arguments, classifier, test_loader)        
        print('Digits upto {}, accuracy {}'.format(dg, acc.item()))

    if arguments.debug and os.path.exists(model_path):
        print('loading model... for digit {}'.format(dg))

        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        EPconv = t1 = t2 = 0   
    else:
        print('training model... for digit {}'.format(dg))
        optimizer = AdamNormGrad(model.parameters(), lr=arguments.lr)
        model = model.cuda()
        t1 = time.time()
        EPconv = tr.experiment_vae(arguments, train_loader, val_loader, test_loader, model, 
                                   optimizer, dr, arguments.model_name, prev_model=prev_model, 
                                   dg=dg, perm=perm) 
        t2 = time.time()

    if (arguments.use_classifier or arguments.semi_sup) and arguments.use_mixingw_correction and (arguments.prior != 'standard'):
        if arguments.use_classifier:
            yhat_means = model.balance_mixingw(classifier, dg=dg, perm=perm)
        if arguments.semi_sup:
            yhat_means = model.balance_mixingw(dg=dg, perm=perm)
        if arguments.use_visdom:
            means = model.reconstruct_means()
            opts = {}
            opts['title'] = 'current means'
            vis.images(means, win='means_cur', opts=opts)
            vis.text(str(model.mixingw_c), win='mixingw')
            vis.text(str(yhat_means), win='yhat_means')

                    
    if (dg > 0) and arguments.separate_means:
        model.merge_latent()
    print('evaluating the model...')

    # when doing the hyperparameter search, pay attention to what results you are saving
    if 1: 
        results = ev.evaluate_vae(arguments, model, train_loader, test_loader, 0, results_path, 'test', use_mixw_cor=arguments.use_mixingw_correction)
        results['digit'] = dg
        if arguments.use_classifier: results['class'] = acc.item()
        results['time'] = t2 - t1
        results['epochs'] = EPconv
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

    if arguments.use_classifier:
        prev_classifier = copy.deepcopy(classifier)
    

    # little questionable 
    if arguments.add_cap and (dg < 9) and (arguments.prior != 'standard'):
        model.add_latent_cap(dg)
    else:
        model.restart_latent_space()

