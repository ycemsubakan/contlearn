from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
import torch.nn.functional as F

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256, log_mog_diag, log_mog_full
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
import itertools as it
import sklearn.mixture as mix
import pdb
import os
import pickle
import copy

from utils.nn import he_init, GatedDense, NonLinear, \
    Conv2d, GatedConv2d, GatedResUnit, ResizeGatedConv2d, MaskedConv2d, ResUnitBN, ResizeConv2d, GatedResUnit, GatedConvTranspose2d


class SSVAE(Model):
    def __init__(self, args):
        super(SSVAE, self).__init__(args)

        assert self.args.prior != 'GMM'   
        assert self.args.prior != 'vampprior'   
        assert not self.args.separate_means 
    
        # encoder: q(z | x)
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), 300),
            GatedDense(300, 300)
        )

        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedDense(self.args.z1_size, 300),
            GatedDense(300, 300)
        )

        #if self.args.input_type == 'binary':
        self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), 
                                     activation=nn.Sigmoid())
        #elif self.args.input_type in ['gray', 'continuous', 'color']:
        #    self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
        #    self.p_x_logvar = NonLinear(300, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))
        self.mixingw_c = np.ones(self.args.number_components)

        #self.semi_supervisor = nn.Sequential(Linear(args.z1_size, args.num_classes), 
        #                                         nn.Softmax())
        self.semi_supervisor = nn.Sequential(GatedDense(args.z1_size, args.z1_size),
                                                 nn.Dropout(0.5),
                                                 GatedDense(args.z1_size, args.num_classes),
                                                 nn.Softmax())
        
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior in ['vampprior', 'vampprior_short']:
            self.add_pseudoinputs()

    def restart_latent_space(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.args.number_components_init, 300, bias=False, activation=nonlinearity)

        if self.args.use_vampmixingw:
            self.mixingw = NonLinear(self.args.number_components_init, 1, bias=False, activation=nn.Softmax(dim=0))

    def merge_latent(self):
        # always to be called after separate_latent() or add_latent_cap()
        nonlinearity = None #nn.Hardtanh(min_val=0.0, max_val=1.0)
        
        prev_weights = self.means[0].linear.weight.data
        last_prev_weights = self.means[1].linear.weight.data 
        all_old = torch.cat([prev_weights, last_prev_weights], dim=1)

        self.means[0] = NonLinear(all_old.size(1), 300, bias=False, activation=nonlinearity).cuda()
        self.means[0].linear.weight.data = all_old

    def separate_latent(self):
        # always to be called after merge_latent()
        nonlinearity = None #nn.Hardtanh(min_val=0.0, max_val=1.0)

        number_components_init = self.args.number_components_init
        number_components_prev = (self.args.number_components) - number_components_init
        
        prev_components = copy.deepcopy(self.means[0].linear.weight.data[:, :number_components_prev:])

        last_prev_components = copy.deepcopy(self.means[0].linear.weight.data[:, number_components_prev:])

        self.means[0] = NonLinear(number_components_prev, 300, bias=False, activation=nonlinearity).cuda()
        self.means[1] = NonLinear(number_components_init, 300, bias=False, activation=nonlinearity).cuda()
        self.means[0].linear.weight.data = prev_components
        self.means[1].linear.weight.data = last_prev_components

    def add_latent_cap(self, dg):
        if self.args.prior == 'vampprior_short':
            nonlinearity = None #nn.Hardtanh(min_val=0.0, max_val=1.0)
                
            number_components_prev = copy.deepcopy(self.args.number_components) 
            self.args.number_components = (dg+2)*copy.deepcopy((self.args.number_components_init))

            if self.args.separate_means:    
                add_number_components = self.args.number_components - number_components_prev
                # set the idle inputs 
                #self.idle_input = torch.eye(self.args.number_components,
                #                            self.args.number_components).cuda()
                #self.idle_input1 = torch.eye(add_number_components,
                #                             add_number_components).cuda()
                #self.idle_input2 = torch.eye(number_components_prev,
                #                             number_components_prev).cuda()

                us_new = NonLinear(add_number_components, 300, bias=False, activation=nonlinearity).cuda()
                #us_new.linear.weight.data = 0*torch.randn(300, add_number_components).cuda()
                if dg == 0:
                    self.means.append(us_new)
                else:
                    # self.merge_latent() - nope, because we do this because validation evaluation (in main_mnist.py)
                    self.means[1] = us_new

            else:
                self.idle_input = torch.eye(self.args.number_components,
                                            self.args.number_components).cuda()

                us_new = NonLinear(self.args.number_components, 300, bias=False, activation=nonlinearity).cuda()
                if not self.args.restart_means:
                    oldweights = self.means.linear.weight.data 
                    us_new.linear.weight.data[:, :number_components_prev] = oldweights 
                self.means = us_new

                if self.args.use_vampmixingw:
                    self.mixingw = NonLinear(self.args.number_components, 1, bias=False, 
                                             activation=nn.Softmax(dim=0)) 

    def reconstruct_means(self, head=None):
        K = self.means.linear.weight.size(1)
        eye = torch.eye(K, K)
        if self.args.cuda: eye = eye.cuda()

        X = self.means(eye)
        z_mean = self.q_z_mean(X)

        recons, _ = self.p_x(z_mean)

        return recons 

    def balance_mixingw(self, dg, perm, dont_balance=False, vis=None):

        # get means
        K = self.means.linear.weight.size(1)
        eye = torch.eye(K, K)
        if self.args.cuda:
            eye=eye.cuda() 
        X = self.means(eye)
        z_mean = self.q_z_mean(X)
        yhat_means = self.semi_supervisor(z_mean)
        
        pis = self.mixingw(self.idle_input).squeeze()
    
        # to numpy:
        if self.args.cuda:
            y_hat_means = yhat_means.detach().cpu().numpy()
            pis = pis.detach().cpu().numpy()
        else:
            y_hat_means = yhat_means.detach().numpy()
            pis = pis.detach().numpy()
        perm = perm.numpy()
        curr_per_class_weight = np.matmul(pis, y_hat_means)

        print('\ncurrent per class cluster assignment:')
        print(np.round(curr_per_class_weight,2))
        print('\n')

        if dont_balance:
            return yhat_means, curr_per_class_weight
        
        mixingw_c = np.zeros(self.args.number_components)
        per_class_scaling = np.zeros(self.args.num_classes)
        for d in range(dg+1):
            idx = int(perm[d])
            per_class_scaling[idx] = 1 / curr_per_class_weight[idx] / (dg+1)
        
        self.mixingw_c = np.matmul(y_hat_means, per_class_scaling)

        print('\nrebalanced per class cluster assignment:')
        post_per_class_weight = np.matmul(self.mixingw_c*pis, y_hat_means)
        print(np.round(post_per_class_weight,2))
        print('\n')

        return yhat_means, curr_per_class_weight, post_per_class_weight

    def calculate_loss(self, x, y=None, beta=1., average=False, head=None):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar, y_hat = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type in ['gray', 'color']: 
            #RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
            RE = - (x - x_mean).abs().sum(dim=1)

        #elif self.args.input_type == 'color':
        #    RE = -log_Normal_diag(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        # loss
        loss = - RE + beta * KL #+ self.args.Lambda * CE
       
        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)
        
        if y is None:
            return loss, RE, KL, x_mean
        
        # CE
        if len(y.shape)==1:
            CE =  F.nll_loss(torch.log(y_hat), y)
        else:
            CE = - (y * torch.log(y_hat)).mean()
        
        # loss
        loss += self.args.Lambda * CE

        if average:
            CE = torch.mean(CE)

        return loss, RE, KL, CE, x_mean

    def calculate_lower_bound(self, X_full, MB=100):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        # dealing the case where the last batch is of size 1
        remainder = X_full.size(0) % MB
        if remainder == 1:
            X_full = X_full[:(X_full.size(0) - remainder)]

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            if not self.args.dataset_name == 'celeba':
                x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))
            else:
                x = X_full[i * MB: (i + 1) * MB]

            loss, RE, KL, _ = self.calculate_loss(x, average=True)

            #RE_all += RE.item()
            #KL_all += KL.item()
            lower_bound += loss

        lower_bound /= I

        return lower_bound
    
    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):

        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, int(R)):
                # Repeat it for all training points
                if self.args.dataset_name == 'celeba':
                    x = x_single.expand(S, x_single.size(1), x_single.size(2), x_single.size(3))
                else:
                    x = x_single.expand(S, x_single.size(1))

                a_tmp, _, _, _ = self.calculate_loss(x)

                a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        #plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_accuracy(self, X_full, y_full, MB=100):
        # CALCULATE ACCURACY:
        acc = 0.

        # dealing the case where the last batch is of size 1
        remainder = X_full.size(0) % MB
        if remainder == 1:
            X_full = X_full[:(X_full.size(0) - remainder)]
            y_full = y_full[:(y_full.size(0) - remainder)]

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            if not self.args.dataset_name == 'celeba':
                x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))
                y = y_full[i * MB: (i + 1) * MB]
            else:
                x = X_full[i * MB: (i + 1) * MB]
                y = y_full[i * MB: (i + 1) * MB]
            
            _, _, _, _, _, y_hat = self.forward(x)
            _, predicted = torch.max(y_hat.data, 1)

            correct = (predicted == y).sum().item()

            acc += correct

        acc /= I

        return acc

    # ADDITIONAL METHODS
    def generate_x(self, N=25, replay=False):

        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

            samples_rand, _ = self.p_x(z_sample_rand)
        elif self.args.prior == 'vampprior':
            clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())

            means = self.means(self.idle_input)
            if self.args.dataset_name == 'celeba':
                means = means.reshape(means.size(0), 3, 64, 64)
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

            samples_rand, _ = self.p_x(z_sample_rand)

            # do a random permutation to see a more representative sampleset

            randperm = torch.randperm(samples_rand.size(0))
            samples_rand = samples_rand[randperm][:N]

        elif self.args.prior == 'vampprior_short':
            if self.args.use_vampmixingw:
                pis = self.mixingw(self.idle_input).squeeze()
            else:
                pis = torch.ones(self.args.number_components) / self.args.number_components
            
            if self.args.use_mixingw_correction and replay:
                mixingw_c = torch.from_numpy(self.mixingw_c)
                if self.args.cuda: mixingw_c = mixingw_c.type(torch.cuda.FloatTensor)
                else: mixingw_c = mixingw_c.type(torch.FloatTensor)
                pis = mixingw_c * pis

            clsts = np.random.choice(range(self.args.number_components), N, 
                                     p=pis.data.cpu().numpy())

            K = self.means.linear.weight.size(1)
            eye = torch.eye(K, K)
            if self.args.cuda: eye = eye.cuda()

            means = self.means(eye)[clsts, :]

            # if used in the separated means case, always use the first head. Therefore you need to merge the means before generation 
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z_mean(means), self.q_z_logvar(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            
            samples_rand, _ = self.p_x(z_sample_rand)

            # do a random permutation to see a more representative sampleset
            #randperm = torch.randperm(samples_rand.size(0))
            #samples_rand = samples_rand[randperm][:N]

        # generate soft labels:
        y_rand = self.semi_supervisor(z_sample_rand) 

        return samples_rand, y_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.q_z_layers(x)

        x = x.squeeze()

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):

        if self.args.dataset_name == 'celeba':
            z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.p_x_layers(z)

        x_mean = self.p_x_mean(z)
        
        #if self.args.input_type == 'binary':
        x_logvar = 0.
        #else:
        #x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)

            #x_logvar = self.p_x_logvar[head](z)

        x_mean = x_mean.reshape(x_mean.size(0), -1)
        if self.args.dataset_name == 'celeba':
            x_logvar = x_logvar.reshape(x_logvar.size(0), -1)
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z, mu=None, logvar=None):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means(self.idle_input)
            if self.args.dataset_name == 'celeba':
                X = X.reshape(X.size(0), 3, 64, 64)

            # calculate params for given data
            z_p_mean, z_p_logvar = self.q_z(X)  # C x M

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            logvars = z_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        elif self.args.prior == 'vampprior_short':
            C = self.args.number_components

            K = self.means.linear.weight.size(1)
            eye = torch.eye(K, K)
            if self.args.cuda:
                eye = eye.cuda()

            X = self.means(eye)

            z_p_mean = self.q_z_mean(X)
            z_p_logvar = self.q_z_logvar(X)

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            logvars = z_p_logvar.unsqueeze(0)

            # havent yet implemented dealing with mixing weights in the separated means case
            if self.args.use_vampmixingw:
                pis = self.mixingw(eye).t()
                eps = 1e-30
                a = log_Normal_diag(z_expand, means, logvars, dim=2) + torch.log(pis)  # MB x C
            else:
                a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C

            a_max, _ = torch.max(a, 1)  # MB x 1
            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        else:
            raise Exception('invalid prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)
    
        y_hat = self.semi_supervisor(z_q_mean)
        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar, y_hat

    def get_embeddings(self, train_loader, cuda=True, flatten=True):
        # get hhats for all batches
        if self.args.dataset_name == 'celeba':
            nbatches = 300
        else:
            nbatches = 1000

        all_hhats = []
        for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            if cuda:
                data = data.cuda()

            if self.args.dataset_name != 'celeba':
                data = data.view(-1, np.prod(self.args.input_size))

            mu, logvar = self.q_z(data)
            hhat = torch.randn(mu.size()).cuda()*(0.5*logvar).exp() + mu

            all_hhats.append(hhat.data.squeeze())
            print('processing batch {}'.format(i))

        return torch.cat(all_hhats, dim=0)



