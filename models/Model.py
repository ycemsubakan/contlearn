from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.nn import normal_init, NonLinear
import copy
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS
    def add_pseudoinputs(self, separate_means=True):

        self.args.number_components = copy.deepcopy(self.args.number_components_init)
        nonlinearity = None #nn.Hardtanh(min_val=0.0, max_val=1.0)

        if self.args.prior == 'vampprior': 
            self.means = NonLinear(self.args.number_components_init, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        elif self.args.prior == 'vampprior_short':
            self.means = NonLinear(self.args.number_components_init, self.Khid, bias=False, activation=nonlinearity)

        if self.args.use_vampmixingw:
            self.mixingw = NonLinear(self.args.number_components_init, 1, bias=False, activation=nn.Softmax(dim=0))


        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

        if self.args.separate_means:
            self.means = nn.ModuleList([self.means])

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.args.number_components_init, self.args.number_components_init), requires_grad=False)
        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()


    def initialize_GMMparams(self, Kmog=None, GMM=None, mode='GMMinit'): 
        if mode == 'random':
            self.Kmog = Kmog
            self.mus = nn.Parameter(0.01*torch.randn(self.Ks[0], Kmog))
            self.sigs = nn.Parameter(torch.ones(self.Ks[0], Kmog))
            self.pis = nn.Parameter(torch.ones(Kmog)/Kmog)
        elif mode == 'GMMinit':
            self.args.prior = 'GMM'
            self.mus = nn.Parameter(torch.from_numpy(self.GMM.means_).t().float())
            self.sigs = nn.Parameter(torch.from_numpy(self.GMM.covariances_).t().float())
            self.pis = nn.Parameter(torch.from_numpy(self.GMM.weights_).float())
            self.Kmog = self.pis.size(0)
        else:
            raise ValueError('What Mode?')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

#=======================================================================================================================
