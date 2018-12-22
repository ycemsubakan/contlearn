import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_extras import GatedDense as dense
import torch.optim as optim
import itertools as it
from torch.autograd import Variable
import pdb

class VAE(nn.Module):
    def __init__(self, L1, L2, Ks, M, outlin='sigmoid', use_gates=False):
        super(VAE, self).__init__()

        self.L1 = L1
        self.L2 = L2
        self.Ks = Ks
        self.M = M
        self.base_dist = 'fixed_iso_gauss'
        self.outlin = outlin
        self.gated = use_gates 

        if use_gates:
            nonlin = 'sigmoid'
            self.fc1 = dense(self.L1, self.Ks[1], activation=nonlin)
            #initializationhelper(self.fc1, 'relu')

            self.fc21 = dense(self.Ks[1], self.Ks[0], activation=nonlin)
            #initializationhelper(self.fc21, 'relu')

            self.fc22 = dense(self.Ks[1], self.Ks[0], activation=nonlin)
            #initializationhelper(self.fc22, 'relu')

            self.fc3 = dense(self.Ks[0], self.Ks[1], activation=nonlin)
            #initializationhelper(self.fc3, 'relu')

            self.fc4 = dense(self.Ks[1], self.L2, activation=nonlin)
            #initializationhelper(self.fc4, 'relu')

        else:
            self.fc1 = nn.Linear(self.L1, self.Ks[1])
            #initializationhelper(self.fc1, 'relu')

            self.fc21 = nn.Linear(self.Ks[1], self.Ks[0])
            #initializationhelper(self.fc21, 'relu')

            self.fc22 = nn.Linear(self.Ks[1], self.Ks[0])
            #initializationhelper(self.fc22, 'relu')

            self.fc3 = nn.Linear(self.Ks[0], self.Ks[1])
            #initializationhelper(self.fc3, 'relu')

            self.fc4 = nn.Linear(self.Ks[1], self.L2)
            #initializationhelper(self.fc4, 'relu')

            
    def initialize_GMMparams(self, GMM=None, mode='GMMinit'): 
        if mode == 'random':
            Kmog = 10

            self.Kmog = Kmog
            self.mus = nn.Parameter(1*torch.randn(self.Ks[0], Kmog))
            self.sigs = nn.Parameter(torch.ones(self.Ks[0], Kmog))
            self.pis = nn.Parameter(torch.ones(Kmog)/Kmog)
        elif mode == 'GMMinit':
            self.mus = nn.Parameter(torch.from_numpy(GMM.means_).t().float())
            self.sigs = nn.Parameter(torch.from_numpy(GMM.covariances_).t().float())
            self.pis = nn.Parameter(torch.from_numpy(GMM.weights_).float())
            self.Kmog = self.pis.size(0)
        else:
            raise ValueError('What Mode?')


    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z1 = F.tanh(self.fc3(z))
        if self.outlin == 'sigmoid':
            return F.sigmoid(self.fc4(z1))
        else:
            return self.fc4(z1)

    def forward(self, inp):

        mu, logvar = self.encode(inp)
        h = self.reparameterize(mu, logvar)

        #print('mean of mu {} variance of mu {}'.format(torch.mean(h).data[0], torch.var(h).data[0]))
        return self.decode(h), mu, logvar, h

    def criterion(self, recon_x, x, mu, logvar):
        eps = 1e-20
        #criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)
        recon_x = recon_x.view(-1, self.L2)
        x = x.view(-1, self.L2)

        #crt = lambda xhat, tar: torch.sum(((xhat - tar).abs()), 1)
        #mask = torch.ge(recon_x, 1).float()
        #mask2 = torch.le(recon_x, 0).float()
        #recon_x = mask*(1-eps) + (1-mask)*recon_x
        #recon_x = mask2*eps + (1-mask)*recon_x
        crt = lambda xhat, tar: -torch.sum(tar*torch.log(xhat+eps) + (1-tar)*torch.log(1-xhat+eps), 1)

        BCE = crt(recon_x, x)
        v = 1
        KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
        # Normalise by same number of elements as in reconstruction
        #KLD = KLD /(x.size(0) * x.size(1))
        return BCE + KLD

    def generate_data(self, N, base_dist='fixed_iso_gauss'):

        if base_dist == 'fixed_iso_gauss':
            seed = torch.randn(N, self.Ks[0]) 
            if next(self.parameters()).is_cuda:
            #self is self.cuda(): 
                seed = seed.cuda()
            seed = Variable(seed)
            gen_data = self.decode(seed)
            return gen_data, seed
        elif base_dist == 'mog':
            clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())
            mus = self.mus[:, clsts]
            randn = torch.randn(mus.size())
            if next(self.parameters()).is_cuda:
                randn = randn.cuda()

            zs = mus + (self.sigs[:, clsts].sqrt())*randn
            gen_data = self.decode(zs.t())
            return gen_data, zs
        elif base_dist == 'mog_skt':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()

            if self.cuda:
                seed = seed.cuda()
            return self.decode(seed), seed
        elif base_dist == 'mog_cuda':
            seed = self.GMM.sample(N)
            return self.decode(seed), seed
        else:
            raise ValueError('what base distribution?')

    def VAE_trainer(self, cuda, train_loader, 
                    EP = 400,
                    vis = None, config_num=0, **kwargs):

        try:
            optimizer = kwargs['optimizer']
        except:
            optimizer = 'Adam'

        try:
            replay_gen = kwargs['replay_gen']
        except:
            replay_gen = None

        self.train(mode=True)

        L1 = self.L1
        L2 = self.L2
        Ks = self.Ks

        lr = 1e-4
        if optimizer == 'Adam':
            optimizerG = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))
        elif optimizer == 'RMSprop':
            optimizerG = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            optimizerG = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'LBFGS':
            optimizerG = optim.LBFGS(self.parameters(), lr=lr)

        nbatches = 1400
        for ep in range(EP):
            for i, (tar, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                if cuda:
                    tar = tar.cuda()

                #if 1:
                #    tar = tar[:, :2, :, :]

                tar = tar.view(-1, L2)
                tar = Variable(tar)

                if replay_gen is not None: 
                    gen_data = replay_gen.generate_data(tar.size(0))
                    tar = torch.cat([gen_data[0], tar], dim=0)
                    inds = torch.from_numpy(np.random.choice(tar.size(0), tar.size(0), replace=False)).cuda()
                    tar = torch.index_select(tar, dim=0, index=inds)


                # generator gradient
                self.zero_grad()
                out_g, mu, logvar, h = self.forward(tar)
                err_G = self.criterion(out_g, tar, mu, logvar)
                err_G = err_G.mean(0)

                err_G.backward()

                # step 
                optimizerG.step()
                                                                                       
                
                if (i == 0) and (ep % 100) == 0:
                    # visdom plots
                    
                    # generate samples first
                    self.eval()
                    self.train(mode=False)
                    N = 100
                    gen_data, seed = self.generate_data(N)
                    self.train(mode=True)

                    print('EP [{}/{}], error = {}, batch = [{}/{}], config num {}'.format(ep+1, EP, err_G.data[0], i+1, len(train_loader), config_num))


                    if 1: 
                        Nims = 64
                        sz = 800
                        tar = tar.view(-1, 1, self.M, self.M)

                        a, b = 1, 0
                        opts={}
                        opts['title'] = 'VAE Approximations'
                        vis.images(a*out_g.data.cpu().reshape(tar.shape)[:Nims] + b, opts=opts, win='vae_approximations')
                        opts['title'] = 'VAE Input images'
                        vis.images(a*tar.data.cpu()[:Nims] + b, opts=opts, win='vae_x')

                        opts['title'] = 'VAE Generated images'
                        vis.images(a*gen_data.data.cpu().reshape(N, 1, self.M, self.M)[Nims:] + b, opts=opts, win='vae_gendata')

                    
        return h






