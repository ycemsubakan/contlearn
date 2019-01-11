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


class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.q_z_layers = nn.ModuleList([nn.Sequential(
            GatedDense(np.prod(self.args.input_size), 300),
            GatedDense(300, 300)
        )])

        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedDense(self.args.z1_size, 300),
            GatedDense(300, 300)
        )

        #if self.args.input_type == 'binary':
        self.p_x_mean = nn.ModuleList([NonLinear(300, np.prod(self.args.input_size), 
                                                 activation=nn.Sigmoid())])
        #elif self.args.input_type in ['gray', 'continuous', 'color']:
        #    self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
        #    self.p_x_logvar = NonLinear(300, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))
        self.mixingw_c = np.ones(self.args.number_components)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior in ['vampprior', 'vampprior_short']:
            self.add_pseudoinputs()
        elif self.args.prior == 'GMM': 
            self.initialize_GMMparams(Kmog=10, mode='random') 

    def add_head(self, input_size): 
        q_z_layers_new = nn.Sequential(
            GatedDense(np.prod(input_size), 300),
            GatedDense(300, 300)
        )
        self.q_z_layers.append(q_z_layers_new)

        #elif self.args.input_type in ['gray', 'continuous', 'color']:
            #p_x_mean_new = NonLinear(300, np.prod(input_size), activation=nn.Sigmoid())

            #logvar_layers = [self.p_x_logvar]
            #self.p_x_logvar_new = NonLinear(300, np.prod(input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))
            #logvar_layers.append(self.p_x_logvar_new)
            #self.p_x_logvar = nn.ModuleList(logvar_layers)

        p_x_mean_new = NonLinear(300, np.prod(input_size), activation=nn.Sigmoid())
        self.p_x_mean.append(p_x_mean_new)

        if self.args.prior == 'vampprior':
            nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
            us_new = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
            self.means.append(us_new)
        elif self.args.prior == 'vampprior_joint':
            nonlinearity = None #nn.Hardtanh(min_val=0.0, max_val=1.0)
            
            us_new = NonLinear(2*self.args.number_components, 300, bias=False, activation=nonlinearity)
            oldweights = self.means.linear.weight.data 
            us_new.linear.weight.data[:, :self.args.number_components] = oldweights 
            self.means = us_new

            # fix the idle input size also
            self.idle_input = torch.eye(2*self.args.number_components,
                                        2*self.args.number_components).cuda()


        self.extra_head = True

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

    def reconstruct_means(self, head=0):
        if self.args.separate_means:
            K = self.means[head].linear.weight.size(1)
            eye = torch.eye(K, K).cuda()

            X = self.means[head](eye)
        else: 
            K = self.means.linear.weight.size(1)
            eye = torch.eye(K, K).cuda()

            X = self.means(eye)
        z_mean = self.q_z_mean(X)

        recons, _ = self.p_x(z_mean, head=0)

        return recons 

    def balance_mixingw(self, classifier, dg, perm=torch.arange(10), vis=None):
        # functions that are related: 
        # training.train_classifier
        # evaluate.

        means = self.reconstruct_means()
        yhat_means = torch.argmax(classifier.forward(means), dim=1)

        mixingw_c = torch.zeros(self.args.number_components, 1).squeeze().cuda()
        ones = torch.ones(self.args.number_components).squeeze().cuda()

        for d in perm[:(dg+1)]:
            mask = (yhat_means == int(d.item()))
            pis = self.mixingw(self.idle_input).squeeze()
            pis_select = torch.masked_select(pis, mask)
            sm = pis_select.sum()

            # correct the mixing weights 
            mixingw_c = mixingw_c + ones*(mask.float())/(sm*(dg+1))

        self.mixingw_c = mixingw_c.data.cpu().numpy()
        return yhat_means

    def compute_class_entropy(self, classifier, dg):
        means = self.reconstruct_means()
        yhat_means = classifier.forward(means) # needs softmax 

        pis = self.mixingw(self.idle_input)
        pdb.set_trace()

        ws = torch.matmul(yhat_means, pis) 
        return ent(ws[perm[:dg+1]])




    def calculate_loss(self, x, beta=1., average=False, head=0, use_mixw_cor=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        
        # pass through VAE
        fw_head = min(head, len(self.q_z_layers)-1)

        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x, head=fw_head)

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
        if self.args.prior == 'GMM':
            log_p_z = self.log_p_z(z_q, mu=z_q_mean, logvar=z_q_logvar)
        else:
            log_p_z = self.log_p_z(z_q, head=head, use_mixw_cor=use_mixw_cor)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL, x_mean

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100, use_mixw_cor=False):
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

                a_tmp, _, _, _ = self.calculate_loss(x, use_mixw_cor=use_mixw_cor)

                a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        #plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=100, use_mixw_cor=False):
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

            loss, RE, KL, _ = self.calculate_loss(x,average=True, use_mixw_cor=use_mixw_cor)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]
            lower_bound += loss.cpu().data[0]

        lower_bound /= I

        return lower_bound

    # ADDITIONAL METHODS
    def generate_x(self, N=25, head=0, replay=False):
        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

            samples_rand, _ = self.p_x(z_sample_rand, head=head)
        elif self.args.prior == 'vampprior':
            clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())

            means = self.means[head](self.idle_input)
            if self.args.dataset_name == 'celeba':
                means = means.reshape(means.size(0), 3, 64, 64)
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means, head=head)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

            samples_rand, _ = self.p_x(z_sample_rand, head=head)

            # do a random permutation to see a more representative sampleset

            randperm = torch.randperm(samples_rand.size(0))
            samples_rand = samples_rand[randperm][:N]

        elif self.args.prior == 'vampprior_short':
            if self.args.use_vampmixingw:
                pis = self.mixingw(self.idle_input).squeeze()
            else:
                pis = torch.ones(self.args.number_components) / self.args.number_components
            
            if self.args.use_mixingw_correction and replay:
                pis = torch.from_numpy(self.mixingw_c).cuda() * pis

            clsts = np.random.choice(range(self.args.number_components), N, 
                                     p=pis.data.cpu().numpy())

            if self.args.separate_means:
                K = self.means[head].linear.weight.size(1)
                eye = torch.eye(K, K).cuda()

                means = self.means[0](eye)[clsts, :]
            else: 
                K = self.means.linear.weight.size(1)
                eye = torch.eye(K, K).cuda()

                means = self.means(eye)[clsts, :]

            # if used in the separated means case, always use the first head. Therefore you need to merge the means before generation 
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z_mean(means), self.q_z_logvar(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            
            samples_rand, _ = self.p_x(z_sample_rand, head=head)

            # do a random permutation to see a more representative sampleset
            #randperm = torch.randperm(samples_rand.size(0))
            #samples_rand = samples_rand[randperm][:N]
        elif self.args.prior == 'GMM':
            if self.GMM.covariance_type == 'diag':
                clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())
                mus = self.mus[clsts, :]
                randn = torch.randn(mus.size())
                if next(self.parameters()).is_cuda:
                    randn = randn.cuda()

                zs = mus + (self.sigs[clsts, :].sqrt())*randn
            elif self.GMM.covariance_type == 'full':
                Us = [torch.svd(cov)[0].mm(torch.sqrt(torch.svd(cov)[1]).diag()).unsqueeze(0) for cov in self.sigs]
                Us = torch.cat(Us, dim=0)
                
                noise = torch.randn(N, self.mus.size(1), 1).cuda()
               
                clsts = (torch.from_numpy(np.random.choice(range(self.mus.size(0)), size=N, p=self.pis.detach().cpu().numpy())).type(torch.LongTensor)).cuda()
                Us_zs = torch.index_select(Us, dim=0, index=clsts)
                means_zs = torch.index_select(self.mus, dim=0, index=clsts)

                zs = torch.matmul(Us_zs, noise).squeeze() + means_zs
            samples_rand, _ = self.p_x(zs)

        return samples_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x, head=0):
        x = self.q_z_layers[head](x)

        x = x.squeeze()

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z, head=0):

        if self.args.dataset_name == 'celeba':
            z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.p_x_layers(z)

        x_mean = self.p_x_mean[head](z)
        
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
    def log_p_z(self, z, mu=None, logvar=None, head=0, use_mixw_cor=False):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means[head](self.idle_input)
            if self.args.dataset_name == 'celeba':
                X = X.reshape(X.size(0), 3, 64, 64)

            # calculate params for given data
            z_p_mean, z_p_logvar = self.q_z(X, head=head)  # C x M

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

            # calculate params
            if self.args.separate_means:
                K = self.means[head].linear.weight.size(1)
                eye = torch.eye(K, K).cuda()

                X = self.means[head](eye)
            else: 
                K = self.means.linear.weight.size(1)
                eye = torch.eye(K, K).cuda()

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
                if use_mixw_cor: 
                    pis = pis * torch.from_numpy(self.mixingw_c).cuda().unsqueeze(0) 
                eps = 1e-30
                a = log_Normal_diag(z_expand, means, logvars, dim=2) + torch.log(pis)  # MB x C
            else:
                a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C

            a_max, _ = torch.max(a, 1)  # MB x 1
            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        elif self.args.prior == 'GMM':
            if self.GMM.covariance_type == 'full':
                #z_np = z.data.cpu().numpy()
                #lls = self.GMM.score_samples(z_np)
                #log_prior = torch.from_numpy(lls).float().cuda() + math.log(2*math.pi)*(0.5*z.size(1))
                log_prior = log_mog_full(z, self.mus, self.sigs, self.pis, 
                                         self.icovs_ten, self.det_terms)

            else:
                log_prior = log_mog_diag(z, self.mus, self.sigs, self.pis)
        else:
            raise Exception('invalid prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x, head=0):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x, head=head)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q, head=head)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar


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

    def fit_GMM(self, train_loader, Kmog, cov_type='diag', model_name=None):
        self.args.prior = 'GMM'
        self.Kmog = Kmog
        hhat = self.get_embeddings(train_loader)
        
        path = 'gmm_params/'
        if not os.path.exists(path + self.args.dataset_name):
            if not os.path.exists(path):
                os.mkdir(path)
            os.mkdir(path + self.args.dataset_name)
        
        gmm_path = path + model_name + 'gmm_' + cov_type + '.pk' 
        if 1 & os.path.exists(gmm_path):
            self.GMM = pickle.load(open(gmm_path, 'rb'))
        else:
            self.GMM = mix.GaussianMixture(n_components=Kmog, verbose=1, n_init=10, max_iter=200, covariance_type=cov_type, warm_start=True)
            self.GMM.fit(hhat.cpu().numpy())
            pickle.dump(self.GMM, open(gmm_path, 'wb'))

        # then initialize the GMM, and change self.args.prior
        self.mus = nn.Parameter(torch.from_numpy(self.GMM.means_).float())
        self.pis = nn.Parameter(torch.from_numpy(self.GMM.weights_).float())
        if cov_type == 'diag':
            self.sigs = nn.Parameter(torch.from_numpy(self.GMM.covariances_).float())
        else:
            self.sigs = torch.from_numpy(self.GMM.covariances_).float().cuda()
            self.icovs_ten = (torch.zeros(self.pis.size(0), self.mus.size(1), self.mus.size(1))).cuda()
            cov_eps = 1e-10
            for k in range(self.pis.size(0)):
                self.icovs_ten[k, :, :] = torch.inverse(self.sigs[k, :, :] + torch.eye(self.mus.size(1)).cuda()*cov_eps)
            
            self.det_terms = torch.Tensor([0.5*(cov_eps + torch.svd(cov.squeeze())[1]).log().sum() for cov in self.sigs]).cuda()

        self.Kmog = self.pis.size(0)
        self.args.prior = 'GMM'


class classifier(nn.Module):
    def __init__(self, args, K, L, Lclass=10):
        super(classifier, self).__init__()
        
        self.K = K
        self.L = L
        self.args = args
        
        activation = nn.ReLU()
        self.layer = nn.Sequential(
            GatedDense(L, K, activation=activation),
            GatedDense(K, Lclass, activation=None)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class conv_vae(VAE):
    def __init__(self, args):
        super(conv_vae, self).__init__(args)

        # encoder: q(z | x)
        d = 32
        act = F.relu 
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], d, 4, 2, 1, activation=act),
            GatedConv2d(d, 2*d, 4, 2, 1, activation=act),
            GatedConv2d(2*d, 4*d, 4, 2, 1, activation=act),
            GatedConv2d(4*d, 8*d, 4, 2, 1, activation=act),
            GatedConv2d(8*d, self.args.z1_size, 4, 1, 0, activation=None)
        )

        self.q_z_mean = Linear(self.args.z1_size, self.args.z1_size)
        self.q_z_logvar = NonLinear(self.args.z1_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedConvTranspose2d(self.args.z1_size, 8*d, 4, 1, 0, activation=act),
            GatedConvTranspose2d(8*d, 4*d, 4, 2, 1, activation=act),
            GatedConvTranspose2d(4*d, 2*d, 4, 2, 1, activation=act),
            GatedConvTranspose2d(2*d, d, 4, 2, 1, activation=act),
            GatedConvTranspose2d(d, d, 4, 2, 1, activation=None)
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(d, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(d, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(d, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

        
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()
        elif self.args.prior == 'GMM': 
            self.initialize_GMMparams(Kmog=10, mode='random') 


