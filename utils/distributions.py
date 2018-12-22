from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import pdb


min_epsilon = 1e-5
max_epsilon = 1.-1e-5
#=======================================================================================================================
def log_mog_diag(x, mus, sigs, pis, average=False, dim=None):
    log_normal_all = -0.5 * ( sigs.unsqueeze(0).log() + (torch.pow(x.unsqueeze(1) - mus.unsqueeze(0), 2) / sigs.unsqueeze(0)) ).sum(-1)
    log_normal_all = log_normal_all + pis.log().unsqueeze(0)   

    mx, _ = torch.max(log_normal_all, 1) 
    
    # calculte log-sum-exp
    log_mog = mx + torch.log(torch.sum(torch.exp(log_normal_all - mx.unsqueeze(1)), 1))  
    return log_mog

def log_mog_full(x, mus, sigs, pis, icovs_ten, det_terms, average=False, dim=None):
    L = x.size(1)
    K = sigs.size(0)
   
    cov_eps = 1e-10

    dists = (x.unsqueeze(1) - mus.unsqueeze(0)).unsqueeze(2)

    xS = torch.matmul(dists, icovs_ten.unsqueeze(0))

    vals = -0.5*(xS * dists).sum(-1).squeeze()
    #det_terms = torch.Tensor([0.5*(cov_eps + torch.svd(cov.squeeze())[1]).log().sum() for cov in sigs])
    #det_terms = det_terms.cuda()
    log_normal_all = vals - det_terms.unsqueeze(0) + torch.log(pis).unsqueeze(0)

    mx, _ = torch.max(log_normal_all, 1) 
    
    # calculte log-sum-exp
    log_mog = mx + torch.log(torch.sum(torch.exp(log_normal_all - mx.unsqueeze(1)), 1))  
    return log_mog


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )

def logisticCDF(x, u, s):
    return 1. / ( 1. + torch.exp( -(x-u) / s ) )

def sigmoid(x):
    return 1. / ( 1. + torch.exp( -x ) )

def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256
