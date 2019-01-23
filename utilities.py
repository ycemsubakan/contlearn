import torch
import numpy as np
import pdb
import torch.nn.utils.rnn as rnn_utils 
import torch.utils.data as data_utils
import os
import torch.nn.init as torchinit
from torchvision import datasets, transforms
import itertools as it
import copy

def separate_datasets(loader, dataset_type, Klabels, folder):
    fts = []
    labels = []
    for i, (ft, tar) in enumerate(loader):   
        fts.append(ft)
        labels.append(tar)
    
    all_fts = torch.cat(fts, dim=0)
    all_labels = torch.cat(labels, dim=0)

    datasets = [] 
    for lb in range(Klabels):
        mask = torch.eq(all_labels, lb)
        inds = torch.nonzero(mask).squeeze()
        dt = torch.index_select(all_fts, dim=0, index=inds)
        lbls = torch.index_select(all_labels, dim=0, index=inds)

        datasets.append(data_utils.TensorDataset(dt, lbls))

    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(datasets, folder + dataset_type + '.t' ) 


def get_mnist_loaders(digits, dataset_type, arguments, path = 'mnist_files/', 
                      model=None, dg=None):

    sets = torch.load(path + dataset_type +  '.t')
    dataset = data_utils.ConcatDataset([sets[dg] for dg in digits])
    
    if model != None:
        N = len(dataset)
        x_replay = data_utils.TensorDataset(model.generate_x(dg*N, replay=True).data.cpu(),
                                            torch.zeros(dg*N).long())
        dataset = data_utils.ConcatDataset([dataset, x_replay])


    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    shuffle = True
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size,
                                   shuffle=shuffle, **kwargs)
    return loader


def filter_mnist_loader(digits, loader, arguments):
    dataset = []
    for i, (ft, tar) in enumerate(loader):   
        # digit 1
        masks = torch.stack([torch.eq(tar, dg) for dg in digits]).sum(0)
        inds = torch.nonzero(masks).squeeze()
        ft1 = torch.index_select(ft, dim=0, index=inds)
        dataset.append(ft1)

    dataset = torch.cat(dataset, dim=0)

    dataset = data_utils.TensorDataset(dataset, dataset)

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    shuffle = True
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=shuffle, **kwargs)

    return loader


def get_loaders(loader_batchsize, **kwargs):
    arguments=kwargs['arguments']
    data = arguments.data

    if data == 'mnist':
        kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0,), (1,))
                           ])),
            batch_size=loader_batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((7,), (0.3081,))
                           ])),
            batch_size=loader_batchsize, shuffle=True, **kwargs)

    return train_loader, test_loader


