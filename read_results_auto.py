import pickle
#import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import copy
import sys 

from itertools import cycle
from copy import deepcopy

import seaborn as sns
sns.set(font_scale=2.0)  # Make sure everything is readable.
sns.set_style("whitegrid")


def get_label(method, model, Fix_complexity):

    if model == "WGAN_GP":
        model="WGAN-GP"

    if method == "Marginal_Replay":
        label = "Marginal_Replay_" + model
    elif method == "Conditional_Replay":
        label = "Conditional_Replay_" + model
    elif method == "ewc":
        label = "Ewc"
    elif method == "baseline":
        label = "Finetuning"

    if Fix_complexity:
        label = "Unbalanced_"+ label
    else:
        label = "Balanced_"+ label

    return label

def add_ticks(dataset):
    if dataset == 'mnist_plus_fmnist':
        plt.xticks(range(20), range(20))
    else:
        plt.xticks(range(10), range(10))

def compile_results(models, path, T=10):
    all_ll_means = []
    NCs = []

    all_lls = []
    all_cls = []
    all_ents = []
    for i, mdl in enumerate(models): 
        test_lls = []
        test_cls = []
        ents = []

        NC = 0
        for fl in mdl:
            results = pickle.load(open(path + fl, 'rb'))
            try:
                temp1 = [res['test_ll'] for res in results]
            except:
                pass

            try:
                temp2 = [100*res['class'] for res in results]
            except:
                temp2 = [100*res['test_acc'] for res in results]

            temp3 = [res['ent_emp'] for res in results]

            if (len(temp1) == T):  #(('permutation_0' in fl) or ('permutation_3' in fl)):
                test_lls.append(temp1)
                test_cls.append(temp2)
                ents.append(temp3)
                NC = NC + 1

        test_lls = -np.array(test_lls)
        test_cls = np.array(test_cls)
        ents = np.array(ents)

        all_ll_means.append( (test_lls.mean(0), fl)  )

        all_lls.append( (test_lls, fl) ) 
        all_cls.append( (test_cls, fl) ) 
        all_ents.append( (ents, fl) )

        NCs.append(NC)

    return all_lls, all_cls, all_ents, all_ll_means, NCs

def plotting(results, NCs=None, mode='ML', group='all', plot_std=True):
    if NCs == None:
        NCs = [1]*len(results)

    for i, (res, fl) in enumerate(results):

        linear = 0

        if mode == 'ML':
            lbl = ''
            if 'standard' in fl: lbl += 'VAE'
            elif 'vampprior' in fl: lbl += 'VampPrior'

            if 'replay_size_increase' in fl:
                lbl += ' + Increasing'
                linear = 1

            if 'costcorrection_1' in fl:
                if not linear:
                    lbl += ' + Cost_Correction'
                linear = 1

            if 'entrmax_1' in fl:
                if linear:
                    continue
                lbl += ' + MaxEnt'

            if 'use_mixingw_correction_1' in fl:
                if 'semi_sup_1' in fl:
                    lbl += ' + Semi_Supervised_Rebalancing'
                if 'useclassifier_1' in fl:
                    lbl += ' + SepClassifierRebalancing'

        elif mode=='GAN':
            if 'unbalanced' in fl:
                if 'CGAN' in fl:
                    lbl = 'CGAN'
                else:
                    lbl = 'GAN'
            else:
                linear = 1
                if 'CGAN' in fl:
                    lbl = 'CGAN + Increasing'
                else:
                    lbl = 'GAN + Increasing'

        if group=='constant' and linear:
            continue
        if group=='linear' and not linear:
            continue

        if plot_std:
            #pdb.set_trace()
            #best_result_mean = np.mean(best_result, axis=0)
            std = np.std(res, axis=0)
            std *= 0.5
            #plt.plot(range(num_task), best_result_mean, label=label, linestyle=next(style_c))
            plt.plot(np.arange(res.shape[1]), res.mean(0), '-' + colors[i] + markers[i], label=lbl + str(NCs[i]))
            plt.fill_between(np.arange(res.shape[1]), res.mean(0) - std,
                                         res.mean(0) + std, color=colors[i], alpha=0.4)
        else:
            plt.plot(np.arange(res.shape[1]), res.mean(0), '-' + colors[i] + markers[i], label=lbl + str(NCs[i]))


def plot_cumulative(Fig_dir, Dataset, list_filename, list_task, list_seed, list_complexity):
    style_c = cycle(['-', '--', ':', '-.'])
    #ax = plt.subplot(111)

    nb_epoch = 25
    nb_tasks = 20

    all_cls = []
    for Task in list_task:
        for Fix_complexity in list_complexity:
            style_c = cycle(['-', '--', ':', '-.'])
            for value_all_seed, dataset, task, algo, model, fix_complexity in list_filename:

                # when all algo are in a figure we save it to start a new figure
                if Task in task and dataset == Dataset and Fix_complexity==fix_complexity:

                    label = get_label(algo, model, Fix_complexity)

                    nb_tasks = value_all_seed.shape[1]
                    nb_epoch = int(value_all_seed.shape[2] / nb_tasks)

                    value_mean_past_task = deepcopy(value_all_seed)

                    print("value_mean_past_task.sh")
                    print(value_mean_past_task.shape)

                    # first sum everything
                    for i in range(1,nb_tasks):
                        # format class by class (each number represente a task of 25 epoch
                        # task 0 : 0 1 2 3 4 5 6 7 8 9
                        # task 1 : 1 2 3 4 5 6 7 8 9 -1
                        # task 2 : 2 3 4 5 6 7 8 9 -1 -1
                        # ....
                        value_mean_past_task[:, 0, nb_epoch*i:] += value_all_seed[:, i, 0:nb_epoch*nb_tasks-nb_epoch*i]

                    # do the mean
                    for i in range(1,nb_tasks):
                        value_mean_past_task[:, 0, nb_epoch*i:nb_epoch*(i+1)] /= i+1
                    # for i in range(1,10):
                    #     print("!!!!!!!!!!!!!!!!!!!!!!!!")
                    #     print(value_all_seed[:, :i+1, 25*i:25*(i+1)].mean(1).shape)
                    #     print(value_all_seed[:, :i+1, 25*i:25*(i+1)])
                    #     print(value_all_seed[:, :i+1, 25*i:25*(i+1)].shape)
                    #     print(value_all_seed[:, :i+1, :].shape)
                    #     values = deepcopy(value_all_seed[:, :i+1, 25*i:25*(i+1)].mean(1)) # mean over past tasks)
                    #     value_mean_past_task[:, 0, 25*i:25*(i+1)] = values
                    #
                    #     print(value_mean_past_task[0, 0, 200:225])

                    value_cumulative_task = value_mean_past_task[:, 0, :]

                    value_cumulative_task=value_cumulative_task.reshape(len(list_seed), -1)

                    ssamples = np.arange(24, value_cumulative_task.shape[1], 25)
                    ssamples_values = value_cumulative_task[:, ssamples]
                    
                    if fix_complexity:
                        balance = 'unbalanced'
                    else:
                        balance = 'balanced'
                    lb = algo + model + balance
                    all_cls.append((ssamples_values, lb)) 
                    print(lb, ssamples_values.mean(0))

    return all_cls
                 
        
for dataset in ['mnist', 'fmnist','mnist_plus_fmnist']:

    if dataset == 'mnist':
        path = 'select_files_mnist/'
        T = 10
        perms = list(range(10))
        title = 'MNIST'
    elif dataset == 'fmnist':
        path = 'select_files_fmnist/'
        T = 10
        perms = list(range(10))
        title = 'Fashion MNIST'
    elif dataset == 'mnist_plus_fmnist':
        path = 'select_files_mpfmnist/'
        T = 20
        perms = list(range(5))
        title = 'MNIST + FashionMNIST'

    files = os.listdir(path)

    ganresultspath = "everything.pk"
    writepath = 'figs/'

    filesf = []
    priors = ['vampprior_short', 'standard']
    replays = ['replay_size_increase', 'replay_size_constant']
    mixingw_cor = ['use_mixingw_correction_0', 'use_mixingw_correction_1']
    cost_cor = ['costcorrection_0', 'costcorrection_1']
    maxent = ['use_entrmax_1', 'use_entrmax_0']

    models = [] 

    i = 0
    for pr in priors:
        for rp in replays:
            for mc in mixingw_cor:
                for cc in cost_cor:
                    for me in maxent:
                        models.append([])
                        for fl in files:
                            if (pr in fl) and (rp in fl) and (mc in fl) and (cc in fl) and (me in fl):

                                permconds = [(('permutation_' + str(pr))in fl) for pr in perms]
                                if np.sum(permconds):
                                    models[i].append(fl)    
                        
                        # deal with non-existing combinations
                        if len(models[i]) == 0:
                            models.pop()    
                        else: 
                            i = i +  1
                            print(len(models[-1]))

    test_lls = []
    colors = 'rbmkycgrbmrbmkycgrbm'
    markers = 'oxv^oxv^oxv^oxv^'
    figsize=[8, 8]
    dpi = 96

    legends = ['1', '2', '3', '4', '5', '6', '7', '8']

    test_lls, test_cls, ents, test_means, NCs = compile_results(models, path=path, T=T)
    print('completion status ', NCs)


    #for group in ['all', 'constant', 'linear']:
    for group in ['constant', 'linear']:

        #### VAE

        colors = 'rbmkycgrbmrbmkycgrbm'
        markers = 'oxv^oxv^oxv^oxv^'
        
        title_ = title
        if group == 'constant':
            title_ += ' -- O(1) solutions'
        if group == 'linear':
            title_ += ' -- O(t) solutions'
        
        # Plot likelihoods
        plt.figure(figsize=figsize, dpi=dpi)
        plotting(test_lls, NCs, group=group)
        plt.xlabel('Task id')
        plt.ylabel('Test Average LogLikelihood')
        plt.legend()
        add_ticks(dataset)
        plt.title(title_)
        plt.savefig(writepath + 'likelihood_'+group+ '_' + dataset + '.eps', format='eps')

        # Plot entropies
        plt.figure(figsize=figsize, dpi=dpi)
        plotting(ents, NCs, group=group)
        plt.xlabel('Task id')
        plt.ylabel('Class Distribution Entropy')
        plt.legend()
        add_ticks(dataset)
        plt.title(title_)
        plt.savefig(writepath + 'entropies_'+group+ '_' + dataset + '.eps', format='eps')
       
        # Plot accuracy
        plt.figure(figsize=figsize, dpi=dpi)
        plotting(test_cls, NCs, group=group)
        plt.xlabel('Task id')
        plt.ylabel('Test Classification Accuracy')
        plt.legend()
        add_ticks(dataset)
        plt.title(title_)

        #### GAN images
        colors = 'rbmkycgrbmrbmkycgrbm'
        markers = '>s<>s<>s<>s<'


        Fig_dir = os.path.join('.')
        if dataset == 'mnist':
            list_dataset = ['mnist']
        elif dataset == 'mnist_plus_fmnist':
            list_dataset = ['mnishion']
        elif dataset == 'fmnist':
            list_dataset = ['fashion']

        list_complexity = [True, False] # True means Unbalanced, False means Balanced
        list_seed_mnist = ['0','1','2','3','4','5','6','7','8','9']
        list_seed_mnishion = ['0','1','2','3','4']
        list_task = ['disjoint']
        data = pickle.load(open(ganresultspath, "rb"))

        for dataset in list_dataset:
            if dataset == "mnist" or dataset=="fashion":
                list_seed = list_seed_mnist
            elif dataset == "mnishion":
                list_seed = list_seed_mnishion
            else:
                ValueError("Dataset not implemented")
        test_cls_gan = plot_cumulative(Fig_dir, dataset, data, list_task, list_seed, list_complexity)

        plotting(test_cls_gan, mode='GAN', group=group)
        plt.xlabel('Task id')
        plt.ylabel('Test Classification Accuracy')
        plt.legend()
        add_ticks(dataset)
        plt.savefig(writepath + 'classification_' + group + '_' + dataset + '.eps', format='eps')





plt.show()

