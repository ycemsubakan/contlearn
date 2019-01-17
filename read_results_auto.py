import pickle
#import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import copy

path = 'select_files_fmnist/'
#path = 'results_massimo/'
files = os.listdir(path)

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
                            models[i].append(fl)    
                    
                    # deal with non-existing combinations
                    if len(models[i]) == 0:
                        models.pop()    
                    else: 
                        i = i +  1
                        print(len(models[-1]))



#for i, fl in enumerate(files):
#    if 'db_1' not in fl:
#        filesf.append(fl)

test_lls = []
colors = 'rbmkycgrbmrbmkycgrbm'
markers = 'oxv^oxv^oxv^oxv^'

legends = ['1', '2', '3', '4', '5', '6', '7', '8']

plt.figure(figsize=[20, 10], dpi=100)
T = 10

all_ll_means = []
for i, mdl in enumerate(models): 
    test_lls = []
    test_cls = []

    NC = 0
    for fl in mdl:
        results = pickle.load(open(path + fl, 'rb'))
        try:
            temp1 = [res['ent_emp'] for res in results]
        except:
            pass

        try:
            temp2 = [res['class'] for res in results]
        except:
            temp2 = [res['test_acc'] for res in results]

        if (len(temp1) == T) and ('permutation_2' in fl) : #and ('permutation_4' not in fl) and ('permutation_0' not in fl):
            test_lls.append(temp1)
            test_cls.append(temp2)
            NC = NC + 1
    test_lls = np.array(test_lls)
    test_cls = np.array(test_cls)

    all_ll_means.append(test_lls.mean(0))
    
    
    lbl = copy.deepcopy(fl)
    lbl = lbl.replace('_wu100_z1_40_z2_40', ' ').replace('replay_size', ' ').replace('add_cap_0_usevampmixingw_1_separate_means_0_useclassifier_1', ' ').replace('dynamic_mnist_permutation', ' ')
    #plt.subplot(121)
    if test_lls.shape[0] > 0:
        plt.plot(np.arange(test_lls.shape[1]), test_lls.mean(0), '-' + colors[i] + markers[i], label=lbl + 'NC' + str(NC))
        #parts = plt.violinplot(positions=np.arange(T) + (0.1*i) , dataset=test_lls, widths=0.1)

    #for pc in parts['bodies']:
    #    pc.set_facecolor(colors[i])
    #    pc.set_edgecolor(colors[i])
    #    pc.set_alpha(1)

    #plt.subplot(122)
    #plt.plot(np.arange(test_lls.shape[1]), test_cls.mean(0), '-' + colors[i] + markers[i], label=lbl + 'NC' + str(NC))
    M = 20
    #plt.plot(np.arange(len(test_lls)), test_lls, '-' + colors[i] + markers[i], label=lbl)


#plt.subplot(121)
plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Negative LogLikelihood')

#plt.xticks(range(20), range(20))
plt.legend()

#plt.subplot(122)
plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Classification Accuracy')

#plt.xticks(range(20), range(20))

    #plt.savefig('Figure_contlearn.png')
plt.show()

plt.figure()
##second figure
#all_ll_means_max = np.array(all_ll_means).max(0)
#
#
#for i, mdl in enumerate(models): 
#    test_lls = []
#    test_cls = []
#
#    NC = 0
#    for fl in mdl:
#        results = pickle.load(open(path + fl, 'rb'))
#        temp1 = [res['test_ll'] for res in results]
#        if (len(temp1) == T): #and ('permutation_1' in fl): #and ('permutation_4' not in fl) and ('permutation_0' not in fl):
#            test_lls.append(temp1)
#            NC = NC + 1
#    test_lls = np.array(test_lls)
#
#    lbl = copy.deepcopy(fl)
#    lbl = lbl.replace('_wu100_z1_40_z2_40', ' ').replace('replay_size', ' ').replace('add_cap_0_usevampmixingw_1_separate_means_0_useclassifier_1', ' ').replace('dynamic_mnist_permutation', ' ')
#    if len(test_lls) > 0:
#        plt.plot(np.arange(T), test_lls.mean(0)/all_ll_means_max, '-' + colors[i] + markers[i], label=lbl + 'NC' + str(NC))
#
#plt.legend()
#plt.show()
# 
