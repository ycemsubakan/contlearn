import pickle
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import copy

path = 'select_files/'
files = os.listdir(path)

test_lls = []
colors = 'rbmkycgrbm'
markers = 'oxv^oxv^oxv^'

legends = ['1', '2', '3', '4', '5', '6', '7', '8']

plt.figure(figsize=[20, 10], dpi=100)
for i, fl in enumerate(files):
    results = pickle.load(open(path + fl, 'rb'))
    test_lls = [res['test_ll'] for res in results]
    test_cls = [res['class'] for res in results]

    lbl = copy.deepcopy(fl)
    lbl = lbl.replace('K500_wu100_z1_40_z2_40', ' ').replace('replay_size', ' ').replace('add_cap_0_usevampmixingw_1_separate_means_0_useclassifier_1', ' ').replace('dynamic_mnist_vae', ' ')
    plt.subplot(121)
    plt.plot(np.arange(len(test_lls)), test_lls, '-' + colors[i] + markers[i], label=lbl)

    plt.subplot(122)
    plt.plot(np.arange(len(test_cls)), test_cls, '-' + colors[i] + markers[i], label=lbl)


plt.subplot(121)
plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Negative LogLikelihood')

plt.xticks(range(10), range(10))
plt.legend()

plt.subplot(122)
plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Classification Accuracy')

plt.xticks(range(10), range(10))

plt.show()
#plt.savefig('Figure_contlearn.png')
