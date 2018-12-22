import pickle
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

results1 = pickle.load(open('results_files/dynamic_mnist_vae_standard_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_0.pk', 'rb'))
#results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increaseprototype_add_cap_1.pk', 'rb'))

#results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1.pk', 'rb'))
results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increaseprototype_add_cap_1.pk', 'rb'))


test_lls1 = [res['test_ll'] for res in results1]
test_lls2 = [res['test_ll'] for res in results2]

plt.plot(np.arange(len(test_lls1)), test_lls1, '-rx', label='vae')
plt.plot(np.arange(len(test_lls2)), test_lls2, '-bx', label='VampP with replay')

plt.xticks(range(10), range(10))
plt.legend()

plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Negative LogLikelihood')
plt.show()
