import pickle
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

results1 = pickle.load(open('results_files/dynamic_mnist_vae_standard_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_0.pk', 'rb'))
#results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increaseprototype_add_cap_1.pk', 'rb'))

#results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1.pk', 'rb'))
results2 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1restart_us.pk', 'rb'))
results3 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_joint_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1_usevampmixingw_1_restart_us_sbatch.pk', 'rb'))
results4 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_short_K500_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1_usevampmixingw_1_separate_means_0_useclassifier_1_.pk', 'rb'))
#results5 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_short_K500_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_0_usevampmixingw_1_separate_means_0_useclassifier_1_.pk', 'rb'))
results5 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_short_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1_usevampmixingw_1_separate_means_0_useclassifier_1_add_cap.pk', 'rb'))
results6 = pickle.load(open('results_files/dynamic_mnist_vae_vampprior_short_K50_wu100_z1_40_z2_40replay_size_increasereplay_add_cap_1_usevampmixingw_1_separate_means_0_useclassifier_1_use_mixingw_correction_1add_cap.pk', 'rb'))

test_lls1 = [res['test_ll'] for res in results1]
test_lls2 = [res['test_ll'] for res in results2]
test_lls3 = [res['test_ll'] for res in results3]
test_lls4 = [res['test_ll'] for res in results4]
test_lls5 = [res['test_ll'] for res in results5]
test_lls6 = [res['test_ll'] for res in results6]


test_cls4 = [100*res['class'] for res in results4]
test_cls5 = [100*res['class'] for res in results5]
test_cls6 = [100*res['class'] for res in results6]


plt.plot(np.arange(len(test_lls1)), test_lls1, '-rx', label='vae')
plt.plot(np.arange(len(test_lls2)), test_lls2, '-bx', label='VampP with replay uniform sampl.')
plt.plot(np.arange(len(test_lls3)), test_lls3, '-gx', label='VampP with replay weighted sampl.')
plt.plot(np.arange(len(test_lls4)), test_lls4, '-mx', label='VampP weighted, add cap, const. replay')
plt.plot(np.arange(len(test_lls5)), test_lls5, '-yx', label='VampP. weighted sampl. same cap, inc. replay')
plt.plot(np.arange(len(test_lls6)), test_lls6, '-kx', label='VampP. weighted sampl. same cap, inc. replay, balancing')


plt.plot(np.arange(len(test_cls4)), test_cls4, '-mo', label='vamp class.')
plt.plot(np.arange(len(test_cls5)), test_cls5, '-yo', label='vamp class.')
plt.plot(np.arange(len(test_cls6)), test_cls6, '-ko', label='vamp class. balancing')


plt.xticks(range(10), range(10))
plt.legend()

plt.xlabel('task id (mnist digit)')
plt.ylabel('Test Average Negative LogLikelihood')
plt.show()
