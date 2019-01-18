import pickle
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

results1 = pickle.load(open('results_files/mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_increasereplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_0_use_entrmax_1debugging.pk', 'rb'))

pdb.set_trace()

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
