import visdom 
import pickle
import pdb

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2',
                    use_incoming_socket=False)
assert vis.check_connection()

images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_standard_K50_replay_size_increasereplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_0mpfmnistb1.pk', 'rb'))

opts = {'title': 'vae_increase' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])

images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_1mpfmnistb1.pk', 'rb'))

opts = {'title': 'vamp_constantentr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
