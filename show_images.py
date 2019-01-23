import visdom 
import pickle
import pdb
import torchvision
import os
import torch

vis = visdom.Visdom(port=5800, server='', env='',
                    use_incoming_socket=False)
assert vis.check_connection()

save_path = '/home/user1/Dropbox/GANs/ICML2019/paper/gen_samples/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

images = pickle.load(open('azure_images/task19mnist_plus_fmnist_permutation_0db_0vae_standard_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_0_use_entrmax_0mpfmnistvae.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vae_cr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')

images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_standard_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_0mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vae_rcc' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')


images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_standard_K50_replay_size_increasereplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_0mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vae_ir' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')



images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_1mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vamp_cr_megr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')

images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_increasereplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_1mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)


opts = {'title': 'vamp_ir_megr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')


images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_increasereplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_0mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)


opts = {'title': 'vamp_ir' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')




images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_0_use_entrmax_1mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)



opts = {'title': 'vamp_cr_megr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')


images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_0_use_entrmax_0mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vamp_cr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')


images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_0mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vamp_rcc' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')


images = pickle.load(open('gensamples_files/task19mnist_plus_fmnist_permutation_0db_0vae_vampprior_short_K50_replay_size_constantreplay_add_cap_1_usemixingw_1_sep_means_0_useclass_1_semi_sup_0_Lambda_1_use_mixingw_correction_0_use_replaycostcorrection_1_use_entrmax_1mpfmnistb1.pk', 'rb'))
images = torch.from_numpy(images)
images = images.reshape(-1, 1, 28, 28)

opts = {'title': 'vamp_rcc_megr' }
vis.images(images.reshape(-1, 1, 28, 28), opts=opts, win=opts['title'])
torchvision.utils.save_image(images, save_path + opts['title'] + '.png')




