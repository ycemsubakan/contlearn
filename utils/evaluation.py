from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visual_evaluation import plot_images

import numpy as np

import time

import os
import pdb
import itertools as it
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae_multihead(args, model, train_loader, data_loader, epoch, dr, mode, 
                 prev_model=None):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)

        if args.dataset_name == 'patch_celeba':
            data = data.reshape(-1, int(np.prod(args.input_size)))

        x = data

        # calculate loss function
        if prev_model != None:
            x_replay = prev_model.generate_x(data.size(0))
            loss_1, RE_1, KL_1, _ = model.calculate_loss(x_replay.reshape(x.size(0), -1), average=True, head=0)
            loss_2, RE_2, KL_2, _ = model.calculate_loss(x.reshape(x.size(0), -1), average=True, head=1)

            evaluate_loss += loss_1.data[0] + loss_2.data[0]
            evaluate_re += -RE_1.data[0] - RE_2.data[0]

            evaluate_kl += KL_1.data[0] + KL_2.data[0]
        else: 
            loss, RE, KL, _ = model.calculate_loss(x.reshape(x.size(0), -1), average=True, head=0)

            evaluate_loss += loss.data[0] 
            evaluate_re += -RE.data[0] 

            evaluate_kl += KL.data[0]

        # print N digits
        #if batch_idx == 1 and mode == 'validation':
        #    if epoch == 1:
        #        if not os.path.exists(dr + 'reconstruction/'):
        #            os.makedirs(dr + 'reconstruction/')
        #        # VISUALIZATION: plot real images
        #        plot_images(args, data.data.cpu().numpy()[0:9], dr + 'reconstruction/', 'real', size_x=3, size_y=3)
        #    x_mean = model.reconstruct_x(x)
        #    plot_images(args, x_mean.data.cpu().numpy()[0:9], dr + 'reconstruction/', str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)

        test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

        # grab the train data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        full_data = []
        if not args.dataset_name == 'celeba':
            for data, _ in train_loader:
                full_data.append(data)

            full_data = torch.cat(full_data, 0)
        else:
            for data, _ in it.islice(train_loader, 0, 100):
                full_data.append(data)

            full_data = torch.cat(full_data, 0)

        if args.cuda:
            test_data, test_target = test_data.cuda(), test_target.cuda()
            #if not args.dataset_name == 'celeba':
            full_data = full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # print(model.means(model.idle_input))

        # VISUALIZATION: plot real images
        #plot_images(args, test_data.data.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        #plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        if args.model_name != 'vrae':
            samples_rand = model.generate_x(25)

            #plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            #plot_images(args, pseudoinputs[0:25], dir, 'pseudoinputs', size_x=5, size_y=5)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        #if not args.dataset_name == 'celeba':
        elbo_train = model.calculate_lower_bound(full_data, MB=args.MB)
        #else:
         #   elbo_train = torch.Tensor([0])
        t_ll_e = time.time()
        print('Train lower-bound value {} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        if 1:
            log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        else:
            log_likelihood_test = 0
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return {'test_loss': evaluate_loss.item(), 
                'test_re' : evaluate_re.item(),
                'test_kl': evaluate_kl.item(), 
                'test_ll' : log_likelihood_test,
                'train_ll' : log_likelihood_train, 
                'test_elbo' : elbo_test.item(), 
                'train_elbo' : elbo_train.item()}
    else:
        return {'test_loss': evaluate_loss.item(), 
                'test_re' : evaluate_re.item(),
                'test_kl': evaluate_kl.item()} 


def evaluate_vae(args, model, train_loader, data_loader, epoch, dr, mode, 
                 prev_model=None, use_mixw_cor=False):
    
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    if args.semi_sup: evaluate_ce = 0
    
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)

        # to avoid the singleton case (otherwise it breaks the code)
        if data.size(0) == 1:
            data = torch.cat([data, data], dim=0)
        
        if args.dataset_name == 'patch_celeba':
            data = data.reshape(-1, int(np.prod(args.input_size)))

        # to avoid the singleton case (otherwise it breaks the code)
        if data.size(0) == 1:
            data = torch.cat([data, data], dim=0)
        x = data

        # calculate loss function
        if args.semi_sup:
            loss, RE, KL, CE, _ = model.calculate_loss(x.reshape(x.size(0), -1), target, average=True)
        else:
            loss, RE, KL, _ = model.calculate_loss(x.reshape(x.size(0), -1), average=True, head=0, use_mixw_cor=use_mixw_cor)

        evaluate_loss += loss.item() 
        evaluate_re += -RE.item()
        evaluate_kl += KL.item()
        if args.semi_sup: evaluate_ce += CE.item()

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)

        test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

        # grab the train data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        full_data = []
        if not args.dataset_name == 'celeba':
            for data, _ in train_loader:
                full_data.append(data)

            full_data = torch.cat(full_data, 0)
        else:
            for data, _ in it.islice(train_loader, 0, 100):
                full_data.append(data)

            full_data = torch.cat(full_data, 0)

        if args.cuda:
            test_data, test_target = test_data.cuda(), test_target.cuda()
            #if not args.dataset_name == 'celeba':
            full_data = full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        #plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        if args.model_name != 'vrae':
            samples_rand = model.generate_x(25)

            #plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        #if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            #pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            #plot_images(args, pseudoinputs[0:25], dir, 'pseudoinputs', size_x=5, size_y=5)

        # CALCULATE accuracy
        if args.semi_sup:
            t_ll_s = time.time()
            acc_test = model.calculate_accuracy(test_data, test_target, MB=args.MB)
            t_ll_e = time.time()
            print('Test accuracy value {:.2f} in time: {:.2f}s'.format(acc_test, t_ll_e - t_ll_s))
        
        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE lower-bound
        t_ll_s = time.time()
        if not args.debug: elbo_train = model.calculate_lower_bound(full_data, MB=args.MB).item()
        else: elbo_train = -1
        t_ll_e = time.time()
        print('Train lower-bound value {} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        if not args.debug: 
            log_likelihood_test = model.calculate_likelihood(test_data, 
                dir, mode='test', S=args.S, MB=args.MB).item()
        else: 
            log_likelihood_test = -1 
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB))
        #commented because it takes too much time
        log_likelihood_train = -1
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if args.semi_sup: evaluate_ce /= len(data_loader)  # ce already averages over batch size
    
    output = {'test_loss': evaluate_loss, 
                'test_re' : evaluate_re,
                'test_kl' : evaluate_kl}
    
    if args.semi_sup:
        output['test_ce'] = evaluate_ce
        
    if mode == 'test':
        output['test_ll'] = log_likelihood_test
        output['train_ll'] = log_likelihood_train
        output['test_elbo'] = elbo_test
        output['train_elbo'] = elbo_train
        if args.semi_sup:
            output['test_acc'] = acc_test
    
    return output

def evaluate_classifier(args, classifier, data_loader):

    all_lbls = []
    all_preds = []

    for batch_idx, (data, target) in enumerate(it.islice(data_loader, 0, None)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        all_preds.append(classifier.forward(data))
        all_lbls.append(target)

    all_preds_cat = torch.argmax(torch.cat(all_preds, dim=0), dim=1)
    all_lbls_cat = torch.cat(all_lbls, dim=0)

    # label at task t  =  perm[t]   
    # example : perm = [3 0 1 4 2], label_0 = 3, label_1 = 0, label_2 = 1, label_3 = 4, label_4 = 2

    # forward map: 0 -> 3, 1 -> 0, 2 -> 1, 3 -> 4, 4-> 2
    # backward map: 0 -> 1, 1->2, 2->4, 3->0, 4->3

    acc = (all_preds_cat == all_lbls_cat).float().mean()
    return acc, all_preds_cat

