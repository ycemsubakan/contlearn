from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
import torch.nn.functional as F 
import pdb
import itertools as it
import torchvision
import visdom
import time
import math
import torch.nn as nn
import copy

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev',
                    use_incoming_socket=False)
assert vis.check_connection()





def experiment_vae_multihead(arguments, train_loader, val_loader, test_loader, 
                             model, optimizer, dr, model_name='vae', prev_model=None):
    from utils.evaluation import evaluate_vae as evaluate
    from utils.helpers import print_and_log_scalar

    # Save the arguments to keep track of the used config
    torch.save(arguments, dr + '.config')

    # best_model = model
    best_loss = 100000
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    time_history = []

    for epoch in range(1, arguments.epochs + 1):
        time_start = time.time()
        if prev_model == None:
            model, train_loss_epoch, train_re_epoch, train_kl_epoch = train_vae(epoch, arguments, train_loader, model, optimizer)
            samples = model.generate_x(100)
            vis.images(samples.reshape(-1, arguments.input_size[0], arguments.input_size[1],
                                       arguments.input_size[2]), win='samples_x')
        else:
            model, train_loss_epoch, train_re_epoch, train_kl_epoch = train_vae_multihead(epoch, arguments, train_loader, model, optimizer, prev_model=prev_model)
            samples1 = model.generate_x(100, head=0)
            samples2 = model.generate_x(100, head=1)
            vis.images(samples1.reshape(-1, 1, 28, 28), win='xgen1')
            vis.images(samples2.reshape(-1, 3, 32, 32), win='xgen2')

        val_results = evaluate(arguments, model, train_loader, val_loader, epoch, dr, mode='validation', prev_model=prev_model)

        val_loss_epoch, val_re_epoch, val_kl_epoch = val_results['test_loss'], val_results['test_re'], val_results['test_kl']      
        
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
           val_kl_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {}s\n'
              '* Train loss: {}   (RE: {}, KL: {})\n'
              'o Val.  loss: {}   (RE: {}, KL: {})\n'
              '--> Early stopping: {}/{} (BEST: {})\n'.format(
            epoch, arguments.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch,
            e, arguments.early_stopping_epochs, best_loss
        ))
        
        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('model saved')
            torch.save(model.state_dict(), dr + '.model')
        else:
            e += 1
            if epoch < arguments.warmup:
                e = 0
            if e > arguments.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break

    # FINAL EVALUATION
    #model.load_state_dict(torch.load(dr + '.model'))
    #res = evaluate(arguments, model, train_loader, test_loader, 9999, dr, mode='test')

    #test_loss, test_re, test_kl, test_log_likelihood, train_log_likelihood, test_elbo, train_elbo = res['test_loss'], res['test_re'], res['test_kl'], res['test_ll'], res['train_ll'], res['test_elbo'], res['train_elbo']

    #print('FINAL EVALUATION ON TEST SET\n'
    #      'LogL (TEST): {:.2f}\n'
    #      'LogL (TRAIN): {:.2f}\n'
    #      'ELBO (TEST): {:.2f}\n'
    #      'ELBO (TRAIN): {:.2f}\n'
    #      'Loss: {:.2f}\n'
    #      'RE: {:.2f}\n'
    #      'KL: {:.2f}'.format(
    #    test_log_likelihood,
    #    train_log_likelihood,
    #    test_elbo,
    #    train_elbo,
    #    test_loss,
    #    test_re,
    #    test_kl
    #))

def experiment_vae(arguments, train_loader, val_loader, test_loader, 
                   model, optimizer, dr, model_name='vae', prev_model=None, 
                   classifier=None, prev_classifier=None, optimizer_cls=None, dg=0):
    from utils.evaluation import evaluate_vae as evaluate
    from utils.helpers import print_and_log_scalar

    # Save the arguments to keep track of the used config
    torch.save(arguments, dr + '.config')

    # best_model = model
    best_loss = 100000
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []
    if arguments.semi_sup: train_ce_history = []
        
    val_loss_history = []
    val_re_history = []
    val_kl_history = []
    if arguments.semi_sup: val_ce_history = []

    time_history = []

    for epoch in range(1, arguments.epochs + 1):
        time_start = time.time()
        #if prev_model == None:
        model, train_results = train_vae(epoch, 
                        arguments, train_loader, model, optimizer, classifier=classifier, 
                        prev_classifier=prev_classifier, prev_model=prev_model, 
                        optimizer_cls=optimizer_cls, dg=dg)

        # merge the means for evaluation and sampling
        if (dg > 0) and arguments.separate_means: 
            model.merge_latent()
        samples = model.generate_x(100)
        if arguments.use_visdom:
            vis.images(samples.reshape(-1, arguments.input_size[0], arguments.input_size[1],
                                       arguments.input_size[2]), win='samples_x')

        
        val_results = evaluate(arguments, model, train_loader, val_loader, epoch, dr, mode='validation', prev_model=prev_model)
        
        # separate the means for the rest of the training
        if (dg > 0) and arguments.separate_means:
            model.separate_latent()

        if arguments.prior == 'vampprior_short':
            means = model.reconstruct_means(head=0)
            if arguments.use_visdom:
                vis.images(means.reshape(-1, arguments.input_size[0], arguments.input_size[1], arguments.input_size[2]), win='means')

            #if dg > 0:
            #    means = model.reconstruct_means(head=1)
            #    vis.images(means.reshape(-1, arguments.input_size[0], arguments.input_size[1],
            #                             arguments.input_size[2]), win='means2')

        
        time_end = time.time()
        time_elapsed = time_end - time_start

        # round the numbers:
        train_results = {k: np.round(v,2) for k, v in train_results.items()}
        val_results = {k: np.round(v,2) for k, v in val_results.items()}

        # appending history
        train_loss_history.append(train_results['train_loss'])
        train_re_history.append(train_results['train_re'])
        train_kl_history.append(train_results['train_kl'])
        if arguments.semi_sup: train_ce_history.append(train_results['train_ce'])
        
        val_loss_history.append(val_results['test_loss'])
        val_re_history.append(val_results['test_re'])
        val_kl_history.append(val_results['test_kl'])
        if arguments.semi_sup: val_ce_history.append(val_results['test_ce'])

        # printing results
        if arguments.semi_sup:
            print('Epoch: {}/{}, Time elapsed: {}s\n'
              '* Train loss: {}   (RE: {}, KL: {}, CE: {})\n'
              'o Val.  loss: {}   (RE: {}, KL: {}, CE: {})\n'
              '--> Early stopping: {}/{} (BEST: {})\n'.format(
            epoch, arguments.epochs, time_elapsed,
            train_results['train_loss'], train_results['train_re'], 
            train_results['train_kl'], train_results['train_ce'],
            val_results['test_loss'], val_results['test_re'], 
            val_results['test_kl'], val_results['test_ce'],
            e, arguments.early_stopping_epochs, best_loss
        ))
        else:    
            print('Epoch: {}/{}, Time elapsed: {}s\n'
              '* Train loss: {}   (RE: {}, KL: {})\n'
              'o Val.  loss: {}   (RE: {}, KL: {})\n'
              '--> Early stopping: {}/{} (BEST: {})\n'.format(
            epoch, arguments.epochs, time_elapsed,
            train_results['train_loss'], train_results['train_re'], train_results['train_kl'],
            val_results['test_loss'], val_results['test_re'], val_results['test_kl'],
            e, arguments.early_stopping_epochs, best_loss
        ))

        
        # early-stopping
        if val_results['test_loss'] < best_loss:
            e = 0
            best_loss = val_results['test_loss']
            # best_model = model
            print('model saved')
            torch.save(model.state_dict(), dr + '.model')
        else:
            e += 1
            if epoch < arguments.warmup:
                e = 0
            if e > arguments.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_results['test_loss']):
            break


def train_classifier(args, train_loader,  
                     classifier=None, prev_classifier=None, prev_model=None, 
                     optimizer_cls=None, dg=0):

    # start training
    EP = 25 
    for ep in range(EP):
        for batch_idx, (data, target) in enumerate(it.islice(train_loader, 0, None)):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # to avoid the singleton case
            if data.size(0) == 1:
                data = torch.cat([data, data], dim=0)

            optimizer_cls.zero_grad()
            yhat = classifier.forward(data)
            cent = nn.CrossEntropyLoss()

            targets = torch.empty(yhat.size(0), dtype=torch.long).fill_(dg).cuda()
            loss_cls = cent(yhat, targets)

            if dg > 0: 
                if args.replay_size == 'increase':
                    cst = copy.deepcopy(dg) 
                else:
                    cst = 1
                x_gen = prev_model.generate_x(cst*args.batch_size, replay=True)
                prev_targets = torch.argmax(prev_classifier.forward(x_gen), dim=1) 
                yhat_prev = classifier.forward(x_gen)
                loss_cls_prev = cent(yhat_prev, prev_targets)

                if args.use_replaycostcorrection and (args.replay_size == 'constant'):
                    loss_cls = loss_cls + dg*loss_cls_prev
                else:
                    loss_cls = loss_cls + loss_cls_prev

            # backward pass
            loss_cls.backward()
            # optimization
            optimizer_cls.step()
        
        print('EP {} batch {}, loss {}'.format(ep, batch_idx, loss_cls))




def train_vae(epoch, args, train_loader, model, 
              optimizer, classifier=None, prev_classifier=None, prev_model=None, 
              optimizer_cls=None, dg=0):

    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    if args.semi_sup: train_ce = 0

    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(it.islice(train_loader, 0, None)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # to avoid the singleton case
        if data.size(0) == 1:
            data = torch.cat([data, data], dim=0)

        if (prev_model != None) and (args.replay_type == 'replay'):
            if args.replay_size == 'constant':
                cst = 1 
            else:
                cst = copy.deepcopy(dg)
            # generate replay data:
            if args.semi_sup:
                x_replay, y_replay = prev_model.generate_x((cst)*data.size(0), replay=True)
            else:    
                x_replay = prev_model.generate_x((cst)*data.size(0), replay=True)
            if epoch % args.num_classes == 0:
                print('replay size {}'.format(x_replay.size(0)))
            
            if args.replay_size == 'increase': 
                data = torch.cat([data, x_replay.data], dim=0)
                if args.semi_sup:
                    target = torch.cat([target, y_replay], dim=0)

        
        #if len(data.shape) > 2:
        #    data = data.reshape(data.size(0), -1)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        
        # loss evaluation (forward pass)
        if args.separate_means and (dg > 0) and (args.replay_size == 'constant'):
           
            if args.semi_sup:
                raise Exception('not implmented yet!')
 
            loss1, RE1, KL1, _ = model.calculate_loss(x_replay, beta=beta, average=True, head=0)
            loss2, RE2, KL2, _ = model.calculate_loss(x, beta=beta, average=True, head=1)

            loss = loss1 + loss2
            RE = RE1 + RE2
            KL = KL1 + KL2 
        
        elif not args.separate_means  and (dg > 0) and (args.replay_size == 'constant'):
            
            if args.semi_sup:
                loss1, RE1, KL1, CE1, _ = model.calculate_loss(x_replay, y_replay, beta=beta, average=True, head=0)
                loss2, RE2, KL2, CE2, _ = model.calculate_loss(x, target, beta=beta, average=True, head=0)

                if args.use_replaycostcorrection:
                    loss = (dg)*loss1 + loss2
                else:
                    loss = loss1 + loss2

                RE = RE1 + RE2
                KL = KL1 + KL2 
                CE = CE1 + CE2
            
            else:
                loss1, RE1, KL1, _ = model.calculate_loss(x_replay, beta=beta, average=True, head=0)
                loss2, RE2, KL2, _ = model.calculate_loss(x, beta=beta, average=True, head=0)

                if args.use_replaycostcorrection:
                    loss = (dg)*loss1 + loss2
                else:
                    loss = loss1 + loss2

                RE = RE1 + RE2
                KL = KL1 + KL2 
        
        elif ( (args.separate_means == False) and (dg == 0) ) or (args.replay_size == 'increase'):
            
            if args.semi_sup:
                loss, RE, KL, CE, _ = model.calculate_loss(x, target, beta=beta, average=True)
            else:    
                loss, RE, KL, _ = model.calculate_loss(x, beta=beta, average=True)
        
        if (args.replay_type == 'prototype') and (dg > 0):
            
            if args.semi_sup:
                raise Exception('not implemented yet!')
            
            loss_p, _, _, _ = model.calculate_loss(model.prototypes, beta=beta, average=True)
            loss = loss + loss_p
        
        if batch_idx % 300 == 0:
            print('batch {}, loss {}'.format(batch_idx, loss))

        # backward pass
        loss.backward()
        # optimization
        optimizer.step()
        
        if model.args.prior == 'GMM':
            model.pis.data = model.pis.data.abs() / model.pis.data.abs().sum()
            if model.GMM.covariance_type == 'diag':
                model.sigs.data = F.relu(model.sigs.data)

        train_loss += loss.item()
        train_re += -RE.item()
        train_kl += KL.item()
        if args.semi_sup: train_ce += CE.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size
    
    train_results = {}
    train_results['train_loss'] = train_loss
    train_results['train_re']   = train_re
    train_results['train_kl']   = train_kl
    train_results['train_kl']   = train_kl
    
    if args.semi_sup:
        train_ce /= len(train_loader)
        train_results['train_ce'] = train_ce
         
    return model, train_results

def train_vae_multihead(epoch, args, train_loader, model, optimizer, prev_model):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(it.islice(train_loader, 0, None)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        x_replay = prev_model.generate_x(data.size(0))

        #if len(data.shape) > 2:
        #    data = data.reshape(data.size(0), -1)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)

        loss_2, RE_2, KL_2, xhat = model.calculate_loss(x.reshape(x.size(0), -1), beta, average=True, head=1)
        loss_1, RE_1, KL_1, _ = model.calculate_loss(x_replay.reshape(x.size(0), -1), beta, average=True, head=0)
        loss = loss_1 + loss_2

        
        if batch_idx % 300 == 0:
            print('batch {}, loss {}'.format(batch_idx, loss))
           
            opts = {'title' : 'xhat'}
            vis.images(xhat.reshape(-1, 3, 32, 32), win='xhat', opts=opts)
            
            opts = {'title' : 'x'}
            vis.images(x.reshape(-1, 3, 32, 32), win='x', opts=opts)

            if args.dataset_name == 'celeba':
                gen_data = model.generate_x(64).reshape(-1, 3, 64, 64)
                torchvision.utils.save_image(gen_data, 
                                         'temp/{}_samples_{}.png'.format(args.prior, args.dataset_name))

        if epoch % 10 == 0:
            if args.dataset_name == 'patch_celeba':
                gen_data = model.generate_x(64).reshape(-1, 3, 16, 16)
                torchvision.utils.save_image(gen_data, 
                                         'temp/{}_samples_{}.png'.format(args.prior, args.dataset_name))


        # backward pass
        loss.backward()
        # optimization
        optimizer.step()
        
        if model.args.prior == 'GMM':
            model.pis.data = model.pis.data.abs() / model.pis.data.abs().sum()
            if model.GMM.covariance_type == 'diag':
                model.sigs.data = F.relu(model.sigs.data)

        train_loss += loss.item()
        train_re += -RE_1.item() - RE_2.item()
        train_kl += KL_1.item() + KL_2.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl



def train_rnn_on_zs(args, vae, train_loader, rn, optimizer, phase='two_step'):

    Lout = int(np.prod(args.input_size))
    for ep in range(args.epochs):
        for i, (dt, _) in enumerate(train_loader):

            #vis.images(dt[0].reshape(-1, 1, 28, 28), win='ex1')
            dt = dt.cuda()

            optimizer.zero_grad()
            mu, logvar = vae.q_z(dt.reshape(-1, Lout))
            h_target = torch.randn(mu.size()).cuda()*(0.5*logvar).exp() + mu
            h_target = h_target.reshape(dt.size(0), dt.size(1), mu.size(-1))

            hhat, _ = rn.forward(h_target[:, :-1, :])

            #err = (hhat - h_target[:, 1:, :].reshape(-1, hhat.size(-1))).pow(2).mean()

            x_mean, _ = vae.p_x(hhat)
            x_mean_target, _ = vae.p_x(h_target.reshape(-1, hhat.size(-1)))
            
            dt_targets = dt[:, 1:, :, :, :].reshape(x_mean.size(0), -1) 
            err = (dt_targets - x_mean).abs().mean()

            err.backward()
            optimizer.step()
            
            print('error {} epoch {} phase {}'.format(err.item(), ep, phase))
        opts = {}
        opts['title'] = 'xhat'
        sz = args.input_size

        if args.dataset_name == 'patch_celeba':
            N = 16
            nrow = 4
        else:
            N = 100
            nrow = 8

        vis.images(x_mean[:N].reshape(-1, sz[0], sz[1], sz[2]).cpu().data, win='xhat', opts=opts, nrow=nrow)
        
        opts['title'] = 'real targets'
        vis.images(dt[0][:N].reshape(-1, sz[0], sz[1], sz[2]).cpu() , win='real_targets', opts=opts, nrow=nrow)

        opts['title'] = 'xhat targets'
        vis.images(x_mean_target[:N].reshape(-1, sz[0], sz[1], sz[2]).cpu().data, win='xhat_target', opts=opts, nrow=nrow)

        opts['title'] = 'hhat'
        vis.heatmap(hhat[:15], win='hhat', opts=opts)

        opts['title'] = 'htargets'
        vis.heatmap(h_target[0][1:16], win='htarget', opts=opts)



