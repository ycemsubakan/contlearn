from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np

from scipy.io import loadmat
import os
import torchvision
import pickle
import itertools as it
import pdb
import visdom


def load_static_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

def load_celeba(args, **kwargs):
    args.input_size = [3, 64, 64]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    from torchvision import transforms, datasets

    data_transform = transforms.Compose([
                    transforms.Scale(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1])
                    ])

    path_to_folder = os.path.expanduser('~') + '/crop_celeba_train'
    dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                               shuffle=False, num_workers=5, 
                                               drop_last=True, pin_memory=True)

    #data = iter(train_loader).next()[0]
    #pdb.set_trace()

    path_to_folder = os.path.expanduser('~') + '/crop_celeba_test'
    dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                              shuffle=True, num_workers=5, 
                                              drop_last=True, pin_memory=True)

    path_to_folder = os.path.expanduser('~') + '/crop_celeba_valid'
    dataset = datasets.ImageFolder(root=path_to_folder,transform=data_transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=5, 
                                             drop_last=True, pin_memory=True)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        # preparing data
        all_train = []
        for i, (dt, _) in enumerate(it.islice(train_loader, 0, 10)):
            all_train.append(dt.reshape(dt.size(0),-1))
        x_train = (torch.cat(all_train, 0)).numpy()

        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

def load_patch_celeba(args, **kwargs):
    train_loader, val_loader, test_loader, args = load_celeba(args)
  
    train_loader = create_patches(args, train_loader) 
    val_loader = create_patches(args, val_loader) 
    test_loader = create_patches(args, test_loader) 
    return train_loader, val_loader, test_loader, args

def load_fashion_mnist(args, label_offset=0, **kwargs):
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.dynamic_binarization = False
    n_validation = 10000

    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../fmnist_data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                               batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../fmnist_data', train=False, download=True,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                              batch_size=args.batch_size, shuffle=True)

    train_ft = train_loader.dataset.train_data.float().reshape(-1, 784) / 255
    train_tar = train_loader.dataset.train_labels + label_offset

    x_train = train_ft[:-n_validation] 
    x_val = train_ft[-n_validation:] 

    y_train = train_tar[:-n_validation] 
    y_val = train_tar[-n_validation:] 

    x_test = test_loader.dataset.test_data.float().reshape(-1, 784) / 255
    y_test = test_loader.dataset.test_labels + label_offset

    train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(x_val, y_val)
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.pseudoinputs_mean = 0.05
    args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

def load_mnist_plus_fmnist(args, **kwargs):
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    import torch.utils.data as data_utils
    
    train_loader, val_loader, test_loader, _ = load_dynamic_mnist(args)
    train_loader2, val_loader2, test_loader2, _ = load_fashion_mnist(args, label_offset=10)

    train_dataset = data_utils.ConcatDataset([train_loader.dataset, train_loader2.dataset]) 
    val_dataset = data_utils.ConcatDataset([val_loader.dataset, val_loader2.dataset]) 
    test_dataset = data_utils.ConcatDataset([test_loader.dataset, test_loader2.dataset]) 

    shuffle = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size,
                                         shuffle=shuffle, **kwargs)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size,
                                         shuffle=shuffle, **kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=shuffle, **kwargs)

    return train_loader, val_loader, test_loader, args

 
def load_svhn(args, **kwargs):

    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'color'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    datapath = '../data'
    dl = True
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root=datapath, 
                                                             split='train', 
                                                             download=dl,
                                                             transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root=datapath, 
                                                             split='test', 
                                                             download=dl,
                                                             transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                               batch_size=args.batch_size, shuffle=True)

    # I know I shouldn't be using test for val, but just bare with me for the moment
    val_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root=datapath, 
                                                             split='test', 
                                                             download=dl,
                                                             transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                               batch_size=args.batch_size, shuffle=True)

    args.input_type = 'color'
    # setting pseudo-inputs inits
    args.pseudoinputs_mean = 0.05
    args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args




def sort_loader(args, loader, **kwargs):
    all_sorted = [] 
    all_labels = []
    for dt, labels in loader:
        _, inds = torch.sort(labels, 0)  
        all_sorted.append(dt[inds].unsqueeze(0))
        all_labels.append(labels[inds])
    all_sorted_cat = torch.cat(all_sorted, 0)
    all_labels_cat = torch.ones(all_sorted_cat.size(0))

    dataset = data_utils.TensorDataset(all_sorted_cat, all_labels_cat)
    loader = data_utils.DataLoader(dataset, batch_size=20, shuffle=True, **kwargs)
    return loader 


def create_patches(args, loader, patch_size=16, **kwargs):
    all_patches = []

    ims = []
    for dt, labels in it.islice(loader, 0, 100):
        dt = dt.unsqueeze(1)
        dt_ysplit = torch.split(dt, patch_size, dim=3) 
        dt_xsplit = []
        for dt_y in dt_ysplit:
            dt_xsplit.extend(torch.split(dt_y, patch_size, dim=4))
        ims.append(torch.cat(dt_xsplit, 1))

    all_sorted_cat = torch.cat(ims, 0)
    all_labels_cat = torch.ones(all_sorted_cat.size(0))

    dataset = data_utils.TensorDataset(all_sorted_cat, all_labels_cat)
    loader = data_utils.DataLoader(dataset, batch_size=20, shuffle=True, **kwargs)

    args.input_size = [3, patch_size, patch_size]
    return loader 

def load_sequential_mnist(args, **kwargs):
    train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)

    train_loader = sort_loader(args, train_loader) 
    val_loader = sort_loader(args, val_loader) 
    test_loader = sort_loader(args, test_loader) 

    return train_loader, val_loader, test_loader, args


def load_dynamic_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    #args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_omniglot(args, n_validation=4000, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = loadmat(os.path.join('datasets', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_ft = reshape_data(omni_raw['data'].T.astype('float32'))
    #train_tar = omni_raw['targetchar'].squeeze() - 1
    train_tar = omni_raw['target'].argmax(0)

    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))
    #y_test = omni_raw['testtargetchar'].squeeze() - 1
    y_test = omni_raw['testtarget'].argmax(0)

    # shuffle train data
    randperm = np.random.permutation(train_ft.shape[0])
    train_ft = train_ft[randperm]
    train_tar = train_tar[randperm]

    #np.random.shuffle(train_ft)
    #np.random.shuffle(train_tar)

    # set train and validation data
    x_train = train_ft[:-n_validation]
    x_val = train_ft[-n_validation:]

    y_train = train_tar[:-n_validation] 
    y_val = train_tar[-n_validation:] 

    # binarize
    if args.dynamic_binarization:
        np.random.seed(777)
        args.input_type = 'binary'
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        raise 'we want binary data!'
        # args.input_type = 'gray'

    # idle y's
    #y_train = omni_raw['targetchar']
    #y_val = np.zeros( (x_val.shape[0], 1) )
    #y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

def load_omniglot_char(args, n_validation=3, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = loadmat(os.path.join('datasets', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_ft = reshape_data(omni_raw['data'].T.astype('float32'))
    #train_tar = omni_raw['targetchar'].squeeze() - 1
    zipped_tar = zip(list(omni_raw['target'].argmax(0)), list(omni_raw['targetchar'].squeeze() -1))
    train_tar =  [j + 50*i for (i,j) in zipped_tar]
    unique = list(np.unique(train_tar))
    train_tar_un = np.array([unique.index(tar) for tar in train_tar])

    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))
    #y_test = omni_raw['testtargetchar'].squeeze() - 1
    zipped_tartest = zip(list(omni_raw['testtarget'].argmax(0)), list(omni_raw['testtargetchar'].squeeze() -1))
    test_tar =  [j + 50*i for (i,j) in zipped_tartest]
    unique = list(np.unique(test_tar))
    y_test = np.array([unique.index(tar) for tar in test_tar])

    # do the split for validation / train
    inds_val = np.zeros(train_tar_un.shape[0]).astype('bool')
    for k in range(max(train_tar_un)):
        inds = np.where(train_tar_un == k)[0]
        inds_val[inds[:n_validation]] = 1
    inds_tr = (1 - inds_val).astype('bool')

    # set train and validation data
    x_train = train_ft[inds_tr]
    x_val = train_ft[inds_val]

    y_train = train_tar_un[inds_tr] 
    y_val = train_tar_un[inds_val] 

    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args


# ======================================================================================================================
def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    caltech_raw = loadmat(os.path.join('datasets', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_histopathologyGray(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/HistopathologyGray/histopathology.pkl', 'rb') as f:
        data = pickle.load(f)

    x_train = np.asarray(data['training']).reshape(-1, 28 * 28)
    x_val = np.asarray(data['validation']).reshape(-1, 28 * 28)
    x_test = np.asarray(data['test']).reshape(-1, 28 * 28)

    x_train = np.clip(x_train, 1./512., 1. - 1./512.)
    x_val = np.clip(x_val, 1./512., 1. - 1./512.)
    x_test = np.clip(x_test, 1./512., 1. - 1./512.)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_freyfaces(args, TRAIN = 1565, VAL = 200, TEST = 200, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f)

    data = (data[0] + 0.5) / 256.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28*20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28*20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28*20)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_cifar10(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    training_dataset = datasets.CIFAR10('datasets/Cifar10/', train=True, download=True, transform=transform)
    train_data = np.clip((training_dataset.train_data + 0.5) / 256., 0., 1.)
    train_data = np.swapaxes( np.swapaxes(train_data,1,2), 1, 3)
    train_data = np.reshape(train_data, (-1, np.prod(args.input_size)) )
    np.random.shuffle(train_data)

    x_val = train_data[40000:50000]
    x_train = train_data[0:40000]

    # fake labels just to fit the framework
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )

    # train loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation loader
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # test loader
    test_dataset = datasets.CIFAR10('datasets/Cifar10/', train=False, transform=transform )
    test_data = np.clip((test_dataset.test_data + 0.5) / 256., 0., 1.)
    test_data = np.swapaxes( np.swapaxes(test_data,1,2), 1, 3)
    x_test = np.reshape(test_data, (-1, np.prod(args.input_size)) )

    y_test = np.zeros((x_test.shape[0], 1))

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_dataset(args, **kwargs):
    if args.dataset_name == 'static_mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    elif args.dataset_name == 'dynamic_mnist':
        args.dynamic_binarization = True
        train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)
    elif args.dataset_name == 'sequential_mnist':
        train_loader, val_loader, test_loader, args = load_sequential_mnist(args, **kwargs)
    elif args.dataset_name == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset_name == 'caltech101silhouettes':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)
    elif args.dataset_name == 'histopathologyGray':
        train_loader, val_loader, test_loader, args = load_histopathologyGray(args, **kwargs)
    elif args.dataset_name == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset_name == 'cifar10':
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    elif args.dataset_name == 'celeba':
        train_loader, val_loader, test_loader, args = load_celeba(args, **kwargs)
    elif args.dataset_name == 'patch_celeba':
        train_loader, val_loader, test_loader, args = load_patch_celeba(args, **kwargs)
    elif args.dataset_name == 'svhn':
        train_loader, val_loader, test_loader, args = load_svhn(args, **kwargs)
    elif args.dataset_name == 'fashion_mnist':
        train_loader, val_loader, test_loader, args = load_fashion_mnist(args, **kwargs)
    elif args.dataset_name == 'mnist_plus_fmnist':
        train_loader, val_loader, test_loader, args = load_mnist_plus_fmnist(args, **kwargs)
    elif args.dataset_name == 'omniglot_char':
        train_loader, val_loader, test_loader, args = load_omniglot_char(args, **kwargs)
    else:
        raise Exception('There is no support for such dataset name')

    return train_loader, val_loader, test_loader, args
