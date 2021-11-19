import sklearn.metrics
import torch
import torch.nn as nn
import numpy.random as npr
import os
import time
import pickle
import tslearn.metrics as tslm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import argparse
import warnings
# from CNN import CNNMAD, Data_set
from CNN_class_balanced import CNNMAD, Data_set


if __name__ == '__main__':
    # Hyperparameters :
    # Alpha = 0.01, 0.001, 0.0001
    # Beta = 0.01, 0.001, 0.0001
    # LR = 0.01, 0.001
    # batchsize = 128, 256
    # Early stopping every 1000 epochs (on 5000)

    # Sacrificed pair
    # 14-19
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--pair1', type=int, help='The first half of the pair')
    parser.add_argument('-p2', '--pair2', type=int, help='The second half of the pair')
    parser.add_argument('-data', '--dataset', type=str, help="Which dataset to take between HAR and TarnBZH")
    parser.add_argument('-bs', '--batchsize', type=int, help='The batchsize')
    parser.add_argument("-a", "--alpha", type=float, help="Alpha")
    parser.add_argument("-b", "--beta", type=float, help='Beta')
    parser.add_argument('-lr', "--learning_rate", type=float, help="The learning rate")
    parser.add_argument('-p', "--path", type=bool)
    parser.add_argument('-bm', "--bigmodel", type=str)

    args, _ = parser.parse_known_args()

    bigmodel = True
    if args.bigmodel == "False":
        bigmodel = False
    print(bigmodel)

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]


    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t


    if args.dataset == "BZHTARN":
        # Train source dataset
        cn = 6
        chan = 10
        if args.path:
            path = '/share/home/fpainblanc/MAD-CNN/numpy_data/'
        else:
            path = "/home/adr2.local/painblanc_f/Desktop/bzh_datasets/"
        train_source = from_numpy_to_torch(path + 'tarnbzh_2train.npy')
        train_source_label = from_numpy_to_torch(path + 'tarnbzh_2train_labels.npy', float_or_long=False)

        valid_source = from_numpy_to_torch(path + 'tarnbzh_2valid.npy')
        valid_source_label = from_numpy_to_torch(path + 'tarnbzh_2valid_labels.npy', float_or_long=False)

        test_source = from_numpy_to_torch(path + 'tarnbzh_2test.npy')
        test_source_label = from_numpy_to_torch(path + 'tarnbzh_2test_labels.npy', float_or_long=False)

        train_target = from_numpy_to_torch(path + 'tarnbzh_1train.npy')
        train_target_labels = from_numpy_to_torch(path + 'tarnbzh_1train_labels.npy', float_or_long=False)

        valid_target = from_numpy_to_torch(path + 'tarnbzh_1valid.npy')
        valid_target_label = from_numpy_to_torch(path + 'tarnbzh_1valid_labels.npy', float_or_long=False)

        test_target = from_numpy_to_torch(path + 'tarnbzh_1test.npy')
        test_target_label = from_numpy_to_torch(path + 'tarnbzh_1test_labels.npy', float_or_long=False)

        name = "hyperparameters/BZHTARN"

    if args.dataset == "TARNBZH":
        # Train source dataset
        cn = 6
        chan = 10
        if args.path:
            path = '/share/home/fpainblanc/MAD-CNN/numpy_data/'
        else:
            path = "/home/adr2.local/painblanc_f/Desktop/bzh_datasets/"
        train_source = from_numpy_to_torch(path + 'tarnbzh_1train.npy')
        train_source_label = from_numpy_to_torch(path + 'tarnbzh_1train_labels.npy', float_or_long=False)

        valid_source = from_numpy_to_torch(path + 'tarnbzh_1valid.npy')
        valid_source_label = from_numpy_to_torch(path + 'tarnbzh_1valid_labels.npy', float_or_long=False)

        test_source = from_numpy_to_torch(path + 'tarnbzh_1test.npy')
        test_source_label = from_numpy_to_torch(path + 'tarnbzh_1test_labels.npy', float_or_long=False)

        train_target = from_numpy_to_torch(path + 'tarnbzh_2train.npy')
        train_target_labels = from_numpy_to_torch(path + 'tarnbzh_2train_labels.npy', float_or_long=False)

        valid_target = from_numpy_to_torch(path + 'tarnbzh_2valid.npy')
        valid_target_label = from_numpy_to_torch(path + 'tarnbzh_2valid_labels.npy', float_or_long=False)

        test_target = from_numpy_to_torch(path + 'tarnbzh_2test.npy')
        test_target_label = from_numpy_to_torch(path + 'tarnbzh_2test_labels.npy', float_or_long=False)

        name = "hyperparameters/TARNBZH"
    # for pair in [[17, 25]]:
    # for pair in [[7,  13], [7,  24], [9 , 18]]:
    if args.dataset == "HAR":
        for pair in [[7,  24]]:
            # for pair in [[12, 18], [14, 19], [17, 25]]:
            chan = 9
            cn = 6

            source = args.pair1
            target = args.pair2
            # Train source dataset
            train_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                               'train.npy')
            train_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                     'train_labels.npy', float_or_long=False)
            # Valid source dataset
            valid_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                               'valid.npy')
            valid_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                     'valid_labels.npy', float_or_long=False)
            # Test source dataset
            test_source = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                              'test.npy')
            test_source_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                                    'test_labels.npy', float_or_long=False)
            # Train target dataset (no labels)
            train_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                               'train.npy')
            train_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                     'train_labels.npy', float_or_long=False)
            # Valid source dataset
            valid_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                               'valid.npy')
            valid_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                     'valid_labels.npy', float_or_long=False)
            # Test target dataset
            test_target = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                              'test.npy')
            test_target_label = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                    'test_labels.npy', float_or_long=False)

        name = "hyperparameters/UCIHAR/" + str(args.learning_rate) + "/" + str(args.batchsize) + "/" + \
               str(args.alpha) + "/" + str(args.beta) + "/" + str(source) + "_" + str(target) + "/HAR_retry"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CNN_mod = CNNMAD(train_source_data=train_source, train_source_label=train_source_label,
                         train_target_data=train_target, test_target_label =test_target_label,
                         valid_target_data=valid_target, valid_target_label=valid_target_label,
                         valid_source_data=valid_source, valid_source_label=valid_source_label,
                         test_source_data =test_source , test_target_data  =test_target,
                         test_source_label=test_source_label, batchsize=args.batchsize, name=name, lambd=1.,
                         alpha=args.alpha, beta=args.beta, channel=chan, MAD_class=True, num_class=cn,
                         big_model=bigmodel,
                         saving=True, CUDA_train=True)
        best_iteration = CNN_mod.fit(epochs=0, cnn_epochs=30000)
        print(args.pair1, args.pair2, train_source.shape, train_target.shape)
        print(name)









