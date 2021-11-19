import sklearn.metrics
import torch
import torch.nn as nn
import numpy.random as npr
import ot
import time
import pickle
import tslearn.metrics as tslm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import csv


class Data_set(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        if Y is not None:
            self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


class CNNMAD(nn.Module):
    def __init__(self, name, traindata, batchsize, channel, testdata=None, num_class=6):
        super(CNNMAD, self).__init__()
        # torch.manual_seed(10)
        self.name = name
        self.num_class = num_class
        self.DTW = None
        self.trainDataLoader = DataLoader(traindata, batch_size=batchsize, shuffle=True, num_workers=2)
        if testdata is not None:
            self.testDataLoader = DataLoader(testdata, batch_size=batchsize, shuffle=True, num_workers=2)
        self.batchsize = batchsize
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=128, kernel_size=8, stride=1,
                                             padding="same", bias=False),
                                   nn.BatchNorm1d(num_features=128),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same",
                                             bias=False),
                                   nn.BatchNorm1d(num_features=256),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same",
                                             bias=False),
                                   nn.BatchNorm1d(num_features=128),
                                   nn.ReLU())

        self.classifier = nn.Linear(128, num_class)
        self.softmax = nn.LogSoftmax(1)

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.optimizer = torch.optim.Adam([{'params': self.conv2.parameters()},
                                           {'params': self.conv3.parameters()},
                                           {'params': self.conv1.parameters()},
                                           {'params': self.classifier.parameters()}])
        self.crossLoss = nn.CrossEntropyLoss()
        self.epoch = 0
        self.iteration = 0
        self.loss_count = []
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()
            self.classifier = self.classifier.cuda()
            self.crossLoss = self.crossLoss.cuda()
            self.softmax = self.softmax.cuda()

    def g(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def f(self, out, train=True):
        out = out.mean(2)
        if train:
            batchsize = self.current_batchtrain_
        else:
            batchsize = self.current_batchtest_
        out = out.view(batchsize, -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    def forward(self, x, train=True):
        out_conv = self.g(x)
        out = self.f(out_conv, train=train)
        return out, out_conv


    def mini_batch_class_balanced(self, shuffle=True):
        if shuffle:
            rindex = np.random.permutation(len(self.trainSourceLabel))
            self.trainSourceLabel = self.trainSourceLabel[rindex]
            self.trainSourceData = self.trainSourceData[rindex]

        index = []
        for i in range(self.num_class):
            s_index = np.nonzero(self.trainSourceLabel == i)
            s_ind = np.random.permutation(s_index)
            index = np.append(index, s_ind[0:self.sample_size])
            #          print(index)
        index = np.array(index, dtype=int)
        return index

    def train_epoch(self):
            self.train()
            self.new_iteration_cnn()
            index = self.mini_batch_class_balanced()
            self.optimizer.zero_grad()
            inputs = torch.tensor(self.trainSourceData[index]).type(torch.float)
            labels = torch.tensor(self.trainSourceLabel[index])
            self.set_current_batchsize(inputs.shape[0])
            out, out_conv = self.forward(inputs.transpose(1, 2))
            loss = self.crossLoss(torch.exp(out.float()), labels)
            loss.backward()
            self.optimizer.step()
            self.loss_count_cnn.append(loss.detach().item())

    def fit(self, epochs):

        while self.epoch < epochs:
            self.new_epoch()
            self.train_epoch()
            """if self.epoch % 1000 == 0:
                print(self.epoch, self.iteration)
                torch.save(self.state_dict(), "/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.epoch) + '.pt')
                loss_save = np.array(self.loss_count)
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.epoch) + 'loss.npy', loss_save)
                self.evaluate(self.trainDataLoader, dataset_name="train_" + str(self.epoch), inner=True)
                self.evaluate(self.testDataLoader, dataset_name="test_" + str(self.epoch), inner=True)"""

    def new_epoch(self):
        self.epoch += 1

    def new_iteration(self):
        self.iteration += 1

    def test_epoch(self, test_dat):
        # sets the model to train mode: no dropout is applied
        self.eval()
        test_dat = DataLoader(test_dat, batch_size=self.batchsize, shuffle=True)
        with torch.no_grad():
            loss = 0
            correct = 0
            for iteration, data in enumerate(test_dat):
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # inputs = inputs.to(torch.FloatTensor)
                out, out_conv = self.forward(inputs.transpose(1, 2))

                loss += self.crossLoss(out, labels)
                pred = out.argmax(dim=1, keepdim=True)
                pred.eq(labels.view_as(pred)).sum().item()
                correct += pred.eq(labels.view_as(pred)).sum().item()

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(loss, correct, len(test_dat),
                                                                                 100. * correct / len(test_dat)))

    def evaluate(self, test_data, dataset_name="test", inner=False):
        self.eval()
        if inner is False:
            test_data = DataLoader(test_data, batch_size=100, shuffle=True)
        with torch.no_grad():
            loss = 0
            correct = 0
            for iteration, data in enumerate(test_data):
                inputs, target = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda()
                self.set_current_batchsize(inputs.shape[0])
                out, out_cnn = self.forward(inputs.transpose(1, 2))
                out_cnn_mean = out_cnn.mean(2)
                loss += self.crossLoss(out, target)

                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_OTcost_.npy",
                #         self._cost_OT)
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_rout_conv.npy",
                        out_cnn.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_out_conv.npy",
                        out_cnn_mean.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_prediction.npy",
                        pred.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_target.npy",
                        target.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_confusion_mat.npy",
                        sklearn.metrics.confusion_matrix(target.cpu(), pred.cpu()))

            loss /= len(test_data.dataset)

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len(test_data.dataset),
                                                                       100. * correct / len(test_data.dataset)))

    def set_current_batchsize(self, dim, train=True):
        if train:
            self.current_batchtrain_ = dim
        else:
            self.current_batchtest_ = dim

    def set_name(self, new_name):
        self.name = new_name


if __name__ == '__main__':

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

    """pairs = [[14, 19]]
    # pairs = [[12, 18], [14, 19], [17, 25], [18, 23], [2, 11], [6, 23], [7, 13], [7, 24], [9, 18]]

    for pa in pairs:
        source = pa[0]
        target = pa[1]

        train_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                              'train.npy')
        train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(source) +
                                            'train_labels.npy', float_or_long=False)
        test_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) + 'test.npy')
        test_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                           'test_labels.npy', float_or_long=False)
        test4train_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) + 'train.npy')
        test4train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/ucihar_' + str(target) +
                                                 'train_labels.npy', float_or_long=False)

        Datatrain = Data_set(train_dataset_t, train_label_t)
        Datatest = Data_set(test_dataset_t, test_label_t)
        Datatesttrain = Data_set(test4train_t, test4train_label_t)

        print("source is ", source, 'and target is ', target)
        CNN_mod = CNNMAD(traindata=Datatrain, testdata=Datatesttrain, batchsize=256, name="delete_me", channel=9)
        # CNN_mod.fit(5000)
        for i in [1000, 2000, 3000, 4000, 5000]:
            CNN_mod.load_state_dict(torch.load("/share/home/fpainblanc/MAD-CNN/ucihar/" + str(source) + "_" + str(target) +
                                               "/MAD_CNN" + str(i) + ".pt"))
            print("batchsize 256, learning rate 0.001, alpha  0.0001, beta 0.0001")
            CNN_mod.evaluate(Datatest, dataset_name='test')
            # print("train dataset at the end")
            # CNN_mod.evaluate(Datatrain, dataset_name='train')"""

    train_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_1train.npy')
    train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_1train_labels.npy',
                                        float_or_long=False)
    test_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_2test.npy')
    test_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_2test_labels.npy',
                                       float_or_long=False)
    test4train_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_2train.npy')
    test4train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/numpy_data/tarnbzh_2train_labels.npy',
                                             float_or_long=False)

    Datatrain = Data_set(train_dataset_t, train_label_t)
    Datatest = Data_set(test_dataset_t, test_label_t)
    Datatesttrain = Data_set(test4train_t, test4train_label_t)
    print(test_dataset_t.shape, test_label_t.shape)
    """from tslearn.datasets import CachedDatasets
    Xs, y_train, _, _ = CachedDatasets().load_dataset("Trace")
    y = y_train - 1  # everything between 0 and 1
    Xt = Xs.copy()
    y_target = y.copy()
    Xt[np.where(y_target == 0), :] = np.roll(Xt[np.where(y_target == 0), :], 10)  # *0.9
    Xt[np.where(y_target == 1), :] = np.roll(Xt[np.where(y_target == 1), :], 20)
    Xt[np.where(y_target == 2), :] = np.roll(Xt[np.where(y_target == 2), :], 30)  # *1.5
    Xt[np.where(y_target == 3), :] = np.roll(Xt[np.where(y_target == 3), :], 40)  # * .5
    Xs2 = np.empty(shape=(Xs.shape[0], Xs.shape[1], 2))
    Xs2[:, :, 0] = Xs.squeeze(-1)
    Xs2[:, :, 1] = Xs.squeeze(-1)
    Xt2 = np.empty(shape=(Xs.shape[0], Xs.shape[1], 2))
    Xt2[:, :, 0] = Xt.squeeze(-1)
    Xt2[:, :, 1] = Xt.squeeze(-1)
    Xs2 = torch.from_numpy(Xs2)
    Xs2 = Xs2.type(torch.float)
    y = torch.from_numpy(y)
    y = y.type(torch.long)
    Xt2 = torch.from_numpy(Xt2)
    Xt2 = Xt2.type(torch.float)
    y_target = torch.from_numpy(y_target)
    y_target = y_target.type(torch.long)
    Datatrain = Data_set(Xs2, y)
    Datatest = Data_set(Xt2, y_target)
    Datatesttrain = Data_set(Xt2, y_target)"""

    CNN_mod = CNNMAD(traindata=Datatrain, testdata=Datatest, batchsize=64, name="tarnbzh_CNN_only",
                     channel=10, num_class=6)
    CNN_mod.load_state_dict(torch.load("/share/home/fpainblanc/MAD-CNN/tarnbzh_CNN_only5000.pt"))
    """with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            CNN_mod.fit(10)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))"""
    # CNN_mod.fit(5000)
    # CNN_mod.evaluate(Datatest, dataset_name='test')
    # CNN_mod.evaluate(Datatrain, dataset_name='train')


