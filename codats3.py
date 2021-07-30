import torch
import torch.nn as nn
import numpy.random as npr
import ot
import time
import pickle
import tslearn.metrics as tslm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import csv


class Data_set(Dataset):
    def __init__(self, X, Y=None, Z=None):
        self.X = X
        if Y is not None:
            self.Y = Y
        if Z is not None:
            self.Z = Z
        else:
            self.Z = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        if self.Z is not None:
            return self.X[item], self.Y[item], self.Z[item]
        else:
            return self.X[item], self.Y[item]


class CoDATS(nn.Module):
    def __init__(self, name, traindata, testdata, batchsize):
        super(CoDATS, self).__init__()
        # torch.manual_seed(10)
        self.name = name
        self.trainDataLoader = DataLoader(traindata, batch_size=batchsize, shuffle=True)
        if testdata is not None:
            self.testDataLoader = DataLoader(testdata, batch_size=batchsize, shuffle=True)
        self.batchsize = batchsize
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=10, out_channels=128, kernel_size=8, stride=1, padding="same",
                                             bias=False),
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

        self.domain_classifier = nn.Sequential(nn.Linear(128, 500, bias=False),
                                               nn.BatchNorm1d(num_features=500),
                                               nn.ReLU(),
                                               nn.Dropout(0.3),
                                               nn.Linear(500, 500, bias=False),
                                               nn.BatchNorm1d(num_features=500),
                                               nn.ReLU(),
                                               nn.Dropout(0.3),
                                               nn.Linear(500, 2))
        self.softmax_domain = nn.Softmax(1)

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)

        self.optimizer = torch.optim.Adam([{'params': self.conv2.parameters()},
                                           {'params': self.conv3.parameters()},
                                           {'params': self.conv1.parameters()},
                                           {'params': self.domain_classifier.parameters()}])
        self.domainLoss = nn.CrossEntropyLoss()
        self.epoch = 0
        self.loss_domain_count = []
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()
            self.domain_classifier = self.domain_classifier.cuda()
            self.domainLoss = self.domainLoss.cuda()
            self.softmax_domain = self.softmax_domain.cuda()

    def g(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def d(self, out_train, out_test=None, train=True):
        if out_test is None:
            out_train = out_train.mean(2)
            if train:
                batchsize = self.current_batchtrain_
            else:
                batchsize = self.current_batchtest_
            out = out_train.view(batchsize, -1)
            out = self.domain_classifier(out)
            out = self.softmax_domain(out)
        else:
            out_train = out_train.mean(2)
            out_test = out_test.mean(2)
            out = torch.cat((out_train, out_test), 0)
            batchsize = self.current_batchtrain_ + self.current_batchtest_
            out = out.view(batchsize, -1)
            out = self.domain_classifier(out)
            out = self.softmax_domain(out)
        return out

    def forward_domain(self, train, test=None, train_b=True):
        if test is None:
            out_conv_train = self.g(train)
            out = self.d(out_conv_train, train=train_b)
            return out, out_conv_train
        else:
            out_conv_train = self.g(train)
            out_conv_test = self.g(test)
            out = self.d(out_train=out_conv_test, out_test=out_conv_train)
            return out, out_conv_test, out_conv_train

    def dspace(self, out_train, train=True):
        out_train = out_train.mean(2)
        if train:
            batchsize = self.current_batchtrain_
        else:
            batchsize = self.current_batchtest_
        out = out_train.view(batchsize, -1)
        out = self.domain_classifier(out)
        out = self.softmax_domain(out)
        return out

    def space_foward(self, train):
        out_conv = self.g(train)
        out = self.dspace(out_conv)
        return out, out_conv

    def train_epoch(self):
        self.train()
        for iteration, data in enumerate(self.trainDataLoader):
            self.optimizer.zero_grad()
            inputs, domain = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                domain = domain.cuda()
            self.set_current_batchsize(inputs.shape[0])
            out_dom_train, out_conv_train = self.space_foward(inputs.transpose(1, 2))
            loss_domain = self.domainLoss(out_dom_train, domain)

            # inputs_test, domain_test = next(iter(self.testDataLoader))
            """if torch.cuda.is_available():
                inputs_test = inputs_test.cuda()
                domain_test = domain_test.cuda()
            self.set_current_batchsize(inputs_test.shape[0], train=False)"""
            """domain_conc = torch.cat((domain_test, domain), 0)
            out_domain, out_conv_train, out_conv_test = self.forward_domain(inputs.transpose(1, 2),
                                                                            inputs_test.transpose(1, 2))
            loss_domain = self.domainLoss(out_domain.float(), domain_conc)"""
            """out_dom_test, out_conv_test = self.space_foward(inputs_test.transpose(1, 2))
            loss_domain_test = self.domainLoss(out_dom_test.float(), domain_test)
            loss_domain = loss_domain + loss_domain_test"""
            # loss = loss + loss_domain
            loss_domain.backward()
            self.optimizer.step()

        self.loss_domain_count.append(loss_domain.detach().item())

    def fit(self, epochs):
        """
        fits a CNN-MAD model during some epochs
        :param epochs: number of epochs to be done for training
        :return: asks for nothing in return

        X0 = torch.arange(100).repeat(50)
        X1 = torch.arange(50).repeat(50, 2)
        Y0 = torch.zeros(size=(50))
        Y1 = torch.zeros(size=(50))
        series = torch.cat((X0, X1), 0)
        labels = torch.cat((Y0, Y1), 0)
        target = torch.arange(100, 200).repeat(50)
        source = data_set(series, labels)
        target = data_set(target)

        model = CNNMAD(source, target, 32, True)
        model.fit()
        """
        while self.epoch < epochs:
            self.new_epoch()
            self.train_epoch()
            if self.epoch % 1000 == 0:
                print(self.epoch)
                torch.save(self.state_dict(), self.name + str(self.epoch) + '.pt')
                loss_dom_save = np.asarray(self.loss_domain_count)
                np.save(self.name + str(self.epoch) + "loss_dom.npy", loss_dom_save)
                self.evaluate_domain(self.trainDataLoader, data_ok=True)

    def new_epoch(self):
        self.epoch += 1

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

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct,
                                                                                     len(test_dat),
                                                                                     100. * correct /
                                                                                     len(test_dat)))

    def evaluate(self, test_data):
        self.eval()
        test_data = DataLoader(test_data, batch_size=self.batchsize, shuffle=True)
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

                loss += self.crossLoss(out, target)

                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            loss /= len(test_data.dataset)

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
            loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))

    def evaluate_domain(self, test_data, data_ok=False):
        self.eval()
        if data_ok is False:
            test_data = DataLoader(test_data, batch_size=self.batchsize, shuffle=True)
        with torch.no_grad():
            loss_dom = 0
            correct_dom = 0
            for iteration, data in enumerate(test_data):
                inputs, domain = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    domain = domain.cuda()

                self.set_current_batchsize(inputs.shape[0])
                """out_dom, out_cnn_dom = self.forward_domain(inputs.transpose(1, 2))
                loss_dom += self.domainLoss(out_dom, domain)"""

                out_dom, out_cnn_dom = self.space_foward(inputs.transpose(1, 2))
                # print(out_dom)
                loss_dom += self.domainLoss(out_dom, domain)

                pred_dom = out_dom.data.max(1, keepdim=True)[1]
                correct_dom += pred_dom.eq(domain.data.view_as(pred_dom)).cpu().sum()

            loss_dom /= len(test_data.dataset)

        print('Average loss domain: {:.4f}, Accuracy domain: {}/{} ({:.3f}%)'.format(
            loss_dom, correct_dom, len(test_data.dataset),
            100. * correct_dom / len(test_data.dataset)))

    def set_current_batchsize(self, dim, train=True):
        if train:
            self.current_batchtrain_ = dim
        else:
            self.current_batchtest_ = dim


if __name__ == '__main__':

    train_dataset = np.load('tarn_short_norm.npy')
    train_dataset_t = torch.from_numpy(train_dataset)
    train_dataset_t = train_dataset_t.type(torch.float)

    train_label = np.load('tarn_short_norm_label.npy')
    train_label_t = torch.from_numpy(train_label)
    train_label_t = train_label_t.type(torch.long)

    test_dataset = np.load('bzh_short_norm.npy')
    test_pad = np.zeros(shape=(test_dataset.shape[0], train_dataset.shape[1], test_dataset.shape[2]))
    test_pad[:, :test_dataset.shape[1]] = test_dataset

    train_test = np.concatenate((train_dataset, test_pad), 0)
    train_test_t = torch.from_numpy(train_test)
    train_test_t = train_test_t.type(torch.float)

    test_dataset_t = torch.from_numpy(test_pad)
    test_dataset_t = test_dataset_t.type(torch.float)

    test_label = np.load('bzh_short_norm_label.npy')
    test_label_t = torch.from_numpy(test_label)
    test_label_t = test_label_t.type(torch.long)

    test4train = np.load('bzh4tarn.npy')
    test4train_t = torch.from_numpy(test4train)
    test4train_t = test4train_t.type(torch.float)

    test4train_label = np.load('bzh4tarn_label.npy')
    test4train_label_t = torch.from_numpy(test4train_label)
    test4train_label_t = test4train_label_t.type(torch.long)

    """tt_domain1 = np.ones(shape=50)
    tt_domain2 = np.zeros(shape=50)
    tt_domain = np.concatenate((tt_domain1, tt_domain2))
    index = np.arange(100)
    npr.shuffle(index)
    test4train_domain = tt_domain[index]"""

    test4train_domain = np.zeros(shape=(100))
    test4train_domain_t = torch.from_numpy(test4train_domain)
    test4train_domain_t = test4train_domain_t.type(torch.long)

    """test_domain1 = np.ones(shape=200)
    test_domain2 = np.zeros(shape=200)
    test_domain = np.concatenate((test_domain1, test_domain2))
    index = np.arange(400)
    npr.shuffle(index)
    train_domain = test_domain[index]"""

    test_domain = np.zeros(shape=(400))
    test_domain_t = torch.from_numpy(test_domain)
    test_domain_t = test_domain_t.type(torch.long)

    """train_domain1 = np.ones(shape=250)
    train_domain2 = np.zeros(shape=250)
    train_domain = np.concatenate((train_domain1, train_domain2))
    index = np.arange(500)
    npr.shuffle(index)
    train_domain = train_domain[index]"""

    train_domain = np.ones(shape=(500))
    train_domain_t = torch.from_numpy(train_domain)
    train_domain_t = train_domain_t.type(torch.long)

    train_test_domain = np.concatenate((train_domain, test_domain), 0)
    train_test_domain_t = torch.from_numpy(train_test_domain)
    train_test_domain_t = train_test_domain_t.type(torch.long)

    Datatrain = Data_set(train_dataset_t, train_domain_t)
    Datatest = Data_set(test_dataset_t, test_domain_t)
    Datatesttrain = Data_set(test4train_t, test4train_domain_t)
    data_train_pad = Data_set(train_test_t, train_test_domain_t)

    Datatrain_test = Data_set(train_dataset_t, train_label_t)

    # CNN_mod = CNNMAD(traindata=Datatrain, testdata=None, batchsize=256, mad_cnn=False)
    CNN_mod = CoDATS(traindata=data_train_pad, testdata=Datatest, batchsize=100, name="Codats/Codats_class")
    # CNN_mod.load_state_dict(torch.load("Codats/Codats10000.pt"))
    CNN_mod.fit(2000)
    CNN_mod.evaluate_domain(Datatesttrain)
    CNN_mod.evaluate_domain(Datatest)
    CNN_mod.evaluate_domain(Datatrain)
    """for i in range(1, 11):
        CNN_mod = CoDATS(traindata=Datatrain, testdata=Datatest, batchsize=100, name="Codats/Codats_dom")
        CNN_mod.load_state_dict(torch.load("Codats/Codats_dom" + str(int(i * 1000)) + ".pt"))
        # CNN_mod.load_state_dict(torch.load("Codats/Codats10000.pt"))
        # CNN_mod.fit(10000)
        print(i)
        CNN_mod.evaluate_domain(Datatesttrain)
        CNN_mod.evaluate_domain(Datatest)
        CNN_mod.evaluate_domain(Datatrain)"""
    # CNN_mod.evaluate(Datatesttrain)
    """CNN_mod_eval = CNNMAD(Datatrain, Datatest, 128, mad_cnn=False)
    CNN_mod_eval.load_state_dict(torch.load("CNN3000.pt"))
    CNN_mod_eval.evaluate(Datatest)"""
