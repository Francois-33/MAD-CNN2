import sklearn.metrics
import torch
import torch.nn as nn
import numpy.random as npr
import ot
import time
import tslearn.metrics as tslm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np


class OTDTW:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l1", settings=0, classe_unique=None,
                 previous_DTW=None):
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        self.X = X
        self.Y = Y
        self.shapeX = X.shape
        self.shapeY = Y.shape
        if classe is not None:
            classe = classe.cpu()
            classe = classe.numpy()
            classe = classe.astype(int)
            cl_count = 0
            classe_corrected = np.empty((self.shapeX[0], 1), dtype=int)
            for cl in np.unique(classe):
                classe_corrected[np.where(classe == cl)] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
        else:
            self.classe = np.zeros((self.shapeX[0], 1), dtype=int)
        if classe_unique is not None:
            self.classe_unique = classe_unique
        else:
            self.classe_unique = np.unique(self.classe)
        if self.shapeX[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        if weights_X is None:
            self.Xa_one = np.ones(self.shapeX[0]) / self.shapeX[0]
        else:
            self.Xa_one = weights_X
        if weights_Y is None:
            self.Ya_one = np.ones(self.shapeY[0]) / self.shapeY[0]
        else:
            self.Ya_one = weights_Y

        self.OT_tilde = self.init_OT_matrix(settings)
        self.metric = metric
        self.tab_idx = []
        self.dist_OT = []
        self.pi_DTW_idx = []
        self.pi_DTW_path_idx = []
        if self.metric == 'l2':
            self.Xsquared = []
            self.Xsquared_sum = []
            if self.one_dim:
                self.Ysquared = self.Y[:, :, 0] ** 2
            else:
                self.Ysquared = self.Y ** 2
        for cl in self.classe_unique:
            self.tab_idx.append(np.where(self.classe == cl)[0])
            if previous_DTW is None:
                self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            else:
                self.pi_DTW_idx.append(previous_DTW[cl])
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = self.X[self.tab_idx[cl], :, 0] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = self.X[self.tab_idx[cl]] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2.transpose(1, 0, -1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        npr.seed(settings)
        cost_OT = npr.random((self.shapeX[0], self.shapeY[0])) ** 2
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        return OT_tilde

    def init_DTW_matrix(self, settings):
        npr.seed(settings)
        DTW_matrix = np.zeros((self.shapeX[1], self.shapeY[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shapeX[1] - 1) or (ts[1] != self.shapeY[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shapeX[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shapeY[1] - 1:
                indice_moving = 0
            else:
                if settings == 100:
                    indice_moving = 1
                else:
                    indice_moving = npr.randint(3)
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        return DTW_matrix

    def ugly_four_sum(self):
        cost = 0
        for cl in self.classe_unique:
            mat_dtw = self.pi_DTW_idx[cl]
            for i in self.tab_idx[cl]:
                for j in range(0, self.shapeY[0]):
                    for t in range(0, self.shapeX[1]):
                        for tp in range(0, self.shapeY[1]):
                            for d in range(0, self.shapeY[-1]):
                                if self.metric == "l1":
                                    cost += self.OT_tilde[i, j] * mat_dtw[t, tp] * np.absolute(self.X[i, t, d] - self.Y[j, tp, d])
                                else:
                                    cost += self.OT_tilde[i, j] * mat_dtw[t, tp] * np.power(self.X[i, t, d] - self.Y[j, tp, d], 2)
        return cost

    def mat_cost_OT(self):
        mat_cost = np.zeros(shape=(self.shapeX[0], self.shapeY[0]))
        if self.one_dim:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = np.dot(self.Xsquared[cl], np.sum(pi_DTW, axis=1))
                    C2 = np.dot(self.Ysquared, np.sum(pi_DTW.T, axis=1))
                    C3 = np.dot(np.dot(self.X[self.tab_idx[cl], :, 0], pi_DTW[:]), self.Y[:, :, 0].T)
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = np.dot(self.X[self.tab_idx[cl], :, 0], m1.T)
                    C2 = np.dot(self.Y[:, :, 0], m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = np.absolute(C3).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        else:
            for cl in self.classe_unique:
                if self.metric == "l2":
                    pi_DTW = self.pi_DTW_idx[cl]
                    C1 = np.dot(self.Xsquared[cl].transpose(0, -1, 1), np.sum(pi_DTW, axis=1)).sum(-1)
                    C2 = np.dot(self.Ysquared.transpose(0, -1, 1), np.sum(pi_DTW.T, axis=1)).sum(-1)
                    C3 = np.tensordot(np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), pi_DTW), self.Y,
                                      axes=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    m1, m2 = self.get_warp_matrices(cl)
                    C1 = np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), m1.T)
                    C2 = np.dot(self.Y.transpose(0, -1, 1), m2.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = np.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        return mat_cost

    def mat_dist_DTW(self, classe_it=None):
        if self.one_dim:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X[:, :, 0]
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it], :, 0]
            if self.metric == "l2":
                C2 = np.dot(OTc.sum(axis=0), self.Ysquared)
                C3 = np.dot(np.dot(Xc.T, OTc), self.Y[:, :, 0])
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = np.dot(w1, Xc)
                C2 = np.dot(w2, self.Y[:, :, 0])
                C3 = np.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0)
        else:
            if classe_it is None:
                OTc = self.OT_tilde
                Xc = self.X
            else:
                OTc = self.OT_tilde[self.tab_idx[classe_it]]
                Xc = self.X[self.tab_idx[classe_it]]
            if self.metric == "l2":
                C2 = np.dot(OTc.sum(axis=0), self.Ysquared.transpose(1, 0, -1)).sum(-1)
                C3 = np.tensordot(np.dot(Xc.T, OTc), self.Y, axes=([0, 2], [2, 0]))
                res = self.Xsquared_sum[classe_it] + C2[None, :] - 2 * C3
            elif self.metric == "l1":
                w1, w2 = self.extended_OT_mat(classe_it)
                C1 = np.dot(Xc.transpose(-1, 1, 0), w1.T)
                C2 = np.dot(self.Y.transpose(-1, 1, 0), w2.T)
                C3 = np.absolute(C1[:, :, None] - C2[:, None, :])
                res = C3.sum(0).sum(-1)
        return res

    def get_warp_matrices(self, cl):

        num_path = int(self.pi_DTW_idx[cl].sum())
        Mx = np.zeros((num_path, self.shapeX[1]))
        My = np.zeros((num_path, self.shapeY[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == self.shapeX[1] - 1) & (l != self.shapeY[1] - 1):
                arg1 = -1
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = -1
            if (l == self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != self.shapeY[1] - 1) & (k != self.shapeX[1] - 1):
                arg1 = self.pi_DTW_idx[cl][k + 1, l]
                arg2 = self.pi_DTW_idx[cl][k, l + 1]
                arg3 = self.pi_DTW_idx[cl][k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def path2mat(self, path):
        pi_DTW = np.zeros((self.shapeX[1], self.shapeY[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def extended_OT_mat(self, classe=None):

        if classe is None:
            lenx = self.shapeX[0]
            OTc = self.OT_tilde
        else:
            lenx = len(self.tab_idx[classe])
            OTc = self.OT_tilde[self.tab_idx[classe]]
        el_t = np.count_nonzero(OTc)

        Wx = np.zeros((el_t, lenx))
        Wy = np.zeros((el_t, self.shapeY[0]))
        index_counting_y = 0
        index_counting_x = 0
        for i in range(0, lenx):
            cnt_z = np.count_nonzero(OTc[i, :])
            nnz = np.nonzero(OTc[i, :])
            Wx[index_counting_x: index_counting_x + cnt_z, i] = OTc[i, nnz]
            index_counting_x = index_counting_x + cnt_z
            for j in nnz[0]:
                Wy[index_counting_y, j] = OTc[i, j]
                index_counting_y = index_counting_y + 1
        return Wx, Wy

    def stopping_criterion(self, last_pi_DTW):
        stop = True
        for cl in self.classe_unique:
            pi_DTW = self.pi_DTW_idx[cl]
            last_DTW = last_pi_DTW[cl]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def DTW_barycentring_mapping(self):
        """We work for each class in the source dataset and at the end we concatenate all the
          transferred source datasets per class to return a single \hat{X}^{s} of size
          (n_{s}, T^{*}, d), where T^{*} is the highest number of DTW steps amongst each class.
          For the other classes with lesser DTW steps, the series will be padded with 0 at the end.

          we start by transporting both source and target datasets' timestamps
          into the extended dimension of aligned time. Then, we apply barycentric mapping to obtain
          the source dataset transferred into the realm of the target."""

        all_num_path = []
        for cl in self.classe_unique:
            all_num_path.append(int(self.pi_DTW_idx[cl].sum()))

        max_num_path = np.max(all_num_path)
        transp_X = np.zeros(shape=(self.shapeX[0], max_num_path, self.shapeX[-1]))

        for cl in self.classe_unique:
            m1, m2 = self.get_warp_matrices(cl)
            dtw_cl_Y = np.dot(self.Y.transpose(0, -1, 1), m2.T)
            transp = self.OT_tilde[self.tab_idx[cl]] / np.sum(self.OT_tilde[self.tab_idx[cl]], 1)[:, None]
            transp_cl_X = np.dot(transp, dtw_cl_Y.transpose(1, 0, -1))
            transp_X[self.tab_idx[cl], :all_num_path[cl]] = transp_cl_X.transpose(0, -1, 1)
        return transp_X

    def main_training(self, max_init=100, first_step_DTW=True, barycenter=False, reg=0.):
        cost = {"Cost": []}
        stop = False
        current_init = 0
        # Begin training
        while stop is not True and current_init < max_init:
            if (current_init != 0) or (first_step_DTW is False):
                Cost_OT = self.mat_cost_OT()
                if reg != 0.:
                    self.OT_tilde = ot.sinkhorn_unbalanced(a=self.Xa_one, b=self.Ya_one, M=Cost_OT, reg=reg, reg_m=reg)
                else:
                    self.OT_tilde = ot.emd(self.Xa_one, self.Ya_one, Cost_OT, numItermax=1000000)
                score_OT = np.sum(self.OT_tilde * Cost_OT)
                cost["Cost"].append(score_OT)

            dtw_score = 0
            self.pi_DTW_path_idx = []
            for cl in self.classe_unique:
                mat_dist = self.mat_dist_DTW(cl)
                Pi_DTW_path, dtw_score_prov = tslm.dtw_path_from_metric(mat_dist, metric="precomputed")
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                self.pi_DTW_idx[cl] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            cost["Cost"].append(dtw_score)
            if current_init != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.pi_DTW_idx.copy()
            current_init = current_init + 1
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT


class Data_set(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        if self.Y is None:
            return self.X[item]
        else:
            return self.X[item], self.Y[item]


class CNNMAD(nn.Module):
    def __init__(self, name, train_source_data, batchsize, channel, lambd=1., alpha=0.0001, beta=0.0001, eval_step=4000,
                 train_target_data=None, valid_target_data=None, valid_source_data=None, test_source_data=None,
                 test_target_data=None, MAD_class=True, reg=0., num_class=6, lr=0.001):
        super(CNNMAD, self).__init__()
        # torch.manual_seed(10)
        self.name = name
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.eval_step = eval_step
        self.lr = lr
        self.num_class = num_class
        self.last_accuracy = 0
        self.DTW = None
        self.trainDataLoader = DataLoader(train_source_data, batch_size=batchsize, shuffle=True, num_workers=2)
        if train_target_data is not None:
            self.testDataLoader = DataLoader(train_target_data, batch_size=batchsize, shuffle=True, num_workers=2)
        if valid_target_data is not None:
            self.validDataLoader = DataLoader(valid_target_data, batch_size=1024, shuffle=True, num_workers=2)
        if valid_source_data is not None:
            self.validSoucreDataLoader = DataLoader(valid_source_data, batch_size=1024, shuffle=True, num_workers=2)
        if test_source_data is not None:
            self.testSourceData = DataLoader(test_source_data, batch_size=1024, shuffle=True, num_workers=2)
        if test_target_data is not None:
            self.TestTargetData = DataLoader(test_target_data, batch_size=1024, shuffle=True, num_workers=2)
        self.batchsize = batchsize
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=32, kernel_size=8, stride=1,
                                             padding="same", bias=True),
                                   # nn.BatchNorm1d(num_features=128),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same",
                                             bias=True),
                                   # nn.BatchNorm1d(num_features=256),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same",
                                             bias=True),
                                   # nn.BatchNorm1d(num_features=128),
                                   nn.ReLU())

        self.classifier = nn.Linear(32, num_class)
        self.softmax = nn.Softmax(1)

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.optimizer = torch.optim.Adam([{'params': self.conv2.parameters()},
                                           {'params': self.conv3.parameters()},
                                           {'params': self.conv1.parameters()},
                                           {'params': self.classifier.parameters()}],
                                          lr=lr)
        self.crossLoss = nn.CrossEntropyLoss()
        self.epoch = 0
        self.iteration = 0
        self.lowest_loss = 1000000
        self.loss_count = []
        self.loss_MAD_count = []
        """if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()
            self.classifier = self.classifier.cuda()
            self.crossLoss = self.crossLoss.cuda()
            self.softmax = self.softmax.cuda()"""
        self.MAD_class = MAD_class

    def mad(self, out_conv_train, out_conv_test, labels):
        with torch.no_grad():
            if self.MAD_class is not True:
                labels = None
            mad = OTDTW(out_conv_train.transpose(1, 2), out_conv_test.transpose(1, 2), labels, metric="l2",
                        classe_unique=np.arange(self.num_class), previous_DTW=self.DTW)
            self._OT, self.DTW, self._cost_OT, score = mad.main_training(reg=self.reg, first_step_DTW=False)

    def loss_CNN_MAD(self, labels, out_source, out_target, softmax_target):
        """
        Compute the loss for the CNN part of CNN-MAD
        :param labels: the true labels of the source batch
        :param OT: An OT matrix from MAD
        :param DTW: DTW matrices from MAD, one per class
        :param out_source: out of the convolutions for the source batch
        :param out_target: out of the convolutions for the target batch
        :param softmax_target: out of the classifier for the target batch
        :return: The loss

        Y0 = torch.zeros(size=(5))
        Y1 = torch.zeros(size=(5))
        labels = torch.cat((Y0, Y1), 0)
        OT = torch.eyes(size=(10, 10))
        DTW = [torch.eyes(size=(20, 20))]
        out_source = torch.ones(size=(10, 20, 2))
        out_target = torch.zeros(size=(10, 20, 2))
        t0 = torch.zeros(size=(10))
        t1 = torch.ones(size(10))
        softmax_target = torch.cat((t0, t1), 1)

        Loss = loss_CNN_MAD(labels, OT, DTW, out_source, out_target, softmax_target)
        """
        out_source_sq = out_source ** 2
        out_target_sq = out_target ** 2
        OT = torch.from_numpy(self._OT)
        res = torch.empty(size=(self.current_batchtrain_, self.current_batchtest_))
        mat_CE = torch.zeros(size=(self.current_batchtrain_, self.current_batchtest_))
        """if torch.cuda.is_available():
            OT = OT.cuda()
            res = res.cuda()
            mat_CE = mat_CE.cuda()"""
        if self.MAD_class:
            loop_iteration = torch.max(labels).item() + 1
        else:
            loop_iteration = 1
        for cl in range(0, loop_iteration):
            idx_cl = torch.where(labels == cl)
            pi_DTW = self.DTW[cl]
            pi_DTW = torch.from_numpy(pi_DTW)
            """if torch.cuda.is_available():
                pi_DTW = pi_DTW.cuda()"""
            pi_DTW = pi_DTW.float()
            C1 = torch.matmul(out_source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
            C2 = torch.matmul(out_target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
            C3 = torch.tensordot(torch.matmul(out_source[idx_cl], pi_DTW), out_target, dims=([1, 2], [1, 2]))
            C12 = C1[:, None] + C2[None, :]
            C23 = 2 * C3
            res[idx_cl] = C12 - C23
        out_CNN = OT * res

        def to_onehot(y):
            n_values = torch.max(y) + 1
            return torch.eye(n_values)[y]
        classe_onehot = to_onehot(labels)
        """if torch.cuda.is_available():
            classe_onehot = classe_onehot.cuda()"""

        def cross_entropy(p, q):
            result = p * torch.log(q)
            result = - result.sum()
            return result

        for i in range(0, self.current_batchtrain_):
            for j in range(0, self.current_batchtest_):
                mat_CE[i, j] = OT[i, j] * cross_entropy(classe_onehot[i], softmax_target[j])
        out_CNN = self.alpha * out_CNN + self.beta * mat_CE
        out_CNN_loss = out_CNN.sum()
        return out_CNN_loss

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

    def train_epoch(self):
        self.train()
        for iteration, data in enumerate(self.trainDataLoader):
            self.new_iteration()
            self.optimizer.zero_grad()
            inputs, labels = data
            """if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()"""
            self.set_current_batchsize(inputs.shape[0])
            out, out_conv = self.forward(inputs.transpose(1, 2))
            loss = self.lambd * self.crossLoss(out.float(), labels)
            if (self.alpha != 0) or (self.beta != 0):
                try:
                    inputs_test = next(self.iter_dataloader)
                except StopIteration:
                    self.iter_dataloader = iter(self.testDataLoader)
                    inputs_test = next(self.iter_dataloader)
                """if torch.cuda.is_available():
                    inputs_test = inputs_test.cuda()"""
                self.set_current_batchsize(inputs_test.shape[0], train=False)
                out_test, out_conv_test = self.forward(inputs_test.transpose(1, 2), train=False)
                self.mad(out_conv, out_conv_test, labels)
                loss_cnn_mad = self.loss_CNN_MAD(labels, out_conv, out_conv_test, out_test)
                loss = loss + loss_cnn_mad
            loss.backward()
            self.optimizer.step()
        if (self.alpha != 0) or (self.beta != 0):
            self.loss_MAD_count.append(loss_cnn_mad.detach().item())
            self.loss_count.append(loss.detach().item() - loss_cnn_mad.detach().item())
        else:
            self.loss_count.append(loss.detach().item())

    def fit(self, epochs, done_epochs=0):
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
        self.epoch = self.epoch + done_epochs
        while self.iteration < epochs:
            self.new_epoch()
            self.iter_dataloader = iter(self.testDataLoader)
            self.train_epoch()
            if (self.iteration % self.eval_step == 0) or (self.iteration == 1) or (self.iteration == epochs):
                current_loss_source, len_source = self.test_epoch(domain="source")
                current_loss_target, len_target = self.test_epoch(domain="target")
                current_loss = (len_source * current_loss_source + len_target * current_loss_target) / \
                               (len_target + len_source)
                if current_loss < self.lowest_loss:
                    self.lowest_loss = current_loss
                    best_model = "_best_model_"
                else:
                    best_model = ""
                torch.save(self.state_dict(), "/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) +
                           best_model + '.pt')
                loss_save = np.array(self.loss_count)
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) + best_model + 'loss.npy',
                        loss_save)
                if (self.alpha != 0) or (self.beta != 0):
                    loss_mad_save = np.asarray(self.loss_MAD_count)
                    np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) + best_model +
                            'loss_mad.npy', loss_mad_save)
                    np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) + best_model +
                            '_OTcost.npy', self._cost_OT)
                    np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) + best_model + '_OT.npy',
                            self._OT)
                    np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.iteration) + best_model +
                            '_DTW.npy', self.DTW)
                print("iteration ", self.iteration, "alpha ", self.alpha, "beta ", self.beta, "learning rate ", self.lr,
                      "batchsize ", self.batchsize, best_model)

        return self.iteration

    def new_epoch(self):
        self.epoch += 1

    def new_iteration(self):
        self.iteration += 1

    def test_epoch(self, domain):
        self.eval()
        with torch.no_grad():
            if domain == "target":
                ValidDataset = self.validDataLoader
            if domain == "source":
                ValidDataset = self.validSoucreDataLoader
            loss = 0
            correct = 0
            for iteration, data in enumerate(ValidDataset):
                inputs, target = data
                """if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda()"""
                self.set_current_batchsize(inputs.shape[0])
                out, out_cnn = self.forward(inputs.transpose(1, 2))
                out_cnn_mean = out_cnn.mean(2)
                loss += self.crossLoss(out, target)

                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_OTcost_.npy",
                #         self._cost_OT)
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + domain + str(self.iteration) + "_valid_rout_conv.npy",
                        out_cnn.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + domain + str(self.iteration) + "_valid_out_conv.npy",
                        out_cnn_mean.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + domain + str(self.iteration) + "_valid_prediction.npy",
                        pred.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + domain + str(self.iteration) + "_valid_target.npy",
                        target.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + domain + str(self.iteration) + "_valid_confusion_mat.npy",
                        sklearn.metrics.confusion_matrix(target.cpu(), pred.cpu()))
            len_data = len(ValidDataset.dataset)
            loss /= len_data
            self.accuracy = 100. * correct / len_data
        print(domain, ":")
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len_data,
                                                                       100. * correct / len_data))
        return loss, len_data

    def evaluate(self, dataset_name="target", name_complement=None):
        self.eval()
        if dataset_name == "target":
            test_data = self.TestTargetData
        if dataset_name == "source":
            test_data = self.TestSourceData
        with torch.no_grad():
            loss = 0
            correct = 0
            for iteration, data in enumerate(test_data):
                inputs, target = data
                """if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda()"""
                self.set_current_batchsize(inputs.shape[0])
                out, out_cnn = self.forward(inputs.transpose(1, 2))
                out_cnn_mean = out_cnn.mean(2)
                loss += self.crossLoss(out, target)

                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + name_complement + "_test_rout_conv.npy",
                        out_cnn.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + name_complement + "_test_out_conv.npy",
                        out_cnn_mean.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + name_complement +"_test_prediction.npy",
                        pred.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + name_complement +"_test_target.npy",
                        target.cpu())
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + name_complement +"_test_confusion_mat.npy",
                        sklearn.metrics.confusion_matrix(target.cpu(), pred.cpu()))

            loss /= len(test_data.dataset)

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len(test_data.dataset),
                                                                       100. * correct / len(test_data.dataset)))

    def set_current_batchsize(self, dim, train=True):
        if train:
            self.current_batchtrain_ = dim
        else:
            self.current_batchtest_ = dim

    def set_loss_weight(self, lambd, alpha, beta):
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta

    def set_name(self, new_name):
        self.name = new_name

    def forward_MAD(self, train=True):
        if train:
            inputs, labels = next(iter(self.trainDataLoader))
            out_conv = self.forward(inputs.transpose(1, 2))
            train_test = "train"
        else:
            inputs, labels = next(iter(self.testDataLoader))
            out_conv = self.forward(inputs.transpose(1, 2))
            train_test = "test"

        self.mad(out_conv, out_conv, labels)

        np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + train_test + str(self.iteration) + 'DTW_forward_MAD.npy',
                self.DTW)
        np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + train_test + str(self.iteration) + 'OT_forward_MAD.npy',
                self._OT)
        np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + train_test + str(self.iteration) +
                'OT_Cost_forward_MAD.npy', self._cost_OT)
        np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + train_test + str(self.iteration) + 'Conv_test.npy',
                out_conv.detach().numpy())
