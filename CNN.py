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
import numpy as np
import csv
from MAD import CNN_class


class OTDTW:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l1", settings=0,
                 soft_prob_target=None, JDOT=False, classe_unique=None, alpha=1, lambd=1):
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        self.X = X
        self.Y = Y
        self.shapeX = X.shape
        self.shapeY = Y.shape
        self.alpha = alpha
        self.lambd = lambd
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
            self.pi_DTW_idx.append(self.init_DTW_matrix(settings))
            if self.metric == 'l2':
                if self.one_dim:
                    X2 = self.X[self.tab_idx[cl], :, 0] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2)
                else:
                    X2 = self.X[self.tab_idx[cl]] ** 2
                    X2_sum = np.dot(self.Xa_one[self.tab_idx[cl]], X2.transpose(1, 0, -1)).sum(-1)
                self.Xsquared.append(X2)
                self.Xsquared_sum.append(X2_sum[:, None])
        self.JDOT = JDOT
        if JDOT:
            soft_prob_target = soft_prob_target.cpu().numpy()
            self.soft_prob_target = soft_prob_target
            self.mat_CE = self.cross_entropy_OT_loss()

    # Initialisation of the OT_tilde matrix

    def init_OT_matrix(self, settings):
        npr.seed(settings)
        cost_OT = npr.random((self.shapeX[0], self.shapeY[0])) ** 2
        OT_tilde = ot.emd(self.Xa_one, self.Ya_one, cost_OT, numItermax=10000000)
        return OT_tilde

    def cross_entropy_OT_loss(self):
        def to_onehot(y):
            n_values = np.max(y) + 1
            return np.eye(n_values)[y]
        classe_onehote = to_onehot(self.classe)
        classe_onehote = classe_onehote.reshape(self.shapeX[0], len(self.classe_unique))

        def cross_entropy(p, q):
            return -sum([p[i] * np.log(q[i]) for i in range(len(p))])

        mat_CE = np.zeros(shape=(self.shapeX[0], self.shapeY[0]))
        for i in range(0, self.shapeX[0]):
            for j in range(0, self.shapeY[0]):
                mat_CE[i, j] = cross_entropy(classe_onehote[i], self.soft_prob_target[j])
        return mat_CE

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

    def main_training(self, max_init=100, first_step_DTW=True, barycenter=False, ent_reg=0, l1_reg=False):
        cost = {"Cost": []}
        stop = False
        current_init = 0
        time_mastering = []
        # Begin training
        while stop is not True and current_init < max_init:
            time_start = time.time()
            if (current_init != 0) or (first_step_DTW is False):
                if self.JDOT:
                    Cost_OT = self.alpha * self.mat_cost_OT() + self.lambd * self.mat_CE
                else:
                    Cost_OT = self.mat_cost_OT()
                if ent_reg != 0:
                    self.OT_tilde = ot.sinkhorn(a=self.Xa_one, b=self.Ya_one, M=Cost_OT, reg=ent_reg,
                                                numItermax=100)
                elif l1_reg:
                    self.OT_tilde = ot.sinkhorn_lpl1_mm(self.Xa_one, self.classe, self.Ya_one, M=Cost_OT, reg=100,
                                                        numItermax=100)
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
            time_end = time.time()
            time_mastering.append(time_end - time_start)
        name_init = "cost_test" + str(current_init) + ".pickle"
        with open(name_init, 'wb') as handle:
            pickle.dump(cost, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if barycenter:
            transp_X = self.DTW_barycentring_mapping()
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT, transp_X
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT, time_mastering, current_init


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
    def __init__(self, name, traindata, batchsize, channel, lambd=1., alpha=0.0001, beta=0.0001,
                 testdata=None):
        super(CNNMAD, self).__init__()
        # torch.manual_seed(10)
        self.name = name
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.trainDataLoader = DataLoader(traindata, batch_size=batchsize, shuffle=True)
        if testdata is not None:
            self.testDataLoader = DataLoader(testdata, batch_size=batchsize, shuffle=True)
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

        self.classifier = nn.Linear(128, 6)
        self.softmax = nn.Softmax(1)

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
        self.loss_count = []
        self.loss_MAD_count = []
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()
            self.classifier = self.classifier.cuda()
            self.crossLoss = self.crossLoss.cuda()
            self.softmax = self.softmax.cuda()

    def mad(self, out_conv_train, data_test, labels):
        with torch.no_grad():
            out_test, out_conv_test = self.forward(data_test.transpose(1, 2), train=False)
            mad = OTDTW(out_conv_train.transpose(1, 2), out_conv_test.transpose(1, 2), labels, JDOT=True,
                        soft_prob_target=out_test, metric="l2")
            OT, DTW, cost, score, timing, it = mad.main_training()
            return OT, DTW

    def loss_CNN_MAD(self, labels, OT, DTW, out_source, out_target, softmax_target):
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
        OT = torch.from_numpy(OT)
        res = torch.empty(size=(self.current_batchtrain_, self.current_batchtest_))
        mat_CE = torch.zeros(size=(self.current_batchtrain_, self.current_batchtest_))
        if torch.cuda.is_available():
            OT = OT.cuda()
            res = res.cuda()
            mat_CE = mat_CE.cuda()
        """
        Numpy 
        res = np.empty(shape=(self.batchsize, self.batchsize))
        for cl in np.unique(targets):
            idx_cl = np.where(targets == cl)
            pi_DTW = DTW[cl]
            C1 = np.dot(out_source_sq[idx_cl].transpose(0, -1, 1), np.sum(pi_DTW, axis=1)).sum(-1)
            C2 = np.dot(out_target_sq.transpose(0, -1, 1), np.sum(pi_DTW.T, axis=1)).sum(-1)
            C3 = np.tensordot(np.dot(out_source[idx_cl].transpose(0, -1, 1), pi_DTW), out_target,
                              axes=([1, 2], [2, 1]))
            res[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
        out_CNN = OT * res

        def to_onehot(y):
            n_values = np.max(y) + 1
            return np.eye(n_values)[y]
        classe_onehot = to_onehot(targets)

        def cross_entropy(p, q):
            return -sum([p[i] * np.log(q[i]) for i in range(len(p))])

        mat_CE = np.zeros(shape=(self.batchsize, self.batchsize))
        for i in range(0, self.batchsize):
            for j in range(0, self.batchsize):
                mat_CE[i, j] = OT[i, j] * cross_entropy(softmax_target[j], classe_onehot[i])

        out_CNN = out_CNN + mat_CE
        out_CNN_loss = out_CNN.sum()"""
        for cl in range(0, torch.max(labels).item() + 1):
            idx_cl = torch.where(labels == cl)
            pi_DTW = DTW[cl]
            pi_DTW = torch.from_numpy(pi_DTW)
            if torch.cuda.is_available():
                pi_DTW = pi_DTW.cuda()
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
        if torch.cuda.is_available():
            classe_onehot = classe_onehot.cuda()

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
            self.optimizer.zero_grad()
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            self.set_current_batchsize(inputs.shape[0])
            out, out_conv = self.forward(inputs.transpose(1, 2))
            loss = self.lambd * self.crossLoss(out.float(), labels)
            if (self.alpha != 0) or (self.beta != 0):
                # Make sure the batch is changing
                inputs_test, labels_test = next(iter(self.testDataLoader))
                if torch.cuda.is_available():
                    inputs_test = inputs_test.cuda()
                    labels_test = labels_test.cuda()
                self.set_current_batchsize(inputs_test.shape[0], train=False)
                OT, DTW = self.mad(out_conv, inputs_test, labels)
                out_test, out_conv_test = self.forward(inputs_test.transpose(1, 2), train=False)
                loss_cnn_mad = -1. * self.loss_CNN_MAD(labels, OT, DTW, out_conv, out_conv_test, out_test)
                loss = loss + loss_cnn_mad
            loss.backward()
            self.optimizer.step()
        if (self.alpha != 0) or (self.beta != 0):
            self.loss_MAD_count.append(loss_cnn_mad.detach().item())
            self.loss_count.append(loss.detach().item() - loss_cnn_mad.detach().item())
        else:
            self.loss_count.append(loss.detach().item())

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
                torch.save(self.state_dict(), "/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.epoch) + '.pt')
                loss_save = np.array(self.loss_count)
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.epoch) + 'loss.npy', loss_save)
                if (self.alpha != 0) or (self.beta != 0):
                    loss_mad_save = np.asarray(self.loss_MAD_count)
                    np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + str(self.epoch) + 'loss_mad.npy',
                            loss_mad_save)
                self.evaluate(self.trainDataLoader, dataset_name="train_" + str(self.epoch), inner=True)
                self.evaluate(self.testDataLoader, dataset_name="test_" + str(self.epoch), inner=True)

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

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(loss, correct, len(test_dat),
                                                                                 100. * correct / len(test_dat)))

    def evaluate(self, test_data, dataset_name="test", inner=False):
        self.eval()
        if inner is False:
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
                out_cnn = out_cnn.mean(2)
                loss += self.crossLoss(out, target)

                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                np.save("/share/home/fpainblanc/MAD-CNN/" + self.name + "_" + dataset_name + "_out_conv.npy",
                        out_cnn.cpu())
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

    def set_loss_weight(self, lambd, alpha, beta):
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta

    def set_name(self, new_name):
        self.name = new_name


if __name__ == '__main__':

    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t

    train_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/tarn_short_norm.npy')
    train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/tarn_short_norm_label.npy', float_or_long=False)
    test_dataset_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/bzh_short_norm.npy')
    test_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/bzh_short_norm_label.npy', float_or_long=False)
    test4train_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/bzh4tarn.npy')
    test4train_label_t = from_numpy_to_torch('/share/home/fpainblanc/MAD-CNN/bzh4tarn_label.npy', float_or_long=False)

    """train_dataset_t = from_numpy_to_torch('numpy_data/ucihar_14train.npy')
    train_label_t = from_numpy_to_torch('numpy_data/ucihar_14train_labels.npy', float_or_long=False)
    test_dataset_t = from_numpy_to_torch('numpy_data/ucihar_19test.npy')
    test_label_t = from_numpy_to_torch('numpy_data/ucihar_19test_labels.npy', float_or_long=False)
    test4train_t = from_numpy_to_torch('numpy_data/ucihar_19train.npy')
    test4train_label_t = from_numpy_to_torch('numpy_data/ucihar_19train_labels.npy', float_or_long=False)"""

    Datatrain = Data_set(train_dataset_t, train_label_t)
    Datatest = Data_set(test_dataset_t, test_label_t)
    Datatesttrain = Data_set(test4train_t, test4train_label_t)

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    CNN_mod = CNNMAD(traindata=Datatrain, testdata=Datatesttrain, batchsize=256, name="run6/MAD_two_losses",
                     lambd=0., alpha=1., beta=0., channel=10)
    # CNN_mod.set_loss_weight(lambd=1., alpha=0.01, beta=0.01)
    CNN_mod.load_state_dict(torch.load("/share/home/fpainblanc/MAD-CNN/run6/MAD_two_losses1000.pt"))
    # CNN_mod.fit(3000)
    CNN_mod.evaluate(Datatest, dataset_name='test')
    CNN_mod.evaluate(Datatrain, dataset_name='train')

    """print('0,01')
    CNN_mod.set_name("run5/two_Classif_and_MAD")
    CNN_mod.load_state_dict(torch.load("/share/home/fpainblanc/MAD-CNN/run5/two_Classif_and_MAD3000.pt"))
    CNN_mod.set_loss_weight(lambd=1., alpha=0.01, beta=0.01)
    # CNN_mod.fit(3000)
    CNN_mod.evaluate(Datatest, dataset_name='test')
    CNN_mod.evaluate(Datatrain, dataset_name='train')

    print('classif')
    CNN_mod.set_name("run5/just_classif")
    CNN_mod.load_state_dict(torch.load("/share/home/fpainblanc/MAD-CNN/run5/Just_classif2000.pt"))
    CNN_mod.set_loss_weight(lambd=1., alpha=0., beta=0.)
    # CNN_mod.fit(3000)
    CNN_mod.evaluate(Datatest, dataset_name='test')
    CNN_mod.evaluate(Datatrain, dataset_name='train')"""

    """for i in range(1, 8):
        CNN_mod.load_state_dict(torch.load("ucihar/MAD_CNN" + str(int(i*1000)) + ".pt"))
        # CNN_mod.fit(10000)
        print(i)
        CNN_mod.evaluate(Datatest)
        CNN_mod.evaluate(Datatrain)"""
    # CNN_mod.evaluate(Datatesttrain)
