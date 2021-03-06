import ot
import pickle
import numpy as np
import numpy.random as npr
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
import tslearn.metrics as tslm
import tensorflow as tf
import csv
import scipy as sp
import time
import matplotlib.pyplot as plt
# import datasets
# import methods


class OTDTW:
    def __init__(self, X, Y, classe=None, weights_X=None, weights_Y=None, metric="l1", settings=0):
        self.X = X
        self.Y = Y
        self.shapeX = X.shape
        self.shapeY = Y.shape
        if classe is not None:
            classe = classe.astype(int)
            cl_count = 0
            classe_corrected = np.empty((self.shapeX[0], 1), dtype=int)
            for cl in np.unique(classe):
                classe_corrected[np.where(classe == cl)] = cl_count
                cl_count = cl_count + 1
            self.classe = classe_corrected
        else:
            self.classe = np.zeros((self.shapeX[0], 1), dtype=int)
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
        """DTW_matrix = np.ones((self.shapeX[1], self.shapeY[1]))
        zero_mat = np.zeros((self.shapeX[1] - 1, self.shapeY[1] - 1))
        DTW_matrix[1:, :self.shapeY[1] - 2] = zero_mat"""
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

    def sinkhorn_lpl1_mm(self, a, labels_a, b, M, reg, eta=0.1, numItermax=10, numInnerItermax=200, stopInnerThr=1e-9,
                         verbose=False, log=False):
        p = 0.5
        epsilon = 1e-3

        indices_labels = []
        classes = np.unique(labels_a)
        for c in classes:
            idxc = np.where(labels_a == c)
            indices_labels.append(idxc)

        W = np.zeros(M.shape)

        for cpt in range(numItermax):
            Mreg = M + eta * W
            transp = ot.sinkhorn(a, b, Mreg, reg, numItermax=numInnerItermax, stopThr=stopInnerThr)
            # the transport has been computed. Check if classes are really
            # separated
            W = np.ones(M.shape)
            for (i, c) in enumerate(classes):
                majs = np.sum(transp[indices_labels[i]], axis=0)
                majs = p * ((majs + epsilon) ** (p - 1))
                W[indices_labels[i]] = majs

        return transp

    def main_training(self, init=0, max_init=100, first_step_DTW=False, barycenter=False):
        cost = {"Cost": []}
        stop = False
        current_init = 0
        time_mastering = []
        # Begin training
        while stop is not True and current_init < max_init:
            time_start = time.time()
            if (current_init != 0) or (first_step_DTW is False):
                Cost_OT = self.mat_cost_OT()
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
        name_init = "cost_test" + str(init) + ".pickle"
        with open(name_init, 'wb') as handle:
            pickle.dump(cost, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if barycenter:
            transp_X = self.DTW_barycentring_mapping()
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT, score_OT, transp_X
        else:
            return self.OT_tilde, self.pi_DTW_idx, Cost_OT


class OTonly(OTDTW):

    def mat_cost_OT(self):
        mat_cost = np.zeros(shape=(self.shapeX[0], self.shapeY[0]))
        pi_DTW = np.eye(N=self.shapeY[1])
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
                    C1 = np.dot(self.Xsquared[cl].transpose(0, -1, 1), np.sum(pi_DTW, axis=1)).sum(-1)
                    C2 = np.dot(self.Ysquared.transpose(0, -1, 1), np.sum(pi_DTW.T, axis=1)).sum(-1)
                    C3 = np.tensordot(np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), pi_DTW), self.Y,
                                      axes=([1, 2], [2, 1]))
                    res = C1[:, None] + C2[None, :] - 2 * C3
                elif self.metric == "l1":
                    C1 = np.dot(self.X[self.tab_idx[cl]].transpose(0, -1, 1), pi_DTW.T)
                    C2 = np.dot(self.Y.transpose(0, -1, 1), pi_DTW.T)
                    C3 = C1[:, None, :] - C2[None, :, :]
                    res = np.absolute(C3).sum(-1).sum(-1)
                mat_cost[self.tab_idx[cl]] = res
        return mat_cost

    def main_training(self, GW=False, COOT=False):
        if GW:
            d1 = sp.spatial.distance.cdist(self.X.reshape(self.shapeX[0], -1), self.X.reshape(self.shapeX[0], -1))
            d2 = sp.spatial.distance.cdist(self.Y.reshape(self.shapeY[0], -1), self.Y.reshape(self.shapeY[0], -1))
            d1 = d1/d1.max()
            d2 = d2/d2.max()
            self.OT_tilde = ot.gromov.gromov_wasserstein(d1, d2, self.Xa_one, self.Ya_one, "square_loss")
            Cost_OT = 0
            score_OT = 0
        elif COOT:
            self.OT_tilde, features_align, score = cot.cot_numpy(self.X.reshape(self.shapeX[0], -1),
                                                                 self.Y.reshape(self.shapeY[0], -1))
            Cost_OT = 0
            score_OT = 0
        else:
            Cost_OT = self.mat_cost_OT()
            self.OT_tilde = ot.emd(self.Xa_one, self.Ya_one, Cost_OT, numItermax=10000000)
            score_OT = np.sum(self.OT_tilde * Cost_OT)

        return self.OT_tilde, Cost_OT, score_OT


class CNN_class():
    def __init__(self, X_train, y_train, batchsize, epoch):
        self.batchsize = batchsize
        self.shape = X_train.shape
        self.epoch = epoch
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = len(np.unique(self.y_train))
        self.model = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                                                                 use_bias=False,
                                                                 input_shape=(self.shape[1], self.shape[2])),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Activation("relu"),
                                          tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding="same",
                                                                 use_bias=False),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Activation("relu"),
                                          tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
                                                                 use_bias=False),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Activation("relu"),
                                          tf.keras.layers.GlobalAveragePooling1D(),
                                          tf.keras.layers.Dense(self.num_classes, activation='softmax')])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.model.summary()

    def train(self):
        return self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batchsize, verbose=False)

    @staticmethod
    def get_warp_matrices(DTW_mat):

        num_path = int(DTW_mat.sum())
        Mx = np.zeros((num_path, DTW_mat.shape[0]))
        My = np.zeros((num_path, DTW_mat.shape[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == DTW_mat.shape[0] - 1) & (l != DTW_mat.shape[1] - 1):
                arg1 = -1
                arg2 = DTW_mat[k, l + 1]
                arg3 = -1
            if (l == DTW_mat.shape[1] - 1) & (k != DTW_mat.shape[0] - 1):
                arg1 = DTW_mat[k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != DTW_mat.shape[1] - 1) & (k != DTW_mat.shape[0] - 1):
                arg1 = DTW_mat[k + 1, l]
                arg2 = DTW_mat[k, l + 1]
                arg3 = DTW_mat[k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def evaluate(self, X_test, y_test, DTW=None, take_classe=False, y_train=None, OT=None):
        if take_classe is False:
            m1, m2 = self.get_warp_matrices(DTW[0])
            testX_transform = np.dot(X_test.transpose(0, -1, 1), m2.T).transpose(0, -1, 1)
            other, accuracy = self.model.evaluate(testX_transform, y_test, batch_size=self.batchsize, verbose=0)
        else:
            y_pred = np.zeros(shape=(X_test.shape[0], self.num_classes))
            all_num_path = []
            for cl in np.unique(y_train):
                all_num_path.append(int(DTW[cl].sum()))
            max_num_path = np.max(all_num_path)
            for cl in np.unique(y_train):
                m1, m2 = self.get_warp_matrices(DTW[cl])
                partial_OT = OT[np.where(y_train == cl)]
                indexing_train, indexing_test = np.nonzero(partial_OT)
                X_test_cl = np.dot(X_test[np.unique(indexing_test)].transpose(0, -1, 1), m2.T)
                X_test_dtw = np.zeros(shape=(len(np.unique(indexing_test)), max_num_path, X_test.shape[-1]))
                X_test_dtw[:, :all_num_path[cl]] = X_test_cl.transpose(0, -1, 1)
                pred_cl = self.predict(X_test_dtw)
                for j in range(0, len(np.unique(indexing_test))):
                    y_pred[indexing_test[j]] = y_pred[indexing_test[j]] + partial_OT[:, j].sum() * pred_cl[j]
                y_acc = np.argmax(y_pred, axis=1)
                accuracy = np.mean(y_test == y_acc)
        return accuracy

    def evaluate_no_DTW(self, X_test, y_test):
        other, accuracy = self.model.evaluate(X_test, y_test, batch_size=self.batchsize, verbose=0)
        return accuracy

    def predict(self, testX):
        predictions = self.model.predict(testX)
        return predictions


if __name__ == "__main__":

    har_indiv = [[2, 11], [7, 13], [12, 16], [12, 18], [9, 18], [14, 19], [18, 23], [6, 23], [7, 24], [17, 25]]
    hhar_indiv = [[1, 3], [3, 5], [4, 5], [0, 6], [1, 6], [4, 6], [5, 6], [2, 7], [3, 8], [5, 8]]
    wisdm_indiv = [[1, 11], [3, 11], [4, 15], [2, 25], [25, 29], [7, 30], [21, 31], [2, 32], [1, 7], [0, 8]]
    wisdm_class = [[2, 25], [25, 29], [2, 32], [0, 8]]
    uwave_indiv = [[2, 5], [3, 5], [4, 5], [2, 6], [1, 7], [2, 7], [3, 7], [1, 8], [4, 8], [7, 8]]

    def exp_launching(dataset_name, indiv_pairs, cnn=True, reg=True, classe=True, WS=True, OTb=False):
        results = []
        for indiv_pair in indiv_pairs:
            train, _ = load(dataset_name + '_' + str(indiv_pair[0]))
            test, _ = load(dataset_name + '_' + str(indiv_pair[1]))
            X_train = train.train_data
            y_train = train.train_labels.astype(int)
            X_test = test.train_data
            y_test = test.train_labels.astype(int)
            if WS:
                weight_x = weighting(y_train, y_test)
            else:
                weight_x = None
            if classe:
                y_train_OTDTW = y_train
            else:
                y_train_OTDTW = None
            if OTb:
                OTc = OTonly(X_train, X_test, classe=None, laplace_reg=False)
                OT, cost, score = OTc.main_training()
                time_mean = 0
                time_std = 0
                iteration = 0
            else:
                OTDTW_bary = OTDTW(X_train, X_test, classe=y_train_OTDTW, weights_X=weight_x, laplace_reg=reg)
                OT, DTW, cost, Score, time_master, iteration = OTDTW_bary.main_training(barycenter=False)
                time_mean = np.mean(time_master)
                time_std = np.std(time_master)
            if cnn:
                if classe:
                    OT_classe = OT
                else:
                    OT_classe=None
                """model_cnn = CNN_class(X_train=transfer_X, y_train=y_train, batchsize=32, epoch=10000)
                model_cnn.train()
                accuracy = model_cnn.evaluate(X_test, y_test, DTW, take_classe=classe, y_train=y_train_OTDTW,
                                              OT=OT_classe)"""
            else:
                yt_onehot = to_onehot(y_train.astype(int))
                y_pred = np.argmax(np.dot(OT.T, yt_onehot), axis=1)
                accuracy = np.mean(y_pred == y_test.astype(int))
            results.append([indiv_pair[0], indiv_pair[1], accuracy, time_mean, time_std, iteration])
        name = dataset_name + '_OTDTW'
        if cnn:
            name = name + '_cnn'
        if classe:
            name = name + '-class'
        if reg:
            name = name + '_reg'
        if WS:
            name = name + '_WS'
        with open(name + 'OT_test.csv', 'w') as file:
            writer = csv.writer(file, delimiter=' ', quotechar='"')
            writer.writerows([["Indiv source", "Indiv target", "Accuracy", "time mean", "time std", "iteration"]])
            writer.writerows(results)

    def get_warp_matrices(DTW_mat):

        num_path = int(DTW_mat.sum())
        Mx = np.zeros((num_path, DTW_mat.shape[0]))
        My = np.zeros((num_path, DTW_mat.shape[1]))

        k = 0
        l = 0
        for j in range(0, num_path):
            Mx[j, k] = 1
            My[j, l] = 1
            if (k == DTW_mat.shape[0] - 1) & (l != DTW_mat.shape[1] - 1):
                arg1 = -1
                arg2 = DTW_mat[k, l + 1]
                arg3 = -1
            if (l == DTW_mat.shape[1] - 1) & (k != DTW_mat.shape[0] - 1):
                arg1 = DTW_mat[k + 1, l]
                arg2 = -1
                arg3 = -1
            if (l != DTW_mat.shape[1] - 1) & (k != DTW_mat.shape[0] - 1):
                arg1 = DTW_mat[k + 1, l]
                arg2 = DTW_mat[k, l + 1]
                arg3 = DTW_mat[k + 1, l + 1]

            pos_move = np.argmax((arg1, arg2, arg3))
            if pos_move == 0:
                k = k + 1
            if pos_move == 1:
                l = l + 1
            if pos_move == 2:
                l = l + 1
                k = k + 1
        return Mx, My

    def weighting(train_labels, test_labels):
        train_weights = np.empty(shape=(len(train_labels)))
        for cl in np.unique(test_labels):
            weight = np.where(test_labels == cl)[0].shape[0] / (np.where(train_labels == cl)[0].shape[0] * len(test_labels))
            train_weights[np.where(train_labels == cl)] = weight
        return train_weights

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]

    for pair in [[2, 11], [7, 13], [12, 16], [12, 18], [9, 18], [18, 23], [6, 23], [7, 24], [17, 25]]:
        source = np.load("/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/ucihar_" + str(pair[0]) +
                         "train.npy")
        source_label = np.load("/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/ucihar_" + str(pair[0]) +
                               "train_labels.npy")
        target = np.load("/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/ucihar_" + str(pair[1]) +
                         "train.npy")
        target_label = np.load("/home/adr2.local/painblanc_f/codats-master/datas/numpy_data/ucihar_" + str(pair[1]) +
                               "train_labels.npy")
        MAD = OTDTW(X=source, Y=target, classe=source_label, metric="l1")
        OT, DTW, Cost = MAD.main_training()
        yt_onehot = to_onehot(source_label.astype(int))
        y_pred = np.argmax(np.dot(OT.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == target_label.astype(int))
        print(pair, accuracy)

