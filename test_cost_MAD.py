import numpy.random as npr
import torch
import torch.nn as nn
import numpy.random as npr
import ot
import tslearn.metrics as tslm
from torch.utils.data import Dataset
import numpy as np
import warnings
from CNN_class_balanced import CNNMAD


series0_source0 = torch.zeros(size=(100, 20, 1)).type(torch.float)
series0_source1 = torch.ones(size=(100, 80, 1)).type(torch.float)
series0_source = torch.cat((series0_source0, series0_source1), dim=1)
series1_source0 = torch.zeros(size=(100, 80, 1)).type(torch.float)
series1_source1 = torch.ones(size=(100, 20, 1)).type(torch.float)
series1_source = torch.cat((series1_source0, series1_source1), dim=1)
series_source = torch.cat((series0_source, series1_source), dim=0)
series0_target0 = torch.zeros(size=(100, 40, 1)).type(torch.float)
series0_target1 = torch.ones(size=(100, 60, 1)).type(torch.float)
series0_target = torch.cat((series0_target0, series0_target1), dim=1)
series1_target0 = torch.zeros(size=(100, 60, 1)).type(torch.float)
series1_target1 = torch.ones(size=(100, 40, 1)).type(torch.float)
series1_target = torch.cat((series1_target0, series1_target1), dim=1)
series_target = torch.cat((series0_target, series1_target), dim=0)

labels0 = torch.zeros(size=(100,)).type(torch.long)
labels1 = torch.ones(size=(100,)).type(torch.long)
labels = torch.cat((labels0, labels1))

cnnmad = CNNMAD(name="", batchsize=10, beta=1.0, alpha=1.0, channel=1, valid_target_label=labels,
                train_target_data=series_target, test_target_data=series_target, test_target_label=labels,
                test_source_label=labels, test_source_data=series_source, train_source_data=series_source,
                train_source_label=labels, valid_source_data=series_source, valid_target_data=series_target ,
                num_class=2, valid_source_label=labels)

cnnmad.set_current_batchsize(200)
cnnmad.set_current_batchsize(200, train=False)
out_source, out_conv_source = cnnmad.forward(cnnmad.trainSourceData.transpose(1, 2))
out_target, out_conv_target = cnnmad.forward(cnnmad.trainTargetData.transpose(1, 2), train=False)
cnnmad.mad(out_conv_source=out_conv_source, out_conv_target=out_conv_target, labels=labels)
DTW = cnnmad.DTW
OT = cnnmad._OT

#  Torch version
source_sq = out_conv_source ** 2
print('source sum', source_sq.sum().item())
target_sq = out_conv_target ** 2
res = torch.zeros(size=(200, 200))
res2 = torch.zeros(size=(200, 200))
out_CNN_before = torch.tensor(OT) * res
for cl in range(0, 2):
      idx_cl = torch.where(labels == cl)
      pi_DTW = DTW[cl]
      pi_DTW = torch.tensor(pi_DTW).float()
      C1 = torch.matmul(source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
      print(source_sq[idx_cl].shape, torch.sum(pi_DTW, dim=1).shape)
      C1bis = torch.mul(source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
      print(C1.shape, C1bis.shape, C1.sum().item(), C1bis.sum().item(), "hehehehehehehe")
      C2 = torch.matmul(target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
      C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target, dims=([1, 2], [1, 2]))
      res[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
      res2[idx_cl] = torch.tensor(OT).type(torch.float)[idx_cl] * (C1[:, None] + C2[None, :] - 2 * C3)
      print((torch.tensor(OT).type(torch.float)[idx_cl] * (C1[:, None] + C2[None, :])).sum().item(),
            (torch.tensor(OT).type(torch.float)[idx_cl] * (2 * C3)).sum().item())
out_CNN = torch.tensor(OT) * res

# Numpy version
out_np_source = out_conv_source.detach().numpy()
out_np_target = out_conv_target.detach().numpy()
np_source_sq = out_np_source ** 2
np_target_sq = out_np_target ** 2
print("sq numpy", np_source_sq.sum())
res_np = np.zeros(shape=(200, 200))
res_np2 = np.zeros(shape=(200, 200))
np_out_CNN_before = OT * res_np
for cl in range(0, 2):
      pi_DTW = DTW[cl]
      idx_cl = np.where(labels == cl)
      C1 = np.dot(np_source_sq[idx_cl], np.sum(pi_DTW, axis=1)).sum(-1)
      print(C1.shape, C1.sum(), 'nununununu')
      C2 = np.dot(np_target_sq, np.sum(pi_DTW.T, axis=1)).sum(-1)
      C3 = np.tensordot(np.dot(out_np_source[idx_cl], pi_DTW), out_np_target.transpose(0, -1, 1), axes=([1, 2], [2, 1]))
      res_np[idx_cl] = C1[:, None] + C2[None, :] - 2 * C3
      res_np2[idx_cl] = OT[idx_cl] * (C1[:, None] + C2[None, :] - 2 * C3)
      print((OT[idx_cl] * (C1[:, None] + C2[None, :])).sum().item(),
            (OT[idx_cl] * (2 * C3)).sum().item())
np_out_CNN = OT * res_np

print(OT.sum())
print(np_out_CNN.sum(), out_CNN.sum().item())
print(np_out_CNN_before.sum(), out_CNN_before.sum().item())
print(res_np.sum(), res.sum().item())
print(res_np2.sum(), res2.sum().item())
