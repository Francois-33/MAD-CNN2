import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

""""tarn = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_short_norm.npy')
bzh = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_short_norm.npy')
tarn_label = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_short_norm_label.npy")
bzh_label = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_short_norm_label.npy")
tarn_transform = TSNE(n_components=2, random_state=0).fit_transform(tarn.reshape((500, -1)))
bzh_transform = TSNE(n_components=2, random_state=0).fit_transform(bzh.reshape((400, -1)))

index_color = ["blue", "orange", "red", "green", "yellow", "black"]
plt.subplot(121)
plt.scatter(tarn_transform[:, 0], tarn_transform[:, 1], marker="o", c=tarn_label, label="Tarn",
            cmap=plt.get_cmap("tab10"))
plt.subplot(122)
plt.scatter(bzh_transform[:, 0], bzh_transform[:, 1], marker="o", c=bzh_label, label="Brittany",
            cmap=plt.get_cmap("tab10"))
# plt.legend()
plt.show()"""
# plt.savefig('tarn_bzh_TSNE.png')

source = np.load("MAD_two_losses_train_out_conv.npy")
# source = np.mean(source, axis=-1)
source_label = np.load('MAD_two_losses_train_target.npy')
source_pred = np.load('MAD_two_losses_train_prediction.npy')
target = np.load('MAD_two_losses_test_out_conv.npy')
# target = np.mean(target, axis=-1)
target_label = np.load('MAD_two_losses_test_target.npy')
target_pred = np.load('MAD_two_losses_test_prediction.npy')
all_data = np.concatenate((source, target), 0)
pca_all_data = PCA(n_components=30, random_state=0).fit_transform(all_data)
transform = TSNE(n_components=2, random_state=1).fit_transform(pca_all_data)
# transform = TSNE(n_components=2, random_state=1).fit_transform(all_data)

sourcent = np.load('CNN_notransfersource.npy')
sourcent_label = np.load('CNN_notransfersource_label.npy')
sourcent_pred = np.load('CNN_notransferTrueprediction.npy')
targetnt = np.load('CNN_notransfertarget.npy')
targetnt_label = np.load('CNN_notransfertarget_label.npy')
targetnt_pred = np.load('CNN_notransferFalseprediction.npy')
all_datant = np.concatenate((sourcent, targetnt), 0)
pca_all_datant = PCA(n_components=30, random_state=0).fit_transform(all_datant)
transformnt = TSNE(n_components=2, random_state=1).fit_transform(pca_all_datant)

# plt.subplot(121)
plt.scatter(transform[:source.shape[0], 0], transform[:source.shape[0], 1], marker='o', label="source MAD",
            c=source_label, cmap=plt.get_cmap("tab10"))
plt.scatter(transform[source.shape[0]:, 0], transform[source.shape[0]:, 1], marker='+', label="target MAD",
            c=target_label, cmap=plt.get_cmap("tab10"))
# plt.scatter(transform[500:, 0][target_pred.squeeze() == target_label], transform[500:, 1][target_pred.squeeze() == target_label], marker='s', s=20, c="red")
plt.legend()
"""plt.subplot(122)
plt.scatter(transformnt[:500, 0], transformnt[:500, 1], marker='o', c=sourcent_label, label="source no transport", cmap=plt.get_cmap("tab10"))
plt.scatter(transformnt[500:, 0], transformnt[500:, 1], marker='+', c=targetnt_label, label="target no transport", cmap=plt.get_cmap("tab10"))
plt.scatter(transformnt[500:, 0][targetnt_pred.squeeze() == targetnt_label], transformnt[500:, 1][targetnt_pred.squeeze() == targetnt_label],  marker='s', s=20, c="red")
plt.legend()"""
plt.title('Latent representation of the train and \nsource dataset of MAD with half classification')
plt.show()

conf_mat_MAD_source = np.load('CNN_MADTrueconfusion_mat.npy')
conf_mat_MAD_target = np.load('CNN_MADFalseconfusion_mat.npy')
conf_mat_nt_source = np.load('CNN_notransferTrueconfusion_mat.npy')
conf_mat_nt_target = np.load('CNN_notransferFalseconfusion_mat.npy')
print(conf_mat_MAD_target)
print(conf_mat_nt_target)

# loss_MAD = np.load('~/Desktop/MAD-CNN-master/MAD_CNN10000.csv')
"""loss_MAD = pd.read_csv('/home/adr2.local/painblanc_f/Desktop/MAD-CNN-master/Codats_dom10000.csv', delimiter="\t",
                       header=0)
# plt.plot(loss_MAD.iloc[2])
plt.plot(loss_MAD.iloc[1])"""

# loss = np.load('/home/adr2.local/painblanc_f/Desktop/MAD-CNN-master/ucihar/MAD_CNN7000loss.npy')
# loss_mad = np.load('/home/adr2.local/painblanc_f/Desktop/MAD-CNN-master/ucihar/MAD_CNN7000loss_mad.npy')

"""loss_cor = np.empty(shape=(10000, 1))
for i in range(0, 10000):
    loss_cor[i] = loss_dom[i*5: (i+1)*5].sum()
loss_cor_first = loss_cor[9500:]
# plt.plot(loss_cor_first)"""

"""plt.plot(loss, label='Classification Loss')
plt.plot(loss_mad, label='MAD Loss')
plt.title("Train MADCNN on UCIHAR-14-19 over 7000 epochs, alpha=lambda=0.001")
plt.legend()
plt.show()"""
