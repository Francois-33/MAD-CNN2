import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

source = np.load("Trace/trace_train_1_rout_conv.npy")
source_label = np.load('Trace/trace_train_1_target.npy')

target = np.load('Trace/trace_test_1_rout_conv.npy')
target_label = np.load("Trace/trace_test_1_target.npy")
# target_label = np.load('No_classif/MAD_no_classif_test_target.npy')
print(source.shape, target.shape)
all_data = np.concatenate((source.reshape(100, -1), target.reshape(100, -1)), 0)
"""pca_all_data = PCA(n_components=30, random_state=0).fit_transform(all_data)
transform = TSNE(n_components=2, random_state=1).fit_transform(pca_all_data)"""
transform = TSNE(n_components=2, random_state=1).fit_transform(all_data)
transform_source = transform[:source.shape[0]]
transform_target = transform[source.shape[0]:]

color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
for i in range(0, 6):
    plt.scatter(transform_source[np.where(source_label == i), 0], transform_source[np.where(source_label == i), 1],
                marker='o', label="source MAD", c=color_index[i])
    plt.scatter(transform_target[np.where(target_label == i), 0], transform_target[np.where(target_label == i), 1],
                marker='+', label="target MAD", c=color_index[i])
    if i == 0:
        plt.legend()
plt.title('Mean of the latent representation of the train and \nsource datasets on MAD-CNN')
# plt.show()
"""
    plt.scatter(transform[:source.shape[0], 0], transform[:source.shape[0], 1], marker='o', label="source MAD",
                c=source_label, cmap=plt.get_cmap("tab10"))
    plt.scatter(transform[source.shape[0]:, 0], transform[source.shape[0]:, 1], marker='+', label="target MAD",
                c=target_label, cmap=plt.get_cmap("tab10"))"""
loss = np.load('/home/adr2.local/painblanc_f/Desktop/MAD-CNN-master/Trace/trace1000loss.npy')
loss_mad = np.load('/home/adr2.local/painblanc_f/Desktop/MAD-CNN-master/Trace/trace1000loss_mad.npy')

plt.plot(loss, label='Classification Loss')
plt.plot(loss_mad, label='MAD Loss')
plt.title("Train MADCNN on UCIHAR-14-19 over 7000 epochs, alpha=lambda=0.001")
plt.legend()
# plt.show()


def to_onehot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


OT = np.load("Trace/trace1_OT.npy")
yt_onehot = to_onehot(source_label.astype(int))
y_pred = np.argmax(np.dot(OT.T, yt_onehot), axis=1)
accuracy = np.mean(y_pred == target_label.astype(int))
print(accuracy)

OT_dist = np.load('Trace/trace1_OTcost.npy')

dist_mat = np.zeros(shape=(4, 4))
for s in np.unique(source_label):
    for t in np.unique(target_label):
        OT1 = OT_dist[np.where(source_label == s)[0]]
        OT2 = OT1[:, np.where(target_label == t)[0]]
        # print(OT_dist[np.where(source_label == s)[0], np.where(target_label == t)[0]])
        # print(OT_dist[np.where(source_label == s), np.where(target_label == t)].sum())
        dist_mat[s, t] = OT2.mean()  # OT_dist[np.where(source_label == s), np.where(target_label == t)].sum()

plt.clf()
plt.imshow(dist_mat)
# plt.show()

