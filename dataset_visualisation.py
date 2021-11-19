import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from celluloid import Camera

"""loss = np.load('rep_MAD/HAR28000loss.npy')
loss_mad = np.load("rep_MAD/HAR28000loss_mad.npy")
accu = [17.647, 80.392, 78.431, 82.353, 60.784, 54.902, 56.863, 60.784]
iter = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]

fig,ax = plt.subplots()
# make a plot
ax.plot(loss, color="red", label="Classification loss")
ax.plot(loss_mad, color="green", label="MAD loss")
ax.set_xlabel("iteration")
ax.set_ylabel("Value of the loss")
plt.legend()
ax2 = ax.twinx()

# make a plot with different y-axis using second axis object
ax2.plot(iter, accu, color="blue", marker="o", label="accuracy")
ax2.set_ylabel('Accuracy')
plt.legend()
plt.show()"""

loss_train = np.load("1e-050.01loss_train.npy")
loss_valid_source = np.load("1e-050.01loss_valid_source.npy")
loss_valid_target = np.load('1e-050.01loss_valid_target.npy')

plt.plot(loss_valid_target, label="Validation target")
plt.plot(loss_valid_source, label="Validation source")
plt.legend()
plt.show()
plt.clf()

plt.plot(loss_train)
plt.show()


loss_train = np.load("0.00011e-05loss_train.npy")
loss_valid_source = np.load("0.00011e-05loss_valid_source.npy")
loss_valid_target = np.load('0.00011e-05loss_valid_target.npy')

plt.plot(loss_valid_target, label="Validation target")
plt.plot(loss_valid_source, label="Validation source")
plt.legend()
plt.show()
plt.clf()

plt.plot(loss_train)
plt.show()

loss_train = np.load("0.00010.0001loss_train.npy")
loss_valid_source = np.load("0.00010.0001loss_valid_source.npy")
loss_valid_target = np.load('0.00010.0001loss_valid_target.npy')

plt.plot(loss_valid_target, label="Validation target")
plt.plot(loss_valid_source, label="Validation source")
plt.legend()
plt.show()
plt.clf()

plt.plot(loss_train)
plt.show()


def get_warp_matrices(DTW, cl, dimX, dimY):
    num_path = int(DTW[cl].sum())
    Mx = np.zeros((num_path, dimX))
    My = np.zeros((num_path, dimY))
    print(Mx.shape)
    k = 0
    l = 0
    for j in range(0, num_path):
        Mx[j, k] = 1
        My[j, l] = 1
        if (k == dimX - 1) & (l != dimY - 1):
            arg1 = -1
            arg2 = DTW[cl][k, l + 1]
            arg3 = -1
        if (l == dimY - 1) & (k != dimX - 1):
            arg1 = DTW[cl][k + 1, l]
            arg2 = -1
            arg3 = -1
        if (l != dimY - 1) & (k != dimX - 1):
            arg1 = DTW[cl][k + 1, l]
            arg2 = DTW[cl][k, l + 1]
            arg3 = DTW[cl][k + 1, l + 1]

        pos_move = np.argmax((arg1, arg2, arg3))
        if pos_move == 0:
            k = k + 1
        if pos_move == 1:
            l = l + 1
        if pos_move == 2:
            l = l + 1
            k = k + 1
    return Mx, My


"""from tslearn.datasets import CachedDatasets
Xs_1, y, _, _ = CachedDatasets().load_dataset("Trace")
y = y - 1  # everything between 0 and 1
Xt = Xs_1.copy()
y_target = y.copy()
Xs = np.empty(shape=(100, 315, 1))
Xt = np.empty(shape=(100, 315, 1))
Xs[:, :275] = Xs_1[:]
for t in range(0, 40):
    Xs[:, 275 + t] = Xs_1[:, 274]
for i in range(0, 4):
    shift = (i + 1) * 10
    anti_shift = 40 - shift
    print(shift, anti_shift, i, np.unique(y_target))
    Xt[np.where(y == i), shift:315-anti_shift] = Xs_1[np.where(y == i)]
    for j in range(0, shift):
        Xt[np.where(y_target == i), j] = Xs_1[np.where(y_target == i), 0]
    for p in range(0, anti_shift):
        Xt[np.where(y_target == i), 314 - p] = Xs_1[np.where(y_target == i), 274]

for u in range(3, 4):
    plt.subplot(211)
    plt.plot(Xs[np.where(y_target == u)[0][u]])
    plt.title("A series from the source domain")
    plt.subplot(212)
    plt.plot(Xt[np.where(y_target == u)[0][u]])
    plt.title('The same series from the target domain')
    plt.tight_layout()
    plt.show()"""

"""source = np.load("Trace/one_class/one_classe_train500_rout_conv.npy")
source_label = np.load('Trace/one_class/one_classe_train500_target.npy')

target = np.load('Trace/one_class/one_classe_test_rout_conv.npy')
target_label = np.load("Trace/one_class/one_classe_test_target.npy")"""


def to_onehot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


from sklearn.metrics.pairwise import euclidean_distances
fig = plt.figure()
camera = Camera(fig)
for t in range(0, 11):
    if t == 0:
        i = 1
    else:
        i = t * 100
    source_target = np.load("rep_MAD/correct/3Trace_classif" + str(i) + "_best_model__OTcost.npy")
    source_conv = np.load("rep_MAD/correct/3Trace_classifsource" + str(i) + "_valid_rout_conv.npy")
    target_conv = np.load("rep_MAD/correct/3Trace_classiftarget" + str(i) + "_valid_rout_conv.npy")
    source_label = np.load("rep_MAD/correct/3Trace_classifsource" + str(i) + "_valid_target.npy")
    target_label = np.load("rep_MAD/correct/3Trace_classiftarget" + str(i) + "_valid_target.npy")

    """target_pred = np.load("rep_MAD/HAR/HARtarget" + str(i) + "_valid_prediction.npy")
    for j in range(0, target_label.shape[0]):
        if target_label[j] != target_pred[j]:
            target_label[j] = 6
    print(sum(target_label == 6)/target_label.shape[0])"""
    DTW = np.load("rep_MAD/correct/3Trace_classif" + str(i) + "_best_model__DTW.npy")
    source_dist = euclidean_distances(source_conv.reshape(source_conv.shape[0], -1), squared=True)
    target_dist = euclidean_distances(target_conv.reshape(target_conv.shape[0], -1), squared=True)
    Cost1 = np.concatenate((source_dist, source_target), axis=1)
    Cost2 = np.concatenate((source_target.T, target_dist), axis=1)
    Cost = np.concatenate((Cost1, Cost2), axis=0)
    transform = MDS(n_components=2, random_state=1, dissimilarity="precomputed").fit_transform(Cost)
    color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple', "yellow"]
    for c in range(0, 4):
        legend_element = [plt.scatter([0], [0], marker='+', c='yellow', label='Wrongly classified'),
                          plt.scatter([0], [0], marker='+', c='blue', label='target MAD'),
                          plt.scatter([0], [0], marker='o', c='blue', label='source MAD')
                          ]
        plt.scatter(transform[np.where(source_label == c), 0], transform[np.where(source_label == c), 1],
                    marker='o', label="source MAD", c=color_index[c])
        plt.scatter(transform[np.where(target_label == c), 0], transform[np.where(target_label == c), 1],
                    marker='+', label="target MAD", c=color_index[c])
        if ((c == 0) & (t == 1)) or ((c == 7) & (t == 1)):
            plt.legend(handles=legend_element)

    epoch = "Epochs " + str(i)
    # plt.text(1000000, 1000000, epoch)
    camera.snap()
    plt.show()
    plt.clf()
    plt.subplot(221)
    plt.imshow(DTW[0])
    plt.title("DTW for class 1")
    plt.subplot(222)
    plt.imshow(DTW[1])
    plt.title("DTW for class 2")
    plt.subplot(223)
    plt.imshow(DTW[2])
    plt.title("DTW for class 3")
    plt.subplot(224)
    plt.imshow(DTW[3])
    plt.tight_layout()
    plt.show()
    plt.clf()

animation = camera.animate(interval=1800)
animation.save("Trace_classif1" + '.gif')

# DTW = np.load('rep_MAD/Trace_shift/Trace_shift10_DTW.npy')
DTW = np.load('rep_MAD/ep10/10L_Trace_shifted10_DTW.npy')
for i in range(0, 4):
    plt.imshow(DTW[i])
    # plt.show()

DTW = np.load('rep_MAD/ep10/10Trace_shift10_DTW.npy')
for i in range(0, 4):
    plt.imshow(DTW[i])
    # plt.show()

DTW = np.load('rep_MAD/ep10/10_Trace_halfshifted10_DTW.npy')
for i in range(0, 4):
    plt.imshow(DTW[i])
    # plt.show()

DTW = np.load('rep_MAD/HARMAD/HAR_MAD10_DTW.npy')
plt.subplot(321)
plt.imshow(DTW[0])
plt.title("DTW for class 1")
plt.subplot(322)
plt.imshow(DTW[1])
plt.title("DTW for class 2")
plt.subplot(323)
plt.imshow(DTW[2])
plt.title("DTW for class 3")
plt.subplot(324)
plt.imshow(DTW[3])
"""plt.title("DTW for class 4")
plt.subplot(325)
plt.imshow(DTW[4])
plt.title("DTW for class 5")
plt.subplot(326)
plt.imshow(DTW[5])
plt.title("DTW for class 6")
plt.tight_layout()
plt.title("DTW for Trace dataset")"""
# plt.show()

DTW = np.load('rep_MAD/HAR/HAR10_DTW.npy')
plt.subplot(321)
plt.imshow(DTW[0])
plt.title("DTW for class 1")
plt.subplot(322)
plt.imshow(DTW[1])
plt.title("DTW for class 2")
plt.subplot(323)
plt.imshow(DTW[2])
plt.title("DTW for class 3")
plt.subplot(324)
plt.imshow(DTW[3])
plt.title("DTW for class 4")
plt.subplot(325)
plt.imshow(DTW[4])
plt.title("DTW for class 5")
plt.subplot(326)
plt.imshow(DTW[5])
plt.title("DTW for class 6")
plt.tight_layout()
plt.title("DTW for Trace dataset")
# plt.show()

for i in range(0, 4):
    plt.imshow(DTW[i])
    # plt.show()

source = np.load("small_CNN_all_class_train100_rout_conv.npy")
source_label = np.load('small_CNN_all_class_train100_target.npy')

target = np.load('small_CNN_all_class_test100_rout_conv.npy')
target_label = np.load("small_CNN_all_class_test100_target.npy")
all_data = np.concatenate((source.reshape(100, -1), target.reshape(100, -1)), 0)
# all_data = np.concatenate((source, target), 0)

transform = MDS(n_components=2, random_state=1).fit_transform(all_data)
transform_source = transform[:source.shape[0]]
transform_target = transform[source.shape[0]:]

color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
for i in range(0, 4):
    plt.scatter(transform_source[np.where(source_label == i), 0], transform_source[np.where(source_label == i), 1],
                marker='o', label="source MAD", c=color_index[i])
    plt.scatter(transform_target[np.where(target_label == i), 0], transform_target[np.where(target_label == i), 1],
                marker='+', label="target MAD", c=color_index[i])
    if i == 0:
        plt.legend()
# plt.show()
plt.clf()
for i in range(0, 4):
    mX, mY = get_warp_matrices(DTW, i, 275, 275)
    source_class = np.dot(source[np.where(source_label == i)], mX.T)
    target_class = np.dot(target[np.where(target_label == i)], mY.T)
    # target_label = np.load('No_classif/MAD_no_classif_test_target.npy')
    all_data = np.concatenate((source_class.reshape(source_class.shape[0], -1),
                               target_class.reshape(target_class.shape[0], -1)), 0)
    # all_data = np.concatenate((source, target), 0)
    transform = MDS(n_components=2, random_state=1).fit_transform(all_data)
    transform_source = transform[:source_class.shape[0]]
    transform_target = transform[source_class.shape[0]:]
    color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
    plt.scatter(transform_source[:, 0], transform_source[:, 1],
                marker='o', label="source MAD", c=color_index[3])
    plt.scatter(transform_target[:, 0], transform_target[:, 1],
                marker='+', label="target MAD", c=color_index[3])
    plt.show()
