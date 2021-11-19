import numpy as np
import matplotlib.pyplot as plt
# NIR b8 and Red b4

tarn = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/old_version/tarn_features_b.npy")
tarn_labels = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/old_version/tarn_label_b.npy')
bzh = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/old_version/bzh_features_b.npy")
bzh_labels = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/old_version/bzh_label_b.npy")

NDVI_tarn_div = (tarn[:, :, 7] + tarn[:, :, 3])
NDVI_tarn_den = (tarn[:, :, 7] - tarn[:, :, 3])
NDVI_bzh_div = (bzh[:, :, 7] + bzh[:, :, 3])
NDVI_bzh_den = (bzh[:, :, 7] - bzh[:, :, 3])

NDVI_tarn = np.zeros(shape=(tarn.shape[0], tarn.shape[1]))
NDVI_bzh = np.zeros(shape=(bzh.shape[0], bzh.shape[1]))

NDVI_tarn[NDVI_tarn_div > 0] = NDVI_tarn_den[NDVI_tarn_div > 0] / NDVI_tarn_div[NDVI_tarn_div > 0]
NDVI_bzh[NDVI_bzh_div > 0] = NDVI_bzh_den[NDVI_bzh_div > 0] / NDVI_bzh_div[NDVI_bzh_div > 0]

NDVI_tarn_mean = np.empty(shape=(6, tarn.shape[1]))
NDVI_bzh_mean = np.empty(shape=(6, bzh.shape[1]))

NDVI_tarn_min = np.empty(shape=(6, tarn.shape[1]))
NDVI_bzh_min = np.empty(shape=(6, bzh.shape[1]))
NDVI_tarn_max = np.empty(shape=(6, tarn.shape[1]))
NDVI_bzh_max = np.empty(shape=(6, bzh.shape[1]))
color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
class_name = ["Wheat", "Corn", "Barley", "Permanent \nMeadow", "Rape", "Temporary \nMeadow"]
plt.figure(figsize=(10, 8))
for i in range(1, 2):
    NDVI_bzh_mean[i] = np.mean(NDVI_bzh[bzh_labels == i], axis=0)
    NDVI_tarn_mean[i] = np.mean(NDVI_tarn[tarn_labels == i], axis=0)
    NDVI_bzh_min[i] = np.quantile(NDVI_bzh[bzh_labels == i], 0.25, axis=0)
    NDVI_tarn_min[i] = np.quantile(NDVI_tarn[tarn_labels == i], 0.25,  axis=0)
    NDVI_bzh_max[i] = np.quantile(NDVI_bzh[bzh_labels == i],  0.75, axis=0)
    NDVI_tarn_max[i] = np.quantile(NDVI_tarn[tarn_labels == i], 0.75, axis=0)
    plt.subplot(211)
    plt.plot(NDVI_tarn_mean[i], marker="+", color=color_index[i])
    plt.fill_between(np.arange(72), NDVI_tarn_min[i], NDVI_tarn_max[i], alpha=0.2, color=color_index[i])
    plt.ylim([0, 1])
    plt.subplot(212)
    plt.plot(NDVI_bzh_mean[i], marker="+", label=class_name[i], color=color_index[i])
    plt.fill_between(np.arange(58), NDVI_bzh_min[i], NDVI_bzh_max[i], alpha=0.2, color=color_index[i])
    plt.ylim([0, 1])
plt.legend(bbox_to_anchor=(1.1, 1.42))
# plt.scatter(np.arange(1412), NDVI_bzh_mean, c=bzh_labels, marker="o")
# plt.suptitle('Average, 2nd and 8th quantile of NDVI for Tarn (top) and Brittany (bottom)')
plt.suptitle('Average, 1st and 3rd quartiles of NDVI for Tarn (top) and Brittany (bottom) for corn only')
plt.show()

for i in range(3, 6):
    NDVI_bzh_mean[i] = np.mean(NDVI_bzh[bzh_labels == i], axis=0)
    NDVI_tarn_mean[i] = np.mean(NDVI_tarn[tarn_labels == i], axis=0)
    NDVI_bzh_min[i] = np.min(NDVI_bzh[bzh_labels == i], axis=0)
    NDVI_tarn_min[i] = np.min(NDVI_tarn[tarn_labels == i], axis=0)
    NDVI_bzh_max[i] = np.max(NDVI_bzh[bzh_labels == i], axis=0)
    NDVI_tarn_max[i] = np.max(NDVI_tarn[tarn_labels == i], axis=0)
    plt.subplot(211)
    plt.plot(NDVI_tarn_mean[i], marker="+")
    plt.fill_between(np.arange(72), NDVI_tarn_min[i], NDVI_tarn_max[i], alpha=0.2)
    plt.ylim([-0.5, 1])
    plt.subplot(212)
    plt.plot(NDVI_bzh_mean[i], marker="+")
    plt.fill_between(np.arange(58), NDVI_bzh_min[i], NDVI_bzh_max[i], alpha=0.2)
    plt.ylim([-0.5, 1])
# plt.scatter(np.arange(1412), NDVI_bzh_mean, c=bzh_labels, marker="o")
plt.suptitle('NDVI mean of Tarn (top) and Brittany (bottom) for the three last classes')
# plt.show()

plt.clf()
color_index = ['blue', 'orange', 'red', 'green', 'pink', 'purple']
for i in range(0, 6):
    plt.subplot(211)
    plt.plot(NDVI_tarn_min[i], marker="+", color=color_index[i])
    plt.plot(NDVI_tarn_max[i], marker="+", color=color_index[i])
    plt.subplot(212)
    plt.plot(NDVI_bzh_min[i], marker="+", color=color_index[i])
    plt.plot(NDVI_bzh_max[i], marker="+", color=color_index[i])
# plt.suptitle('NDVI max and min of Tarn (top) and Brittany (bottom)')
# plt.scatter(np.arange(1412), NDVI_bzh_mean, c=bzh_labels, marker="o")
# plt.show()

from tslearn import metrics
from scipy.spatial.distance import cdist
path, sim = metrics.dtw_path(NDVI_bzh_mean[1], NDVI_tarn_mean[1])
plt.clf()
plt.figure(1, figsize=(15, 8))
# definitions for the axes
left, bottom = 0.01, 0.1
w_ts = 0.06
h_ts = 0.1
left_h = left + w_ts + 0.02
width = 0.45
height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

mat = cdist(NDVI_bzh_mean[1].reshape(58, 1), NDVI_tarn_mean[1].reshape(72, 1))

ax_gram.imshow(mat, origin='lower')
ax_gram.set_xlabel('Tarn')
ax_gram.xaxis.set_label_position('top')
ax_gram.set_ylabel("Brittany")
ax_gram.autoscale(False)
ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
             linewidth=3.)
sz = NDVI_bzh_mean[1].shape[0]

ax_s_x.plot(np.arange(72), NDVI_tarn_mean[1], "b-", linewidth=2., c="orange")
ax_s_x.axis("off")
ax_s_x.set_xlim((0, 71))

ax_s_y.plot(- NDVI_bzh_mean[1], np.arange(sz), "b-", linewidth=2., c="orange")
ax_s_y.axis("off")
ax_s_y.set_ylim((0, sz - 1))

plt.tight_layout()
plt.show()

