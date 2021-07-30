import numpy as np
import matplotlib.pyplot as plt
# NIR b8 and Red b4

tarn = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_features_b.npy")
tarn_labels = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_label_b.npy')
bzh = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_features_b.npy")
bzh_labels = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_label_b.npy")

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
class_name = ["Wheat", "Corn", "Barley", "Permanent \nMeadow", "Rape", "Temporary \nMeadow"]
for i in range(0, 6):
    NDVI_bzh_mean[i] = np.mean(NDVI_bzh[bzh_labels == i], axis=0)
    NDVI_tarn_mean[i] = np.mean(NDVI_tarn[tarn_labels == i], axis=0)
    NDVI_bzh_min[i] = np.quantile(NDVI_bzh[bzh_labels == i], 0.25, axis=0)
    NDVI_tarn_min[i] = np.quantile(NDVI_tarn[tarn_labels == i], 0.25,  axis=0)
    NDVI_bzh_max[i] = np.quantile(NDVI_bzh[bzh_labels == i],  0.75, axis=0)
    NDVI_tarn_max[i] = np.quantile(NDVI_tarn[tarn_labels == i], 0.75, axis=0)
    plt.subplot(211)
    plt.plot(NDVI_tarn_mean[i], marker="+")
    plt.fill_between(np.arange(72), NDVI_tarn_min[i], NDVI_tarn_max[i], alpha=0.2)
    plt.ylim([0, 1])
    plt.subplot(212)
    plt.plot(NDVI_bzh_mean[i], marker="+", label=class_name[i])
    plt.fill_between(np.arange(58), NDVI_bzh_min[i], NDVI_bzh_max[i], alpha=0.2)
    plt.ylim([0, 1])
plt.legend(bbox_to_anchor=(1.1, 1.4))
# plt.scatter(np.arange(1412), NDVI_bzh_mean, c=bzh_labels, marker="o")
plt.suptitle('Average of NDVI for Tarn (top) and Brittany (bottom)')
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
plt.suptitle('NDVI max and min of Tarn (top) and Brittany (bottom)')
# plt.scatter(np.arange(1412), NDVI_bzh_mean, c=bzh_labels, marker="o")
# plt.show()
