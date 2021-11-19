import numpy as np
import seaborn as sns
from celluloid import Camera
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
plt.style.use("seaborn")
"""pred3 = np.load('TarnBZH/TARNBZH_target30000_test_prediction.npy')
targ3 = np.load('TarnBZH/TARNBZH_target30000_test_target.npy')

pred = np.load("TarnBZH/TARNBZH_target15240_test_prediction.npy")
targ = np.load("TarnBZH/TARNBZH_target15240_test_target.npy")

acc3 = np.mean(pred3.squeeze(-1) == targ3)
acc = np.mean(pred.squeeze(-1) == targ)

print(acc3)
print(acc)"""


loss_beta1419 = np.load('Good_one/1414/HARloss_beta.npy', allow_pickle=True)
loss_alpha1419 = np.load('Good_one/1414/HARloss_alpha.npy', allow_pickle=True)
loss_train1419 = np.load('Good_one/1414/HARloss_train.npy')
accuracy_target1419 = np.load('Good_one/1414/HARacc_target.npy')
accuracy_source1419 = np.load('Good_one/1414/HARacc_source.npy')

loss_beta1823 = np.load('Good_one/1823/HAR_2tryloss_beta.npy', allow_pickle=True)
loss_alpha1823 = np.load('Good_one/1823/HAR_2tryloss_alpha.npy', allow_pickle=True)
loss_train1823 = np.load('Good_one/1823/HAR_2tryloss_train.npy')
accuracy_target1823 = np.load('Good_one/1823/HAR_2tryloss_valid_target.npy')
accuracy_source1823 = np.load('Good_one/1823/HAR_2tryloss_valid_source.npy')

"""loss_beta624 = np.load('Good_one/624/HAR_2tryloss_beta.npy', allow_pickle=True)
loss_alpha624 = np.load('Good_one/624/HAR_2tryloss_alpha.npy', allow_pickle=True)
loss_train624 = np.load('Good_one/624/HAR_2tryloss_train.npy')
accuracy_target624 = np.load('Good_one/624/HAR_2tryloss_valid_target.npy')
accuracy_source624 = np.load('Good_one/624/HAR_2tryloss_valid_source.npy')

loss_beta1216 = np.load('Good_one/1216_loss/HAR_noexploss_beta.npy', allow_pickle=True)
loss_alpha1216 = np.load('Good_one/1216_loss/HAR_noexploss_alpha.npy', allow_pickle=True)
loss_train1216 = np.load('Good_one/1216_loss/HAR_noexploss_train.npy')
loss_other1216 = np.load('Good_one/1216_loss/HAR_noexp0loss_CNN_only.npy')
accuracy_target1216 = np.load('Good_one/1216_01/HARacc_target.npy')
accuracy_source1216 = np.load('Good_one/1216_01/HARacc_source.npy')

loss_beta918 = np.load('Good_one/918_10k/HAR_2tryloss_beta.npy', allow_pickle=True)
loss_alpha918 = np.load('Good_one/918_10k/HAR_2tryloss_alpha.npy', allow_pickle=True)
loss_train918 = np.load('Good_one/918_10k/HAR_2tryloss_train.npy')
accuracy_target918 = np.load('Good_one/918_10k/HAR_2tryacc_target.npy')
accuracy_source918 = np.load('Good_one/918_10k/HAR_2tryacc_source.npy')

loss_beta = np.load('Good_one/1725_10k/HAR_2tryloss_beta.npy', allow_pickle=True)
loss_alpha = np.load('Good_one/1725_10k/HAR_2tryloss_alpha.npy', allow_pickle=True)
loss_train = np.load('Good_one/1725_10k/HAR_2tryloss_train.npy')
accuracy_target = np.load('Good_one/1725_10k/HAR_2tryacc_target.npy')
accuracy_source = np.load('Good_one/1725_10k/HAR_2tryacc_source.npy')

loss_beta713 = np.load('Good_one/713/HAR_2tryloss_beta.npy', allow_pickle=True)
loss_alpha713 = np.load('Good_one/713/HAR_2tryloss_alpha.npy', allow_pickle=True)
loss_train713 = np.load('Good_one/713/HAR_2tryloss_train.npy')
accuracy_target713 = np.load('Good_one/713/HAR_2tryacc_target.npy')
accuracy_source713 = np.load('Good_one/713/HAR_2tryacc_source.npy')"""

"""plt.plot(accuracy_target, label="target 1725")
plt.plot(accuracy_source, label="source 1725")
plt.plot(accuracy_target918, label="target 918")
plt.plot(accuracy_source918, label="source 918")"""
plt.plot(accuracy_target1419, label="target 1419")
plt.plot(accuracy_source1419, label="source 1419")
plt.legend()
plt.show()
"""plt.plot(accuracy_target1216, label="target 1216")
plt.plot(accuracy_source1216, label="source 1216")
plt.clf()
# plt.plot(loss_train[:200], label="1725")"""
"""plt.plot(loss_train918[:200], label='918')
plt.plot(loss_train1216[:200], label="Total loss")"""
# plt.plot(loss_train713, label="713")
# plt.plot(loss_train624, label="624")
plt.plot(loss_train1419 - loss_beta1419 - 0.0001 * loss_alpha1419, label="CE loss 1419")
# plt.plot(loss_train1823, label="1823")
# plt.plot(loss_beta[:100], label="beta1725")
# plt.plot(0.1 * loss_beta1216, label="0.1 * Cross entropic similarity")
# plt.plot(0.001 * loss_alpha1216, label='0.001 * MAD loss')
plt.plot(loss_beta1419, label="beta1419")
# plt.plot(loss_alpha1419, label='alpha1419')
plt.legend()
plt.show()
"""plt.title("12-16 pairs, 0.1 beta loss")
plt.legend()
plt.show()

loss_classif = loss_train - 0.00000001 * (loss_alpha + loss_beta)
loss_valid_target = np.load("Good_one/1725_10k/HAR_2tryloss_valid_target.npy")
loss_valid_source = np.load("Good_one/1725_10k/HAR_2tryloss_valid_source.npy")

plt.plot(loss_beta, label="beta")
plt.plot(loss_train, label="train")
plt.plot(0.00000001 * loss_alpha, label="alpha")
plt.legend()
plt.show()

plt.plot(loss_valid_source, label='source')
plt.plot(loss_valid_target, label="target")
plt.legend()
plt.show()

accuracy211 = np.load("Good_one/1725_10k/HAR_2tryacc_target.npy")
accuracy_source = np.load("Good_one/1725_10k/HAR_2tryacc_source.npy")
loss_beta_small = np.empty(shape=(accuracy_source.shape))
for i in range(0, accuracy_source.shape[0]):
    t = i * 20
    loss_beta_small[i] = loss_beta[t]
plt.plot(accuracy211)
# plt.show()
plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(accuracy211)
ax1.plot(accuracy_source)
ax2.plot(loss_beta_small)
plt.show()"""
"""accuracy1725 = np.load("Good_one/shit_model/17_25/HARacc_target.npy")
accuracy918 = np.load("Good_one/shit_model/9_18/HARacc_target.npy")
plt.plot(accuracy1725)"""

# plt.plot(accuracy918)
# plt.show()
plt.clf()
loss_source = np.load("Good_one/shit_model/other12_16/HARloss_valid_source.npy")
loss_target = np.load("Good_one/shit_model/other12_16/HARloss_valid_target.npy")
loss_sources = np.load("Good_one/shit_model/12_16/HARloss_valid_source.npy")
loss_targets = np.load("Good_one/shit_model/12_16/HARloss_valid_target.npy")

plt.plot(loss_source, label="source 200")
plt.plot(loss_target, label='target 200')
plt.plot(loss_targets, label="target 100")
plt.plot(loss_sources, label="source 100")
plt.legend()
# plt.show()

loss_beta200 = np.load('Good_one/shit_model/other12_16/HARloss_beta.npy', allow_pickle=True)
loss_alpha200 = np.load('Good_one/shit_model/other12_16/HARloss_alpha.npy', allow_pickle=True)
loss_train200 = np.load('Good_one/shit_model/other12_16/HARloss_train.npy')
loss_classif200 = loss_train200 - 0.0000001 * (loss_alpha200 + loss_beta200)

plt.plot(loss_beta200, label='beta loss')
plt.plot(0.0000001 * loss_alpha200, label="weighted alpha loss")
plt.plot(loss_classif200, label="Train loss")
plt.title("a = 1e-7 b = 1e-5 iteration 200 HAR")
plt.legend()
plt.tight_layout()
# plt.show()

accuracy7 = np.load("Good_one/10e7/HARacc_target.npy")
accuracy5 = np.load("Good_one/10e5/HARacc_target.npy")
accuracy = np.load("Good_one/HARacc_target.npy")

plt.plot(accuracy, label="a=1e-5 b=1e-4")
plt.plot(accuracy5, label="a=1e-5 b=1e-5")
plt.plot(accuracy7, label="a=1e-7 b=1e-5")
plt.legend()
plt.title('Evolution of accuracy over 100 iterations')
# plt.show()
plt.clf()

loss_beta7 = np.load('Good_one/10e7/HARloss_beta.npy', allow_pickle=True)
loss_alpha7 = np.load('Good_one/10e7/HARloss_alpha.npy', allow_pickle=True)
loss_train7 = np.load('Good_one/10e7/HARloss_train.npy')
loss_classif7 = loss_train7 - 0.0000001 * (loss_alpha7 + 100 * loss_beta7)
loss_valid_target7 = np.load("Good_one/10e7/HARloss_valid_target.npy")
loss_valid_source7 = np.load("Good_one/10e7/HARloss_valid_source.npy")

loss_beta5 = np.load('Good_one/10e5/HARloss_beta.npy', allow_pickle=True)
loss_alpha5 = np.load('Good_one/10e5/HARloss_alpha.npy', allow_pickle=True)
loss_train5 = np.load('Good_one/10e5/HARloss_train.npy')
loss_classif5 = loss_train7 - 0.00001 * (loss_alpha5 + loss_beta5)
loss_valid_target5 = np.load("Good_one/10e5/HARloss_valid_target.npy")
loss_valid_source5 = np.load("Good_one/10e5/HARloss_valid_source.npy")

loss_beta = np.load('Good_one/HARloss_beta.npy', allow_pickle=True)
loss_alpha = np.load('Good_one/HARloss_alpha.npy', allow_pickle=True)
loss_train = np.load('Good_one/HARloss_train.npy')
loss_classif = loss_train7 - 0.00001 * (loss_alpha + 10 * loss_beta)
loss_valid_target = np.load("Good_one/HARloss_valid_target.npy")
loss_valid_source = np.load("Good_one/HARloss_valid_source.npy")


plt.subplot(221)
plt.plot(loss_beta, label='beta loss')
plt.plot(0.00001 * loss_alpha, label="weighted alpha loss")
plt.plot(loss_classif, label="Train loss")
plt.title("a = 1e-5 b = 1e-4 iteration 100 HAR")
plt.legend()

plt.subplot(222)
plt.plot(loss_beta5, label='beta loss')
plt.plot(0.00001 * loss_alpha5, label="weighted alpha loss")
plt.plot(loss_classif5, label="Train loss")
plt.title("a = 1e-5 b = 1e-5 iteration 100 HAR")
plt.legend()

plt.subplot(223)
plt.plot(loss_beta7, label='beta loss')
plt.plot(0.0000001 * loss_alpha7, label="weighted alpha loss")
plt.plot(loss_classif7, label="Train loss")
plt.title("a = 1e-7 b = 1e-5 iteration 100 HAR")
plt.legend()
plt.tight_layout()

plt.subplot(224)
plt.plot(loss_beta200, label='beta loss')
plt.plot(0.0000001 * loss_alpha200, label="weighted alpha loss")
plt.plot(loss_classif200, label="Train loss")
plt.title("a = 1e-7 b = 1e-5 iteration 200 HAR")
plt.legend()
plt.tight_layout()
# plt.show()
plt.clf()

plt.subplot(221)
plt.plot(loss_valid_source, label="loss valid source")
plt.plot(loss_valid_target, label="loss valid target")
plt.title("a = 1e-5 b = 1e-4 iteration 100 HAR")
plt.legend()

plt.subplot(222)
plt.plot(loss_valid_source5, label="loss valid source")
plt.plot(loss_valid_target5, label="loss valid target")
plt.title("a = 1e-5 b = 1e-5 iteration 100 HAR")
plt.legend()

plt.subplot(223)
plt.plot(loss_valid_source7, label="loss valid source")
plt.plot(loss_valid_target7, label="loss valid target")
plt.title("a = 1e-7 b = 1e-5 iteration 100 HAR")
plt.legend()


plt.plot(loss_beta200, label='beta loss')
plt.plot(0.0000001 * loss_alpha200, label="weighted alpha loss")
plt.plot(loss_classif200, label="Train loss")
plt.title("a = 1e-7 b = 1e-5 iteration 200 HAR")
plt.legend()
plt.tight_layout()
# plt.show()

# acc_table = np.empty(shape=(30, 100))
alpha_table = np.empty(shape=(30, 1))
beta_table = np.empty(shape=(30, 1))
i = 0


"""for a in [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
    for b in [1., 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        accuracy = np.load("/share/home/fpainblanc/MAD-CNN/hyperparameters/UCIHAR_IT/0.001/256/" + str(a) + "/" +
                           str(b) + "/" + "14_19/HARacc_target.npy")
        acc_table[i, :] = accuracy
        alpha_table[i] = a
        beta_table[i] = b
        i += 1
print(alpha_table.reshape(5, 6))
print(beta_table.reshape(5, 6))
# np.save("Accuracy_few_iter_HAR2.npy", acc_table)"""
acc_table = np.load("Accuracy_HAR_new.npy")

# acc_table = np.random.rand(30, 100)

fig = plt.figure(figsize=(10, 9))
# camera = Camera(plt.figure(figsize=(5, 6)))
for t in range(0, 10):
    i = 20 * t
    heat_map = sns.heatmap(acc_table[:, t].reshape(10, 9), linewidth=1, annot=True)
    heat_map.set_xticklabels([1., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001])
    heat_map.set_yticklabels([1., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001])
    heat_map.set_title('Iteration ' + str(i + 1))
    # camera.snap()
    plt.show()
    # plt.clf()"""
"""animation = camera.animate()
animation.save("acc_try_1" + '.gif')
"""

