import numpy as np
import numpy.random as npr

index_test = np.arange(0, 1412)
npr.shuffle(index_test)

"""train_dataset = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_features_b.npy")
train_dataset = train_dataset[index_test]
train_dataset = train_dataset[:500]
np.save('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn500.npy', train_dataset)"""

"""mean_train = np.mean(train_dataset)
std_train = np.std(train_dataset)
train_dataset = (train_dataset - mean_train) / std_train"""

"""train_label = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn_label_b.npy')
train_label = train_label[index_test]
train_label = train_label[:500]
np.save("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarn500_label.npy", train_label)"""

"""test_dataset = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_features_b.npy')
test_dataset = test_dataset[index_test]
test_dataset = test_dataset[:500]
np.save("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh500.npy", test_dataset)"""

"""mean_test = np.mean(test_dataset)
std_test = np.std(test_dataset)
test_dataset = (test_dataset - mean_test) / std_test"""
"""test4train = test_dataset[:100]
test4test = test_dataset[100:]"""

"""test4train_t = torch.from_numpy(test4train)
test4train_t = test4train_t.type(torch.float)"""

"""test_label = np.load('/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh_label_b.npy')
test_label = test_label[index_test]
test_label = test_label[:500]
np.save("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/bzh500_label.npy", test_label)"""

"""test4train_label = test_label[:100]
test_label = test_label[100:]
test4train_label_t = torch.from_numpy(test4train_label)
test4train_label_t = test4train_label_t.type(torch.long)"""
"""CNN_tens = CNN_class(X_train=train_dataset, y_train=train_label, batchsize=32, epoch=10000)
CNN_tens.train()
accuracy = CNN_tens.evaluate_no_DTW(test_dataset, test_label)
print(accuracy, 'tensorflow model')"""

train = np.load("/home/adr2.local/painblanc_f/Desktop/bzh_datasets/tarnbzh_2train.npy")
print(train.shape, train.mean())
