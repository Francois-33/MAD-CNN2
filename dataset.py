"""index_test = np.arange(0, 1412)
    npr.shuffle(index_test)"""

# train_dataset = np.load("tarn_features_b.npy")
"""train_dataset = train_dataset[index_test]
train_dataset = train_dataset[:500]"""
"""mean_train = np.mean(train_dataset)
std_train = np.std(train_dataset)
train_dataset = (train_dataset - mean_train) / std_train"""

# train_label = np.load('tarn_label_b.npy')

# test_dataset = np.load('bzh_features_b.npy')
"""test_dataset = test_dataset[index_test]
test_dataset = test_dataset[:500]"""
"""mean_test = np.mean(test_dataset)
std_test = np.std(test_dataset)
test_dataset = (test_dataset - mean_test) / std_test"""
"""test4train = test_dataset[:100]
test4test = test_dataset[100:]"""

"""test4train_t = torch.from_numpy(test4train)
test4train_t = test4train_t.type(torch.float)"""

# test_label = np.load('bzh_label_b.npy')
"""test_label = test_label[index_test]
test_label = test_label[:500]
test4train_label = test_label[:100]
test_label = test_label[100:]
test4train_label_t = torch.from_numpy(test4train_label)
test4train_label_t = test4train_label_t.type(torch.long)"""
"""CNN_tens = CNN_class(X_train=train_dataset, y_train=train_label, batchsize=32, epoch=10000)
CNN_tens.train()
accuracy = CNN_tens.evaluate_no_DTW(test_dataset, test_label)
print(accuracy, 'tensorflow model')"""
