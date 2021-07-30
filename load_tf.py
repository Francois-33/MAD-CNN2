import tensorflow as tf
import numpy as np

uci_tf = tf.data.TFRecordDataset(
    '/home/adr2.local/painblanc_f/codats-master/datasets/tfrecords/ucihar_14_train.tfrecord')

print(type(uci_tf))
uci_numpy = uci_tf.numpy(uci_tf)
print(uci_numpy.shape)
