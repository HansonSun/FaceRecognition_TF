import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import importlib
import tensorflow.contrib.slim as slim
import sys
sys.path.append("./nets")
import numpy as np
import lossfunc
import cv2

# Get MNIST Data
mnist = input_data.read_data_sets('MNIST/MNIST_data/', one_hot=False)
# Build Model
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train') 
img_placeholder=tf.placeholder(tf.float32, [batch_size, 28,28,1])
label_placeholder = tf.placeholder(tf.int64, [batch_size])

with tf.Session() as sess:
    metafile=tf.train.lastest_checkpoint("MNIST/MNIST_model/")
    saver=tf.train.import_meta_graph(metafile+".meta")
    saver.restore(sess,saver,metafile)
    sess.run()

