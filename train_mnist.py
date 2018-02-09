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

# Variables
class_num=10
batch_size = 100
total_steps = 5000

# Build Model
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train') 
img_placeholder=tf.placeholder(tf.float32, [None, 28,28,1])
label_placeholder = tf.placeholder(tf.int64, [None])
global_step=tf.Variable(0,trainable=False)

network = importlib.import_module("lightcnn_b")
prelogits,_ = network.inference(img_placeholder,is_training=phase_train_placeholder,bottleneck_layer_size=1024)
logits = slim.fully_connected(prelogits, class_num, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    scope='Logits_end', reuse=False)

softmaxloss = lossfunc.softmax_loss(logits, label_placeholder) 

# Loss
train_op = tf.train.AdagradOptimizer(0.01).minimize(softmaxloss,global_step=global_step)

# Prediction
correct_prediction = tf.equal(tf.argmax(logits,1),label_placeholder )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train 10000 steps
    for step in range(total_steps + 1):
        images, labels = mnist.train.next_batch(batch_size)

        images=np.reshape(images,(batch_size,28,28,1))
        _,trainloss,acc=sess.run([train_op,softmaxloss,accuracy],feed_dict={img_placeholder:images,label_placeholder:labels ,phase_train_placeholder: True})

        if step%100==0:
            print trainloss,acc

        if step%100==0:
            saver.save(sess,"MNIST/MNIST_model/mnist",global_step=global_step)
            images=np.reshape(mnist.test.images,(10000,28,28,1))
            test_features,acc=sess.run([prelogits,accuracy],feed_dict={img_placeholder:images,label_placeholder:mnist.test.labels,phase_train_placeholder: False})
            #print "test acc",acc
            print test_features.shape

