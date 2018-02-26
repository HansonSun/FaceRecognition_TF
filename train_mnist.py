import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import importlib
import tensorflow.contrib.slim as slim
import sys
sys.path.append("./nets")
import numpy as np
import lossfunc
import cv2
from visual_feature import *
from Loss_ASoftmax import Loss_ASoftmax
import tflearn

def Network(data_input, training = True):
    x = tflearn.conv_2d(data_input, 32, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 32, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.flatten(x)
    feat = tflearn.fully_connected(x, 2, weights_init = 'xavier')
    return feat


# Get MNIST Data
mnist = input_data.read_data_sets('MNIST/MNIST_data/', one_hot=False)
# Variables
class_num=10
batch_size = 100
total_steps = 5000
feat_size=2
# Build Model
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train') 
img_placeholder=tf.placeholder(tf.float32, [batch_size, 28,28,1])
label_placeholder = tf.placeholder(tf.int64, [batch_size,])
global_step=tf.Variable(0,trainable=False,name='global_step')


#-----------------------------------modify here--------------------------------------------------##
network = importlib.import_module("lightcnn_b")
features,_ = network.inference(img_placeholder,is_training=phase_train_placeholder,bottleneck_layer_size=feat_size)
logits = slim.fully_connected(features, class_num,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),scope='Logits_end', reuse=False)
#features = Network(img_placeholder)
#logits, total_loss = Loss_ASoftmax(x = features, y = label_placeholder, l = 1.0, num_cls = 10, m = 2)

softmaxloss = lossfunc.softmax_loss(logits, label_placeholder) 
centerloss, centers, centers_update_op=lossfunc.center_loss(features,label_placeholder,0.9,10)
total_loss=softmaxloss+centerloss*0.01

# Loss
optimizer = tf.train.AdamOptimizer(0.001)
with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)

##------------------------------------------------------------------------------------##



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
        _,trainloss,acc=sess.run([train_op,total_loss,accuracy],feed_dict={img_placeholder:images,label_placeholder:labels ,phase_train_placeholder: True})

        if step%100==0:
            print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )

        if step%1000==0:
            saver.save(sess,"MNIST/MNIST_model/mnist",global_step=global_step)
            images=np.reshape(mnist.test.images,(10000,28,28,1))
            labels=mnist.test.labels

            total_acc=0
            test_features=[]
            test_iter=int(10000/batch_size)
            for i in range(test_iter):  
                fd={img_placeholder:images[i*batch_size:(i+1)*batch_size],label_placeholder:labels[i*batch_size:(i+1)*batch_size],phase_train_placeholder: False}
                test_features_tmp,acc=sess.run([features,accuracy],feed_dict=fd)
                total_acc+=acc
                test_features.append((test_features_tmp) )
            print ("test accuracy %.4f"%(total_acc*1.0/test_iter) ) 
#draw feature 
test_features=(np.array(test_features) ).reshape( (10000,feat_size) )
draw_2d_features(test_features,mnist.test.labels)
