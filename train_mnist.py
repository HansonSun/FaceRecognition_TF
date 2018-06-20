from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import importlib
import tensorflow.contrib.slim as slim
import sys
sys.path.append("custom_nets")
sys.path.append("lossfunc")
import numpy as np
import tools_func
import config

# Get MNIST Data
mnist = input_data.read_data_sets('classification_dataset/MNIST/', one_hot=False)
# Variables
nrof_classes=10
batch_size = 20
total_steps = 10000
bottleneck_layer_size=128
draw_feature_flag=0
# Build Model
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
images_placeholder=tf.placeholder(tf.float32, [batch_size, 28,28,1])
labels_placeholder = tf.placeholder(tf.int64, [batch_size,])
global_step=tf.Variable(0,trainable=False,name='global_step')

#-----------------------------------modify here--------------------------------------------------##
#load network
network = importlib.import_module(config.train_net)
print ( 'trainnet : %s'%config.train_net )
print ( 'losstype : %s'%config.loss_type_list[config.loss_type] )

prelogits,end_points = network.inference(images_placeholder,
                                        phase_train=phase_train_placeholder,
                                        weight_decay=5e-5,
                                        bottleneck_layer_size=bottleneck_layer_size)

if config.loss_type==0  : #softmax loss
    logits = slim.fully_connected(prelogits,nrof_classes,
        activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(5e-5),
        scope='Logits_end',reuse=False)
    softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder),name="loss")
    tf.add_to_collection('losses', softmaxloss)
elif config.loss_type==1: #center loss
    lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
    logits = slim.fully_connected(prelogits,nrof_classes,activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(5e-5),scope='Logits_end',reuse=False)
    #softmax loss
    softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder),name="loss")
    tf.add_to_collection('losses', softmaxloss)
    #center loss
    center_loss, _,_ = lossfunc.cal_loss(prelogits, labels_placeholder, nrof_classes,alpha=config.Centerloss_alpha)
    tf.add_to_collection('losses', center_loss * config.Centerloss_lambda)
else :
    lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
    logits,custom_loss=lossfunc.cal_loss(prelogits,labels_placeholder,nrof_classes)
    tf.add_to_collection('losses', custom_loss)


custom_loss=tf.get_collection("losses")
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')


# Loss
optimizer = tf.train.AdamOptimizer(0.001)
grads = optimizer.compute_gradients(total_loss)
update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
##------------------------------------------------------------------------------------##

# Prediction
correct_prediction = tf.equal(tf.argmax(logits,1),labels_placeholder )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train 10000 steps
    for step in range(total_steps + 1):
        images, labels = mnist.train.next_batch(batch_size)
        images=np.reshape(images,(batch_size,28,28,1))
        _,trainloss,acc=sess.run([train_op,total_loss,accuracy],feed_dict={images_placeholder:images,labels_placeholder:labels ,phase_train_placeholder: True})

        if step%100==0:
            print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )
        if step%1000==0:
            images=np.reshape(mnist.test.images,(10000,28,28,1))
            labels=mnist.test.labels
            total_acc=0
            test_features=[]
            test_iter=int(10000/batch_size)
            for i in range(test_iter):
                fd={images_placeholder:images[i*batch_size:(i+1)*batch_size],labels_placeholder:labels[i*batch_size:(i+1)*batch_size],phase_train_placeholder: False}
                test_features_tmp,acc=sess.run([prelogits,accuracy],feed_dict=fd)
                total_acc+=acc
                test_features.append((test_features_tmp) )
            print ("test accuracy %.4f"%(total_acc*1.0/test_iter) )
#draw feature
if draw_feature_flag==1:
    test_features=(np.array(test_features) ).reshape( (10000,bottleneck_layer_size) )
    tools_func.draw_features(test_features,mnist.test.labels)
