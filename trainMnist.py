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
sys.path.append("TrainingLoss")
import numpy as np
from utils.tools_func import *
import matplotlib.pyplot as plt
import time

colorValues = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff','#ff00ff', '#990000', '#999900', '#009900', '#009999']
LossNames   = ['softmax','Centerloss','AdditiveAngularMargin','AdditiveMargin','AngularMargin','LargeMarginCosine']
        
class MnistEval():
    def __init__(self,loss_type):
        # Get MNIST Data
        self.mnist = input_data.read_data_sets('classification_dataset/MNIST/', one_hot=False)
        self.nrof_classes=10
        self.batch_size = 100
        self.test_iter=1000
        self.total_steps = 10000
        self.learning_rate=0.001
        self.feature_length=2
        self.test_batch_size=200
        self.loss_type=loss_type
        # Init Variables
        self.phase_train  = tf.placeholder(tf.bool, name='phase_train')
        self.images_input = tf.placeholder(tf.float32, [None, 28,28,1] )
        self.labels_input = tf.placeholder(tf.int64,   [None,] )
        self.global_step  = tf.Variable(0,trainable=False,name='global_step')

        self.color_list=[ (colorValues[i]) for i in self.mnist.test.labels]
        fig=plt.figure(figsize=(16,9))
        self.ax = fig.add_subplot(111)if self.feature_length==2 else fig.add_subplot(111, projection = '3d')

    def model(self,inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                          weights_regularizer=slim.l2_regularizer(5e-5)):
       
            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.conv2d(net, 128, [3, 3],padding="VALID",scope='conv4')
            net = slim.flatten(net)
            net=slim.fully_connected(net,self.feature_length,activation_fn=None,scope='fc1', reuse=False)
            return net

    def forward(self):
        self.prelogits=self.model(self.images_input)
        if self.loss_type==0  : #softmax loss
            self.logits = slim.fully_connected(self.prelogits,self.nrof_classes,
                activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(5e-5),
                scope='Logits_end',reuse=False)
            softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_input),name="loss")
            tf.add_to_collection('losses', softmaxloss)
        elif self.loss_type==1: #center loss
            lossfunc=importlib.import_module(LossNames[self.loss_type])
            self.logits = slim.fully_connected(self.prelogits,self.nrof_classes,activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(5e-5),scope='Logits_end',reuse=False)
            #softmax loss
            softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_input),name="loss")
            tf.add_to_collection('losses', softmaxloss)
            #center loss
            center_loss, center = lossfunc.cal_loss(self.prelogits, self.labels_input, self.nrof_classes,alpha=0.1)
            tf.add_to_collection('losses', center_loss * 0.1)
        else :
            lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
            self.logits,custom_loss=lossfunc.cal_loss(prelogits,labels_input,self.nrof_classes)
            tf.add_to_collection('losses', custom_loss)

        custom_loss=tf.get_collection("losses")
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')

    def optimizer(self):
        # optimizer
        optimizer = tf.train.AdamOptimizer( self.learning_rate )
        grads = optimizer.compute_gradients(self.total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        ##------------------------------------------------------------------------------------##

    def process(self):
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1),self.labels_input ), tf.float32))
        # Run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Train 10000 steps
            for step in range(self.total_steps + 1):
                images, labels = self.mnist.train.next_batch(self.batch_size)
                images=np.reshape(images,(self.batch_size,28,28,1))
                fd={self.images_input:images,self.labels_input:labels,self.phase_train: True}
                _,trainloss,acc=sess.run([self.train_op,self.total_loss,self.accuracy],feed_dict=fd)

                if step%self.test_iter==0:
                    #print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )
                    images=np.reshape(self.mnist.test.images,(10000,28,28,1))
                    labels=self.mnist.test.labels
                    total_acc=0
                    self.test_features=[]
                    test_iter=int(10000/self.test_batch_size)
                    for i in range(test_iter):
                        fd={self.images_input:images[i*self.test_batch_size:(i+1)*self.test_batch_size],self.labels_input:labels[i*self.test_batch_size:(i+1)*self.test_batch_size],self.phase_train: False}
                        test_features_tmp,acc=sess.run([self.prelogits,self.accuracy],feed_dict=fd)
                        total_acc+=acc
                        self.test_features.append((test_features_tmp) )

                    self.test_features=np.row_stack(self.test_features)
                    
                    print ("step %d test accuracy %.4f"%(step,total_acc*1.0/test_iter) )
            self.draw_features("images/iter_%d.png"%int(step/self.test_iter),self.test_features,self.mnist.test.labels)

    def draw_features(self,savepath,features,labels):
        #print ("saveing the features")
        if features.shape[1]==2:
            self.ax.scatter(features[...,0],features[...,1],c=self.color_list)
        elif features.shape[1]==3:
            self.ax.scatter(features[...,0],features[...,1],features[...,2],c=self.color_list)
        plt.savefig(savepath)
        plt.draw()
        plt.show()
        plt.pause(0.001)

    def run(self):
        self.forward()
        self.optimizer()
        self.process()
        #self.draw_feature()


if __name__ =="__main__":
    demo=MnistEval(loss_type=1)
    demo.run()