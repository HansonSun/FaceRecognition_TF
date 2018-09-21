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
sys.path.append("TrainingNet")
sys.path.append("TrainingLoss")
import numpy as np
import utils.tools_func
import config



class MnistEval():
    def __init__(self):
        # Get MNIST Data
        self.mnist = input_data.read_data_sets('classification_dataset/MNIST/', one_hot=False)
        self.nrof_classes=10
        self.batch_size = 20
        self.total_steps = 10000
        self.feature_length=2
        self.draw_feature_flag=0
        self.learning_rate=0.001
        # Init Variables
        self.phase_train  = tf.placeholder(tf.bool, name='phase_train')
        self.images_input = tf.placeholder(tf.float32, [self.batch_size, 28,28,1] )
        self.labels_input = tf.placeholder(tf.int64,   [self.batch_size,] )
        self.global_step  = tf.Variable(0,trainable=False,name='global_step')


    def model(self):
        #load network
        self.network = importlib.import_module(config.classification_model_def)
        print ( 'trainnet : %s'%config.classification_model_def )
        print ( 'losstype : %s'%config.loss_type_list[config.loss_type] )

        self.prelogits = self.network.inference(self.images_input,
                                    phase_train=self.phase_train,
                                    feature_length=self.feature_length)

    def loss(self):
        if config.loss_type==0  : #softmax loss
            self.logits = slim.fully_connected(self.prelogits,self.nrof_classes,
                activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(5e-5),
                scope='Logits_end',reuse=False)
            softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_input),name="loss")
            tf.add_to_collection('losses', softmaxloss)
        elif config.loss_type==1: #center loss
            lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
            self.logits = slim.fully_connected(prelogits,self.nrof_classes,activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(5e-5),scope='Logits_end',reuse=False)
            #softmax loss
            softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder),name="loss")
            tf.add_to_collection('losses', softmaxloss)
            #center loss
            center_loss, _,_ = lossfunc.cal_loss(prelogits, labels_placeholder, self.nrof_classes,alpha=config.Centerloss_alpha)
            tf.add_to_collection('losses', center_loss * config.Centerloss_lambda)
        else :
            lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
            self.logits,custom_loss=lossfunc.cal_loss(prelogits,labels_placeholder,self.nrof_classes)
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

                if step%100==0:
                    print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )
                if step%1000==0:
                    images=np.reshape(self.mnist.test.images,(10000,28,28,1))
                    labels=self.mnist.test.labels
                    total_acc=0
                    self.test_features=[]
                    test_iter=int(10000/self.batch_size)
                    for i in range(test_iter):
                        fd={self.images_input:images[i*self.batch_size:(i+1)*self.batch_size],self.labels_input:labels[i*self.batch_size:(i+1)*self.batch_size],self.phase_train: False}
                        test_features_tmp,acc=sess.run([self.prelogits,self.accuracy],feed_dict=fd)
                        total_acc+=acc
                        self.test_features.append((test_features_tmp) )
                    print ("test accuracy %.4f"%(total_acc*1.0/test_iter) )


    def draw_feature(self):
        #draw feature
        self.test_features=(np.array(self.test_features) ).reshape( (10000,self.feature_length) )
        tools_func.draw_features( self.test_features,self.mnist.test.labels)


    def run(self):
        self.model()
        self.loss()
        self.optimizer()
        self.process()
        self.draw_feature()


if __name__ =="__main__":
    demo=MnistEval()
    demo.run()