from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim
import sys
import cv2
sys.path.append("TrainingNet")
sys.path.append("TrainingLoss")
import numpy as np
import utils.tools_func
import config
import keras
from keras.datasets import cifar100

class Cifar10Eval():
    def __init__(self):
        # Get MNIST Data
        (self.images_train,self.labels_train),(self.images_test,self.labels_test) = cifar100.load_data()
        self.labels_train=np.reshape(self.labels_train,(self.labels_train.shape[0]))
        self.labels_test=np.reshape (self.labels_test, (self.labels_test.shape[0]))

        self.nrof_classes=100
        self.batch_size = 20
        self.max_epoch = 100
        self.feature_length=512
        self.learning_rate=0.001
         # Variables
        self.phase_train  = tf.placeholder(tf.bool, name='phase_train')
        self.images_input = tf.placeholder(tf.float32, [self.batch_size, 32,32,3])
        self.labels_input = tf.placeholder(tf.int64, [self.batch_size,])
        self.global_step  = tf.Variable(0,trainable=False,name='global_step')


    def loss(self):
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_input),name="loss")


    def model(self):
        #load network
        network = importlib.import_module(config.classification_model_def)
        print ( 'trainnet : %s'%config.classification_model_def )
        print ( 'losstype : %s'%config.loss_type_list[config.loss_type] )

        self.prelogits = network.inference(self.images_input,
                                           phase_train=self.phase_train,
                                           feature_length=self.feature_length)

        self.logits = slim.fully_connected(self.prelogits,
                                           self.nrof_classes,
                                           activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           weights_regularizer=slim.l2_regularizer(5e-5),
                                           scope='Logits_end',
                                           reuse=False)

    def optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1),self.labels_input ), tf.float32))

    def process(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Train 10000 steps
            for epoch in range(self.max_epoch):

                np.random.seed(1)
                np.random.shuffle(self.images_train)
                np.random.seed(1)
                np.random.shuffle(self.labels_train)

                for step in range( int(50000/self.batch_size) ):

                    images, labels = self.images_train[self.batch_size*step:self.batch_size*(step+1)],self.labels_train[self.batch_size*step:self.batch_size*(step+1)]

                    fd={self.images_input:images,self.labels_input:labels ,self.phase_train: True}
                    _,trainloss,acc=sess.run([self.train_op,self.total_loss,self.accuracy],feed_dict=fd)

                    if step%100==0:
                        print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )
                    
                    if step%1000==0:
                        images=self.images_test
                        labels=self.labels_test
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
        self.test_features=(np.array(test_features) ).reshape( (10000,bottleneck_layer_size) )
        tools_func.draw_features(self.test_features,self.labels_test)

    def run(self):
        self.model()
        self.loss()
        self.optimizer()
        self.process()
        self.draw_feature()


if __name__=="__main__":
    demo=Cifar10Eval()
    demo.run()