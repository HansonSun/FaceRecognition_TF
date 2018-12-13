from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("/home/hanson/facetools")
sys.path.append("TrainingNet")
sys.path.append("TrainingLoss")
import numpy as np
import tensorflow as tf
import importlib
import config
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
from utils.benchmark_validate import *
import shutil
import faceutils as fu



class trainFR():
    def __init__(self):
        #1. load config file
        self.conf=config.get_config()
        #2. create log and model saved dir according to the datetime
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.models_dir = os.path.join("saved_models", subdir, "models")
        if not os.path.isdir(self.models_dir):  # Create the model directory if it doesn't exist
            os.makedirs(self.models_dir)
        self.logs_dir = os.path.join("saved_models", subdir, "logs")
        if not os.path.isdir(self.logs_dir):  # Create the log directory if it doesn't exist
            os.makedirs(self.logs_dir)
        self.topn_models_dir = os.path.join("saved_models", subdir, "topn")#topn dir used for save top accuracy model
        if not os.path.isdir(self.topn_models_dir):  # Create the topn model directory if it doesn't exist
            os.makedirs(self.topn_models_dir)
        self.topn_file=open(os.path.join(self.topn_models_dir,"topn_acc.txt"),"a+")
        #3. load dataset
        self.inputdataset=fu.TFImageDataset(self.conf)
        self.traindata_iterator,self.traindata_next_element=self.inputdataset.load_image_dataset( buffer_size=20000 )
        self.nrof_classes=self.inputdataset.nrof_classes
        self.total_img_num=self.inputdataset.total_img_num

    def make_model(self):
        #2.load dataset and define placeholder
        self.phase_train  = tf.placeholder(tf.bool, name='phase_train')
        self.images_input = tf.placeholder(name='input', shape=[None, self.conf.input_img_height,self.conf.input_img_width, 3], dtype=tf.float32)
        self.labels_input = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)

        #3.load model and inference
        network = importlib.import_module(self.conf.fr_model_def)
        print ("trianing net:%s"%self.conf.fr_model_def)
        print ("input image size [h:%d w:%d c:%d]"%(self.conf.input_img_height,self.conf.input_img_width,3))

        self.prelogits = network.inference(
            self.images_input,
            dropout_keep_prob=0.8,
            phase_train=self.phase_train,
            weight_decay=self.conf.weight_decay,
            feature_length=self.conf.feature_length)

    def loss(self):
        logits = slim.fully_connected(self.prelogits,
                                      self.inputdataset.nrof_classes,
                                      activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(5e-4),
                                      scope='Logits',
                                      reuse=False)
        softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_input),name="loss")

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(self.prelogits)+eps, ord=1, axis=1))
        tf.add_to_collection('losses', prelogits_norm * 5e-4)
        tf.add_to_collection('losses', softmaxloss)

        embeddings = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')
        self.predict_labels=tf.argmax(logits,1)

        #adjust learning rate
        self.global_step = tf.Variable(0, trainable=False)
        if self.conf.lr_type=='exponential_decay':
            self.learning_rate = tf.train.exponential_decay(self.conf.learning_rate,
                                                            self.global_step,
                                                            self.conf.learning_rate_decay_step,
                                                            self.conf.learning_rate_decay_rate,
                                                            staircase=True)
        elif self.conf.lr_type=='piecewise_constant':
            self.learning_rate = tf.train.piecewise_constant(self.global_step, 
                                                             [ int(i*self.inputdataset.nrof_classes/self.conf.batch_size) for i in self.conf.boundaries], 
                                                             self.conf.values)
        elif self.conf.lr_type=='manual_modify':
            pass
        print ("learning rate use %s"%self.conf.lr_type)

        custom_loss=tf.get_collection("losses")
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')


    def optimizer(self):
        if self.conf.optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.conf.optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.9, epsilon=1e-6)
        elif self.conf.optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif self.conf.optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif self.conf.optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        print ("optimizer use %s"%self.conf.optimizer)

        grads = opt.compute_gradients(self.total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(grads, global_step=self.global_step)


    def process(self):
        saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=3)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(self.conf.max_nrof_epochs):
                sess.run(self.traindata_iterator.initializer)
                while True:
                    use_time=0
                    try:
                        images_train, labels_train = sess.run(self.traindata_next_element)

                        start_time=time.time()
                        input_dict={self.phase_train:True,self.images_input:images_train,self.labels_input:labels_train}
                        step,lr,train_loss,_,pre_labels,real_labels = sess.run([self.global_step,
                                                                                self.learning_rate,
                                                                                self.total_loss,
                                                                                self.train_op,
                                                                                self.predict_labels,
                                                                                self.labels_input],
                                                                                feed_dict=input_dict)
                        end_time=time.time()
                        use_time+=(end_time-start_time)
                        train_acc=np.equal(pre_labels,real_labels).mean()

                        #display train result
                        if(step%self.conf.display_iter==0):
                            print ("step:%d lr:%f time:%.3f s total_loss:%.3f acc:%.3f epoch:%d"%(step,lr,use_time,train_loss,train_acc,epoch) )
                            use_time=0
                        
                        if (step%self.conf.test_save_iter==0):
                            filename_cpkt = os.path.join(self.models_dir, "%d.ckpt"%step)
                            saver.save(sess, filename_cpkt)
                            
                            if self.conf.benchmark_dict["test_lfw"] :
                                acc_dict=test_benchmark(self.conf,self.models_dir)

                                if acc_dict["lfw_acc"]>self.conf.topn_threshold:
                                    self.topn_file.write("%s %s\n"%(os.path.join(self.topn_models_dir, "%d.ckpt"%step),str(acc_dict)) )
                                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.meta"%step),os.path.join(self.topn_models_dir, "%d.ckpt.meta"%step))
                                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.index"%step),os.path.join(self.topn_models_dir, "%d.ckpt.index"%step))
                                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.data-00000-of-00001"%step),os.path.join(self.topn_models_dir, "%d.ckpt.data-00000-of-00001"%step))
                            
                        
                    except tf.errors.OutOfRangeError:
                        print("End of epoch ")
                        break
    def run(self):
        self.make_model()
        self.loss()
        self.optimizer()
        self.process()




if __name__=="__main__":
    fr=trainFR()
    fr.run()