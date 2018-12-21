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
from  utils.input_dataset import TFRecordDataset



class trainFR_base():
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
        
        #3. load dataset
        if self.conf.use_tfrecord:
            print ("dataset is tfrecord")
            self.inputdataset=TFRecordDataset(self.conf)
        else:
            print ("dataset is imagedataset")
            self.inputdataset=fu.TFImageDataset(self.conf)

        print ("input image [ height:%d width:%d channel:%d ]"%(self.conf.input_img_height,self.conf.input_img_width,3))
        self.traindata_iterator,self.traindata_next_element=self.inputdataset.generateDataset(  )
        self.nrof_classes=self.inputdataset.nrof_classes
        self.global_step = tf.Variable(0, trainable=False)


    def make_model(self):
        pass

    def adjust_lr(self):
        if self.conf.lr_type=='exponential_decay':
            self.learning_rate = tf.train.exponential_decay(self.conf.learning_rate,
                                                            self.global_step,
                                                            self.conf.learning_rate_decay_step,
                                                            self.conf.learning_rate_decay_rate,
                                                            staircase=True)
        elif self.conf.lr_type=='piecewise_constant':
            self.learning_rate = tf.train.piecewise_constant(self.global_step, self.conf.boundaries, self.conf.values)
        elif self.conf.lr_type=='manual_modify':
            pass
        tf.summary.scalar("learning_rate",self.learning_rate)


    def summary(self):
        #add grad histogram op
        for grad, var in self.grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        #add trainabel variable gradients
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        self.summary_op = tf.summary.merge_all()
        self.summary_writer=tf.summary.FileWriter(self.logs_dir,tf.get_default_graph())
        self.saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=2)

    def make_loss(self):
        pass

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
        print ("optimizer stratege : %s"%self.conf.optimizer)

        self.grads = opt.compute_gradients(self.total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = opt.apply_gradients(self.grads, global_step=self.global_step)


        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')

        return self.train_op


    def process(self):
       pass

    def run(self):
        self.make_model()
        self.adjust_lr()
        self.make_loss()
        self.optimizer()
        self.summary()
        self.train()


    def train_display(self):
        #display train result
        if(self.step%self.conf.display_iter==0):
            print ("step:%d lr:%f time:%.3f s total_loss:%.3f acc:%.3f epoch:%d"%(self.step,lr,use_time,train_loss,train_acc,epoch) )
            use_time=0
        
        if (self.conf.save_iter!=-1 and self.step%self.conf.save_iter==0):
            print ("save checkpoint")
            filename_cpkt = os.path.join(self.models_dir, "%d.ckpt"%self.step)
            saver.save(sess, filename_cpkt)

        if (self.conf.test_iter!=-1 and self.step%self.conf.test_iter==0):
            if self.conf.benchmark_dict["test_lfw"] :
                acc_dict=test_benchmark(self.conf,self.models_dir)
                if acc_dict["lfw_acc"]>self.conf.topn_threshold:
                    with open(os.path.join(self.logs_dir,"topn_acc.txt"),"a+") as topn_file:
                        topn_file.write("%s %s\n"%(os.path.join(self.topn_models_dir, "%d.ckpt"%self.step),str(acc_dict)) )
                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.meta"%self.step),os.path.join(self.topn_models_dir, "%d.ckpt.meta"%self.step))
                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.index"%self.step),os.path.join(self.topn_models_dir, "%d.ckpt.index"%self.step))
                    shutil.copyfile(os.path.join(self.models_dir, "%d.ckpt.data-00000-of-00001"%self.step),os.path.join(self.topn_models_dir, "%d.ckpt.data-00000-of-00001"%step))


if __name__=="__main__":
    fr=trainFR_base()
    fr.run()