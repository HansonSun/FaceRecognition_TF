from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("custom_nets")
sys.path.append("lossfunc")
import numpy as np
import tensorflow as tf
import input_data
import importlib
import config
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
from benchmark_validate import *
import shutil

def training(total_loss, learning_rate, global_step, update_gradient_vars):
    if config.optimizer=='ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif config.optimizer=='ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif config.optimizer=='ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif config.optimizer=='RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif config.optimizer=='MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op


def run_training():
    #1.create log and model saved dir according to the datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    models_dir = os.path.join("saved_models", subdir, "models")
    if not os.path.isdir(models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(models_dir)
    logs_dir = os.path.join("saved_models", subdir, "logs")
    if not os.path.isdir(logs_dir):  # Create the log directory if it doesn't exist
        os.makedirs(logs_dir)
    topn_models_dir = os.path.join("saved_models", subdir, "topn")#topn dir used for save top accuracy model
    if not os.path.isdir(topn_models_dir):  # Create the topn model directory if it doesn't exist
        os.makedirs(topn_models_dir)
    topn_file=open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")
    topn_file.close()


    #2.load dataset and define placeholder
    print ("loading dataset...")
    iterator,next_element = input_data.img_input_data( config.training_dateset_path,config.batch_size)
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    images_placeholder = tf.placeholder(name='input', shape=[None, config.input_img_height,config.input_img_width, 3], dtype=tf.float32)
    labels_placeholder = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)

    #3.load model and inference
    network = importlib.import_module(config.model_def)
    print ("trianing net:%s"%config.model_def)
    print ("input image size [h:%d w:%d c:%d]"%(config.input_img_height,config.input_img_width,3))

    prelogits,end_points = network.inference(
        images_placeholder,
        keep_probability=0.8,
        phase_train=phase_train_placeholder,
        weight_decay=config.weight_decay,
        bottleneck_layer_size=config.embedding_size)

    if config.loss_type==0  : #softmax loss
        logits = slim.fully_connected(prelogits,
                                      config.nrof_classes,
                                      activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(5e-5),
                                      scope='Logits',reuse=False)
        softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder),name="loss")
        tf.add_to_collection('losses', softmaxloss)
    elif config.loss_type==1: #center loss
        lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
        logits = slim.fully_connected(prelogits,
                                      config.nrof_classes,
                                      activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(5e-5),
                                      scope='Logits',
                                      reuse=False)
        #softmax loss
        softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder),name="loss")
        tf.add_to_collection('losses', softmaxloss)
        #center loss
        center_loss, _,_ = lossfunc.cal_loss(prelogits, labels_placeholder, config.nrof_classes,alpha=config.Centerloss_alpha)
        tf.add_to_collection('losses', center_loss * config.Centerloss_lambda)
    else :
        lossfunc=importlib.import_module(config.loss_type_list[config.loss_type])
        logits,custom_loss=lossfunc.cal_loss(prelogits,labels_placeholder,config.nrof_classes)
        tf.add_to_collection('losses', custom_loss)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    predict_labels=tf.argmax(logits,1)

    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    if config.lr_type=='exponential_decay':
        learning_rate = tf.train.exponential_decay(config.learning_rate,global_step,config.learning_rate_decay_step,config.learning_rate_decay_rate,staircase=True)
    elif config.lr_type=='piecewise_constant':
        learning_rate = tf.train.piecewise_constant(global_step, config.boundaries, config.values)
    elif config.lr_type=='manual_modify':
        pass

    custom_loss=tf.get_collection("losses")
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')


    #optimize loss and update
    train_op = training(total_loss,learning_rate,global_step,tf.global_variables())
    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(config.max_nrof_epochs):
            sess.run(iterator.initializer)
            while True:
                use_time=0
                try:
                    images_train, labels_train = sess.run(next_element)

                    start_time=time.time()
                    input_dict={phase_train_placeholder:True,images_placeholder:images_train,labels_placeholder:labels_train}
                    step,lr,train_loss,_,pre_labels,real_labels = sess.run([global_step,
                                                                            learning_rate,
                                                                            total_loss,
                                                                            train_op,
                                                                            predict_labels,
                                                                            labels_placeholder],
                                                                            feed_dict=input_dict)
                    end_time=time.time()
                    use_time+=(end_time-start_time)
                    train_acc=np.equal(pre_labels,real_labels).mean()

                    #display train result
                    if(step%config.display_iter==0):
                        print ("step:%d lr:%f time:%.3f total_loss:%.3f acc:%.3f epoch:%d"%(step,lr,use_time,train_loss,train_acc,epoch) )
                        use_time=0
                    if (step%config.test_save_iter==0):
                        filename_cpkt = os.path.join(models_dir, "%d.ckpt"%step)
                        saver.save(sess, filename_cpkt)

                        if config.test_lfw==1 :
                            acc_dict=test_benchmark(os.path.join(models_dir))
                            if acc_dict["lfw_acc"]>config.topn_threshold:
                                topn_file=open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")
                                topn_file.write("%s %s\n"%(os.path.join(topn_models_dir, "%d.ckpt"%step),str(acc_dict)) )
                                shutil.copyfile(os.path.join(models_dir, "%d.ckpt.meta"%step),os.path.join(topn_models_dir, "%d.ckpt.meta"%step))
                                shutil.copyfile(os.path.join(models_dir, "%d.ckpt.index"%step),os.path.join(topn_models_dir, "%d.ckpt.index"%step))
                                shutil.copyfile(os.path.join(models_dir, "%d.ckpt.data-00000-of-00001"%step),os.path.join(topn_models_dir, "%d.ckpt.data-00000-of-00001"%step))
                                topn_file.close()

                except tf.errors.OutOfRangeError:
                    print("End of epoch ")
                    break


run_training()
