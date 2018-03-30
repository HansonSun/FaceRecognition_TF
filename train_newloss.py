import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
import numpy as np
import tensorflow as tf
import input_data
import importlib
import config
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
from benchmark_validate import *

def training(total_loss, learning_rate, global_step, update_gradient_vars):
    # Generate moving averages of all losses
    #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #losses = tf.get_collection('losses')
    #loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Compute gradients.
    #with tf.control_dependencies([loss_averages_op]):

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

    grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
       config.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op



def run_training():
    #1.create log and model saved dir according to the datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    logs_dir = os.path.join(os.path.expanduser(config.logs_dir), subdir)
    if not os.path.isdir(config.logs_dir):  # Create the log directory if it doesn't exist
        os.makedirs(config.logs_dir)
    models_dir = os.path.join(os.path.expanduser(config.models_dir), subdir)
    if not os.path.isdir(config.models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(config.models_dir)
    topn_models_dir = os.path.join(models_dir,"topn")#topn die used for save top accuracy model
    if not os.path.isdir(topn_models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(topn_models_dir)
    print('Model directory: %s' % config.models_dir)
    print('Log directory: %s' % config.logs_dir)

    #2.load dataset and define placeholder
    print ("loading dataset...")
    iterator,next_element = input_data.img_input_data( config.training_dateset,config.batch_size)
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    images = tf.placeholder(name='input', shape=[None, config.input_img_height,config.input_img_width, 3], dtype=tf.float32)
    labels = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)

    #3.load model and inference
    network = importlib.import_module(config.train_net)
    print ("trianing net:%s"%config.train_net)
    print ("input image size [h:%d w:%d c:%d]"%(config.input_img_height,config.input_img_width,3))

    features,end_points = network.inference(images,
        keep_probability=config.keep_probability,
        phase_train=phase_train_placeholder,
        weight_decay=config.weight_decay,
        bottleneck_layer_size=config.embedding_size)

    logits = slim.fully_connected(features, config.num_output,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(5e-5),
        scope='Logits_end',
        reuse=False)


    embeddings = tf.nn.l2_normalize(features, 1, 1e-10, name='embeddings')
    predict_labels=tf.argmax(logits,1)

    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate,global_step,config.learning_rate_decay_step,config.learning_rate_decay_rate,staircase=True)

    #cal loss and update
    softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")

    tf.add_to_collection('losses', softmaxloss)
    #add center loss
    if config.center_loss_lambda>0.0:
        print "use center loss"
        prelogits_center_loss, _,_ = lossfunc.center_loss(features, labels, config.center_loss_alpha, config.num_output)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * config.center_loss_lambda)


    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss=tf.add_n([softmaxloss]+regularization_losses,name='total_loss')


    #optimize loss and update
    train_op = training(total_loss,learning_rate,global_step,tf.global_variables())
    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=10)

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
                    input_dict={phase_train_placeholder:True,images:images_train,labels:labels_train}
                    step,lr,train_loss,_,pre_labels,real_labels = sess.run([global_step,learning_rate,total_loss,train_op,predict_labels,labels],feed_dict=input_dict)
                    end_time=time.time()
                    use_time+=(end_time-start_time)
                    train_acc=np.equal(pre_labels,real_labels).mean()

                    #display train result
                    if(step%config.display_iter==0):
                        print "step:%d lr:%f time:%.3f total_loss:%.3f acc:%.3f epoch:%d"%(step,lr,use_time,train_loss,train_acc,epoch)
                        use_time=0
                    if (step%config.save_iter==0):
                        filename = os.path.join(models_dir, "%d.cpkt"%step)
                        saver.save(sess, filename)
                        acc_dict=test_benchmark(models_dir)
                        if acc_dict["lfw_acc"]>99.0:
                            filename = "%s_%d[lfw=%.1f,cff=%.1f,cfp=%.1f].cpkt"%(topn_models_dir,step,acc_dict["lfw_acc"],acc_dict["cff_acc"],acc_dict["cfp_acc"])
                            saver.save(sess, filename)


                except tf.errors.OutOfRangeError:
                    print("End of epoch ")
                    break


run_training()
