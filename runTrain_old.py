import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
import numpy as np
import tensorflow as tf
import input_data
import importlib
import config
import input_data
import tensorflow.contrib.slim as slim
import time
import lossfunc
#import lfw_eval
from datetime import datetime

def feature_normalize(x,name):
    mean, std = tf.nn.moments(x,0)
    output=tf.multiply((x-mean),1.0/std,name=name)
    return output

def training(total_loss, learning_rate, global_step, update_gradient_vars):
    # Generate moving averages of all losses
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

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
    config.logs_dir = os.path.join(os.path.expanduser("log"), subdir)
    if not os.path.isdir(config.logs_dir):  # Create the log directory if it doesn't exist
        os.makedirs(config.logs_dir)
    config.models_dir = os.path.join(os.path.expanduser("model"), subdir)
    if not os.path.isdir(config.models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(config.models_dir)
    print('Model directory: %s' % config.models_dir)
    print('Log directory: %s' % config.logs_dir)

    #2.load dataset
    print ("loading dataset...")
    image_batch,label_batch,class_num,total_img_num = input_data.GetPLFromCsv( config.training_dateset )
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    #3.load model and inference
    network = importlib.import_module(config.train_net)
    print ("trian net:%s"%config.train_net)
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')

    features,_ = network.inference(image_batch,
        phase_train=phase_train_placeholder,
        weight_decay=config.weight_decay,
        bottleneck_layer_size=config.emb_size)

    logits = slim.fully_connected(features, class_num,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(5e-5),
        scope='Logits_end',
        reuse=False)


    embeddings = tf.nn.l2_normalize(features, 1, 1e-10, name='embeddings')
    #pre_embeddings = tf.nn.l2_normalize(features, 1, 1e-10, name='pre_embeddings')
    #embeddings=feature_normalize(pre_embeddings,name="pre_embeddings")
    predict_labels=tf.argmax(logits,1)


    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate,global_step,config.decay_step,config.decay_rate,staircase=True)

    #cal loss and update
    softmaxloss = lossfunc.softmax_loss(logits, label_batch)
    tf.add_to_collection('losses', softmaxloss)
    #add center loss
    if config.centerloss_lambda>0.0:
        prelogits_center_loss, _,_ = lossfunc.center_loss(features, label_batch, config.centerloss_alpha, class_num)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * config.centerloss_lambda)


    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss=tf.add_n([softmaxloss]+regularization_losses,name='total_loss')


    #optimize loss and update
    train_op = training(total_loss,learning_rate,global_step,tf.global_variables())
    #train_op = trainning2(total_loss,learning_rate,global_step)
    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #restore the check point
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
            cur_iter=int(checkpoint.model_checkpoint_path.split("-")[-1])+1


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        iter=0
        use_time=0
        display_loss=0
        dispaly_acc=0
        try:
            for step in np.arange(config.max_iter):
                if coord.should_stop():
                    break
                start_time=time.time()
                lr,train_loss,_,pre_labels,real_labels = sess.run([learning_rate,total_loss,train_op,predict_labels,label_batch],feed_dict={phase_train_placeholder:True})
                end_time=time.time()
                iter+=1
                use_time+=(end_time-start_time)
                display_loss+=train_loss
                train_acc=np.equal(pre_labels,real_labels).mean()
                dispaly_acc+=train_acc

                #display train result
                if(iter%config.display_iter==0):
                    epoch_scale=iter*config.train_batch_size*1.0/total_img_num
                    display_loss=display_loss/config.display_iter
                    dispaly_acc=dispaly_acc/config.display_iter
                    print "iterator:%d lr:%f time:%.3f total_loss:%.3f acc:%.3f epoch:%.3f"%(iter,lr,use_time,display_loss,dispaly_acc,epoch_scale)
                    use_time=0
                    display_loss=0.0
                    dispaly_acc=0.0

                #save cpkt
                if (iter % config.snapshot==0):
                    #acc_dict=lfw_eval.test_lfw()
                    #filename = "%s_%d[lfw=%.1f,cff=%.1f,cfp=%.1f].cpkt"%(config.train_net,iter,acc_dict["lfw_acc"],acc_dict["cff_acc"],acc_dict["cfp_acc"])
                    filename = os.path.join(config.models_dir, "%d.cpkt"%iter)
                    saver.save(sess, filename)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

run_training()
