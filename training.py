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

'''
def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

'''

def trainning(loss,learning_rate,global_step):

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op


def run_training():
    
    image_batch,label_batch,class_num = input_data.GetPathsandLabels( config.training_dateset ) 

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')     
    #load model
    network = importlib.import_module("lightcnn_b")
    prelogits = network.inference(image_batch)

    logits = slim.fully_connected(prelogits, class_num, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(5e-5),
                    scope='Logits', reuse=False)
    predict_labels=tf.argmax(logits,1)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate,global_step,config.decay_step,config.decay_rate,staircase=True)


    centerloss, _ = lossfunc.center_loss(prelogits, label_batch, config.centerloss_alpha, class_num)

    softmaxloss = lossfunc.softmax_loss(logits, label_batch)  

    total_loss = softmaxloss + config.centerloss_lambda * centerloss

    train_op = trainning(total_loss,learning_rate,global_step)

    saver=tf.train.Saver(max_to_keep=50)

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
        try:
            for step in np.arange(config.max_iter):
                if coord.should_stop():
                    break
                start_time=time.time()
                lr,train_loss,_,pre_labels,real_labels = sess.run([learning_rate,total_loss,train_op,predict_labels,label_batch])
                end_time=time.time()
                iter+=1
                use_time+=(end_time-start_time)

                train_acc=np.equal(pre_labels,real_labels).mean()

                if(iter%config.display_iter==0):
                    print "iterator:%d lr:%f time:%f total_loss:%f acc:%.3f"%(iter,lr,use_time,train_loss,train_acc)
                    use_time=0

                if (iter % config.snapshot==0): 
                    saver.save(sess, "model/recognize",global_step = iter)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)


run_training()