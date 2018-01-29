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


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def trainning(loss):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= config.base_lr)
        train_op = optimizer.minimize(loss)
    return train_op


def run_training():

    image_batch,label_batch,class_num = input_data.GetPathsandLabels( )      
    #load model
    network = importlib.import_module("lightcnn_c")
    prelogits = network.inference(image_batch)

    logits = slim.fully_connected(prelogits, class_num, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(0.001),
                    scope='Logits', reuse=False)

    predict_labels=tf.argmax(logits,1)

    loss = losses(logits, label_batch)      

    train_op = trainning(loss)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    iter=0
    try:
        for step in np.arange(config.max_iter):
            if coord.should_stop():
                break

            train_loss,_,pre_labels,real_labels = sess.run([loss,train_op,predict_labels,label_batch])
            iter+=1

            train_acc=np.equal(pre_labels,real_labels).mean()

            if(iter%config.display_iter==0):
                print "iterator:%d train loss:%f accuracy:%f"%(iter,train_loss,train_acc)


    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()


run_training()