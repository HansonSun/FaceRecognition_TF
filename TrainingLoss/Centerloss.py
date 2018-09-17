from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

def cal_loss(embeddings,
             labels,
             nrof_classes,
             alpha,
             w_init=tf.constant_initializer(0)):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = embeddings.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=w_init, trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - embeddings)
    centers_op = tf.scatter_sub(centers, labels, diff)
    loss = tf.reduce_mean(tf.square(embeddings - centers_batch))
    #loss = tf.nn.l2_loss(embeddings - centers_batch)
    return loss, centers,centers_op


'''
def cal_loss2(embeddings,
             labels,
             nrof_classes,
             alpha,
             w_init=tf.contrib.layers.xavier_initializer(uniform=False)):

    len_features = embeddings.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, len_features], dtype=tf.float32,
        initializer=w_init, trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - embeddings
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    #loss = tf.nn.l2_loss(embeddings - centers_batch)
    loss = tf.reduce_mean(tf.square(embeddings - centers_batch))
    return loss, centers, centers_update_op
'''

def cal_loss_test( ):
    embeddings=tf.get_variable(name="embeddings",dtype=tf.float32,shape=[5,16],initializer=tf.random_normal_initializer(seed=223))
    np.random.seed(0)
    labels=np.random.randint(0,1,size=(5))
    nrof_classes=2
    w_init_method=tf.random_normal_initializer(seed=666)
    loss,_,_=cal_loss(embeddings, labels, nrof_classes,alpha=0.9,w_init=w_init_method)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    loss_v= sess.run(loss)
    print (loss_v )
    sess.close()
if __name__ == "__main__":
    cal_loss_test()
