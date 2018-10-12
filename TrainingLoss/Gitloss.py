from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np



def cal_loss(features,labels,nrof_classes,alpha,lambda_c=0.2,lambda_g=0.2,w_init=tf.constant_initializer(0)):
    """
    Git Loss for Deep Face Recognition
    """
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, len_features], dtype=tf.float32,initializer=w_init, trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss=tf.nn.l2_loss(features - centers_batch)
    # Pairwise differences
    diffs = (features[:, tf.newaxis] - centers_batch[tf.newaxis, :])
    diffs_shape = tf.shape(diffs)

    # Mask diagonal (where i == j)
    mask = 1 - tf.eye(diffs_shape[0], diffs_shape[1], dtype=diffs.dtype)
    diffs = diffs * mask[:, :, tf.newaxis]

    # combinaton of two losses
    loss2 = tf.reduce_mean(tf.divide(1, 1 + tf.square(diffs)))

    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast((1 + appear_times), tf.float32))
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)  # diff is used to get updated centers.

    # combo_loss = value_factor * loss + new_factor * loss2
    combo_loss = lambda_c * loss + lambda_g * loss2

    return combo_loss, centers_update_op


    
def cal_loss_test( ):
    #synthesis features which batch size is 5 ,lenghtn is 16
    embeddings=tf.get_variable(name="embeddings",dtype=tf.float32,shape=[6,2],initializer=tf.random_normal_initializer(seed=223))
    labels=np.array((0,0,0,1,1,2))
    print ("labels:",labels)
    nrof_classes=3
    w_init_method=tf.random_normal_initializer(seed=666)
    loss,center=cal_loss(embeddings, labels, nrof_classes,alpha=0.1,w_init=w_init_method)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    embeddings_v,loss_v= sess.run([embeddings,loss])
    print (loss_v,embeddings_v )
    sess.close()
if __name__ == "__main__":
    cal_loss_test()
