from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

def cal_loss(embeddings,
                        labels,
                        nrof_classes,
                        w_init=tf.contrib.layers.xavier_initializer(uniform=False),
                        m = 0.35,
                        s=30):
    '''
    paper:<'Additive Margin Softmax for Face Verification'>
    LargeMarginCosine loss as described in : https://arxiv.org/abs/1801.05599
    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    labels : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    '''
    with tf.name_scope('AM_logits'):
        embeddings_norm = tf.nn.l2_normalize(embeddings, 0, 1e-10,name="norm_embedding")

        weights = tf.get_variable(name='embedding_weights', shape=(embeddings.get_shape().as_list()[-1], nrof_classes),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.nn.l2_normalize(weights, 0, 1e-10, name='norm_weights')
        #define s*cod(theta)-m and s*cos(theta)
        cos_theta = tf.matmul(embeddings_norm, weights_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
        cod_theta_m = cos_theta - m
        label_onehot = tf.one_hot(labels, nrof_classes)
        logits = s * tf.where(tf.equal(label_onehot,1), cod_theta_m, cos_theta)
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")
    return logits,loss



def cal_loss_test( ):
    tfe.enable_eager_execution()
    embeddings=tf.get_variable(name="embeddings",dtype=tf.float32,shape=[5,16],initializer=tf.random_normal_initializer(seed=223))
    np.random.seed(0)
    labels=np.random.randint(0,1,size=(5))
    nrof_classes=2
    w_init_method=tf.random_normal_initializer(seed=666)
    logits,loss=cal_loss(embeddings, labels, nrof_classes,w_init=w_init_method)
    print ( logits )
    print ( loss)

if __name__ == "__main__":
    cal_loss_test()
