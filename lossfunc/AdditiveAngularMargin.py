import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import math
import numpy as np

def cal_loss(embeddings,
               labels,
               nrof_classes,
               w_init=tf.contrib.layers.xavier_initializer(uniform=False),
               s=64.,
               m=0.5):
    '''
    LargeMarginCosine loss as described in https://arxiv.org/abs/1801.07698
    'ArcFace: Additive Angular Margin Loss for Deep Face Recognition'
    :param embeddings: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param nrof_classes: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('AdditiveAngularMargin_loss'):

        # inputs and weights norm
        weights = tf.get_variable(name='embedding_weights', shape=(embeddings.get_shape().as_list()[-1], nrof_classes),
                      initializer=w_init, dtype=tf.float32)
        print weights
        weights_norm = tf.nn.l2_normalize(weights, 0, 1e-10, name='norm_weights')
        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10,name="norm_embeddings")
        #cos(theta+m)=cos(theta)*cos(m)-sin(theta)*sin(m)
        cos_t = tf.matmul(embeddings_norm, weights_norm, name='cos_t')

        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi  --->  -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=nrof_classes, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        #s*(cos(theta+m))
        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")
    return logits,loss


def cal_loss_test( ):
    tfe.enable_eager_execution()
    embedding=tf.ones((3,3),dtype=tf.float32)
    labels=np.arange(0,3,1)
    nrof_classes=3
    w_init_method=tf.random_normal_initializer(seed=666)
    logits,loss=cal_loss(embedding, labels, nrof_classes,w_init=w_init_method)
    print logits
    print loss

if __name__ == "__main__":
    cal_loss_test()
