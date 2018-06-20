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
            m = 2,
            l=0,
            w_init=tf.contrib.layers.xavier_initializer(uniform=False)):
    '''
    paper:<'SphereFace: Deep Hypersphere Embedding for Face Recognition'>
    AngularMargin loss as described in : https://arxiv.org/abs/1704.08063
    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    labels : ground truth label of current training batch
    nrof_classes: number of classes
    '''

    batch_size,emb_size=embeddings.get_shape().as_list()
    weights = tf.get_variable("AngularMargin_weights", [emb_size, nrof_classes], dtype=tf.float32,
            initializer=w_init)

    eps = 1e-8
    xw = tf.matmul(embeddings,weights)

    if m == 0:
        return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=xw))
    #|weight|
    w_norm = tf.norm(weights, axis = 0) + eps
    #|emb|*cos(theta)
    logits = xw/w_norm

    if labels is None:
        return logits, None

    ordinal = tf.constant(list(range(0, batch_size)), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis = 1)
    #|emb|
    emb_norm = tf.norm(embeddings, axis = 1) + eps
    sel_logits = tf.gather_nd(logits, ordinal_y)
    #cos(theta)
    cos_th = tf.div(sel_logits, emb_norm)

    if m == 1:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        if m == 2:
            #cos(2*theta)=2cos(theta)^2-1
            cos_sign = tf.sign(cos_th)
            psi_theta = 2*tf.multiply(tf.sign(cos_th), tf.square(cos_th)) - 1
        elif m == 4:
            #cos(4*theta)=1+(-8*cos(theta)^2+8*cos(theta)^4)
            cos_th2 = tf.square(cos_th)
            cos_th4 = tf.pow(cos_th, 4)
            sign0 = tf.sign(cos_th)
            sign3 = tf.multiply(tf.sign(2*cos_th2 - 1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            psi_theta = sign3*(8*cos_th4 - 8*cos_th2 + 1) + sign4
        else:
            raise ValueError('unsupported value of m')
        #|emb|psi(theta)
        scaled_logits = tf.multiply(psi_theta, emb_norm)

        f = 1.0/(1.0+l)
        ff = 1.0 - f
        comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape()))
        updated_logits = ff*logits + f*comb_logits_diff

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))

        return logits, loss

def cal_loss_test( ):
    tfe.enable_eager_execution()
    embeddings=tf.get_variable(name="embeddings",dtype=tf.float32,shape=[5,16],initializer=tf.random_normal_initializer(seed=123))
    np.random.seed(0)
    labels=np.random.randint(0,1,size=(5))
    nrof_classes=2
    w_init_method=tf.random_normal_initializer(seed=666)
    logit,loss=cal_loss(embeddings, labels, nrof_classes,w_init=w_init_method)
    print ( loss )

if __name__ == "__main__":
    cal_loss_test()
