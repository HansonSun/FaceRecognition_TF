import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

def cal_loss(embeddings,
           labels,
           nrof_classes,
           w_init= tf.contrib.layers.xavier_initializer(uniform=False),
           s=30.,
           m=0.4):
    '''
    LargeMarginCosine loss as described in https://arxiv.org/abs/1801.09414
    'CosFace: Large Margin Cosine Loss for Deep Face Recognition'
    :param embeddings: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param nrof_classes: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embeddings_norm = tf.nn.l2_normalize(embeddings, 0, 1e-10,name="norm_embeddings")
        weights = tf.get_variable(name='embedding_weights', shape=(embeddings.get_shape().as_list()[-1], nrof_classes),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.nn.l2_normalize(weights, 0, 1e-10, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embeddings_norm, weights_norm, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=nrof_classes, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        logits = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='LargeMarginCosine_loss_output')
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")
    return logits,loss


def cal_loss_test( ):
    tfe.enable_eager_execution()
    embedding=tf.ones((3,3),dtype=tf.float32)
    labels=np.arange(0,3,1)
    nrof_classes=3
    w_init_method=tf.random_normal_initializer(seed=666)
    loss=cal_loss(embedding, labels, nrof_classes,w_init=w_init_method)
    print loss
if __name__ == "__main__":
    cal_loss_test()
