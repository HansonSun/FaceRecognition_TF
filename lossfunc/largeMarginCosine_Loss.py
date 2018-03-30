import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

def LargeMarginCosine_loss(embedding, labels, num_output, w_init=None, s=30., m=0.4):
    '''
    LargeMarginCosine loss as described in https://arxiv.org/abs/1801.09414
    'CosFace: Large Margin Cosine Loss for Deep Face Recognition'
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param num_output: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], num_output),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=num_output, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
        print output
    return output


def LargeMarginCosine_loss_test( ):
    tfe.enable_eager_execution()
    #embedding=np.ones((10,10),dtype=np.float32)*2
    #labels=np.arange(0,10,1,dtype=np.int64)
    #print labels
    #out=tf.nn.softmax(embedding)
    #print out
    #inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=embedding, labels=labels)
    #print inference_loss
    embedding=np.ones((3,3),dtype=np.float32)
    labels=np.arange(0,3,1)
    num_output=3
    #w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    w_init_method=tf.random_normal_initializer(seed=666)
    logit=LargeMarginCosine_loss(embedding, labels, num_output,w_init=w_init_method)
    out=tf.nn.softmax(logit)
    print out
    inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
    print inference_loss
if __name__ == "__main__":
    LargeMarginCosine_loss_test()
