import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(inputs,
              feature_length=128,
              phase_train=True,
              dropout_keep_prob=0.5,
              weight_decay=5e-5,
              scope='vgg_a',
              w_init=slim.xavier_initializer_conv2d(uniform=True)):

    with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=w_init,
                          weights_regularizer=slim.l2_regularizer(weight_decay)):
        
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.flatten(net)
            net=slim.fully_connected(net,feature_length,activation_fn=None,scope='Bottleneck', reuse=False)
            return net
