import tensorflow as tf
import tensorflow.contrib.slim as slim



def inference(inputs,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope='vgg_a'):
    end_points=[]
    with tf.variable_scope(scope, 'vgg_a', [inputs]):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout7')

            return net, end_points
