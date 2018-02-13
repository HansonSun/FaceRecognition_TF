import tensorflow as tf
slim = tf.contrib.slim

def inference(images, is_training=True,dropout_keep_prob=0.5,scope='LeNet',bottleneck_layer_size=1024,weight_decay=0.0):
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
            weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
            weights_regularizer=slim.l2_regularizer(weight_decay) ):
            net = slim.conv2d(images, 20, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = slim.conv2d(net, 50, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = slim.flatten(net)
            end_points['Flatten'] = net
            net = slim.fully_connected(net, bottleneck_layer_size, scope='fc3')

    return net, end_points
