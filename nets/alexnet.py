import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(inputs,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope='alexnet_v2',weight_decay=0.0):

    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],activation_fn=None,
            weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            outputs_collections=[end_points_collection] ):

            net = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID',scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points[sc.name + '/fc8'] = net
            return net, end_points
'''
            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                biases_initializer=tf.zeros_initializer(),
                                scope='fc8')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
              net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
              end_points[sc.name + '/fc8'] = net
'''

