import tensorflow as tf
import tensorflow.contrib.slim as slim


def _depthwise_separable_conv(inputs,num_pwc_filters,width_multiplier,sc,downsample=False):
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

  # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                num_outputs=None,
                                                stride=_stride,
                                                depth_multiplier=1,
                                                kernel_size=[3, 3],
                                                scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                      num_pwc_filters,
                                      kernel_size=[1, 1],
                                      scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn



def inference(images,
              keep_probability=0,
              phase_train=True,
              bottleneck_layer_size=128,
              weight_decay=0.0,
              reuse=None,
              width_multiplier=1):
    end_points={}
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=None):
        with tf.variable_scope("mobilenet",[images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm],
                          is_training=phase_train,
                          activation_fn=tf.nn.relu,
                          fused=True):
                net = slim.convolution2d(images, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')

                net = slim.avg_pool2d(net ,net.get_shape()[1:3], scope='avg_pool_15')

                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                logits = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='fc_16')

  return logits, end_points
