from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride==1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,padding='SAME', scope=scope)
    else:
        output = slim.conv2d(inputs, num_outputs, kernel_size, stride=1,padding='SAME',scope=scope)
        return slim.max_pool2d(output, [1, 1], stride=stride)

def residual_unit(data,num_filter, stride, dim_match, name, **kwargs):
    use_se = kwargs.get('version_se', 1)
    bn1 = slim.batch_norm(data, scope=name + '_bn1')
    conv1 = slim.conv2d(bn1,num_filter, [3,3], stride=(1,1),scope=name + '_conv1')
    bn2 = slim.batch_norm(conv1, scope=name + '_bn2')
    act1 = tflearn.prelu(bn2)
    conv2 = conv2d_same(act1, num_filter, 3, stride=stride,scope=name + '_conv2')
    bn3 = slim.batch_norm(conv2, scope=name + '_bn3')
    if use_se:
        #se begin
        #body = tflearn.global_avg_pool (bn3,name=name+'_se_pool1')
        body=slim.avg_pool2d(bn3,kernel_size=[int(bn3.shape[1]),int(bn3.shape[2])])
        body = slim.conv2d(body,num_filter//16, [1,1], stride=(1,1),scope=name+"_se_conv1")
        body = tflearn.prelu(body)
        body = slim.conv2d(body,num_filter, [1,1], stride=(1,1),scope=name+"_se_conv2")
        body = tf.sigmoid(body)
        bn3 = tf.multiply(bn3, body)
        #se end

    if dim_match:
        shortcut = data
    else:
        conv1sc = slim.conv2d(data,num_filter, [1,1], stride=stride,padding="VALID",scope=name+'_conv1sc')
        shortcut = slim.batch_norm(conv1sc, scope=name + '_sc')

    return bn3 + shortcut

def resnet(images,units,is_training, num_stages, filter_list, bottleneck_layer_size,**kwargs):
    with slim.arg_scope([slim.batch_norm],
          updates_collections=None,
          variables_collections= [ tf.GraphKeys.TRAINABLE_VARIABLES ],
          is_training=is_training,
          center=True,
          scale=True, 
          epsilon=2e-5, 
          decay=0.9):

        body = images
        body = slim.conv2d(body, filter_list[0], [3,3], stride=(1,1),padding="SAME",scope="conv0")
        body = slim.batch_norm(body,scope='bn0')
        body = tflearn.prelu(body)
        
        for i in range(num_stages):
            body = residual_unit(body, filter_list[i+1], (2, 2), False,name='stage%d_unit%d' % (i + 1, 1), **kwargs)
            for j in range(units[i]-1):
                body = residual_unit(body,filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i+1, j+2), **kwargs)
            
    body=slim.batch_norm(body,scope='bn1')
    print (body)
    body = slim.dropout(body,0.6, is_training=is_training,scope='Dropout')
    #body=slim.flatten(body)
    #body = slim.fully_connected(body, 512, activation_fn=None,scope='pre_fc1')

    net = slim.conv2d(body, bottleneck_layer_size, 7,activation_fn=None,padding="VALID",stride=1,scope='fc1')
    net = tf.squeeze(net, [1, 2], name='Bottleneck')
    body = slim.batch_norm(body,scope='fcend')
    body=tf.identity(body,name="output")
    return body

def inference(images,is_training=True,num_layers=50,bottleneck_layer_size=512, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
         weights_initializer=slim.initializers.xavier_initializer(),
         weights_regularizer=slim.l2_regularizer(5e-5),
         biases_initializer=None,
         activation_fn=None):

        end_points={}
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 49:
            units = [3, 4, 14, 3]
        elif num_layers == 50:
            units = [3, 4, 14, 3]
        elif num_layers == 74:
            units = [3, 6, 24, 3]
        elif num_layers == 90:
            units = [3, 8, 30, 3]
        elif num_layers == 100:
            units = [3, 13, 30, 3]
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

        return resnet(images,
          units=units,
          is_training=is_training,
          num_stages=4,
          filter_list=[64, 64, 128, 256, 512],
          bottleneck_layer_size=bottleneck_layer_size,
          **kwargs),end_points


if __name__=="__main__":
    tf.set_random_seed(1234)
    inputs=tf.random_normal((1,112,112,3))
    output=inference(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result,_=sess.run(output)
        print (result.shape)

        output_graph_def=graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),["output"])
        model_f=tf.gfile.GFile("model.pb","wb")
        model_f.write(output_graph_def.SerializeToString())
