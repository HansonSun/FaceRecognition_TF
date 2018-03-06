from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn


def get_fc1(last_conv, bottleneck_layer_size, fc_type,**kwargs):
  bn_mom = 0.9
  body = last_conv
  is_training=kwargs.get('phase_train',True)

  with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
      is_training=is_training):

    if fc_type=='E':
      body = slim.batch_norm(body, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope='bn1')
      body = slim.dropout(body, keep_prob=0.4,is_training=is_training)
      fc1 = slim.fully_connected(body, bottleneck_layer_size, scope='pre_fc1')
      fc1 = slim.batch_norm(fc1, center=False,scale=False ,epsilon=2e-5, decay=bn_mom, scope='fc1')
    elif fc_type=='F':
      body = slim.batch_norm(body, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope='bn1')
      body = slim.dropout(body, keep_prob=0.4,is_training=is_training)
      fc1 = slim.fully_connected(body, bottleneck_layer_size, scope='fc1')
    elif fc_type=='G':
      body = slim.batch_norm(body, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope='bn1')
      fc1 = slim.fully_connected(body, bottleneck_layer_size, scope='fc1')
    elif fc_type=='H':
      fc1 = slim.fully_connected(body, bottleneck_layer_size, scope='fc1')
    elif fc_type=='I':
      body = slim.batch_norm(body, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope='bn1')
      fc1 = slim.fully_connected(body, bottleneck_layer_size, scope='pre_fc1')
      fc1 = slim.batch_norm(fc1, center=False,scale=False, epsilon=2e-5, decay=bn_mom, scope='fc1')
    elif fc_type=='J':
      fc1 = mslim.fully_connected(body, bottleneck_layer_size, scope='pre_fc1')
      fc1 = slim.batch_norm(fc1, center=False,scale=False, epsilon=2e-5, decay=bn_mom, scope='fc1')
    else:
      bn1 = slim.batch_norm(body, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope='bn1')
      relu1 = tflearn.prelu(bn1)
      # Although kernel is not used here when global_pool=True, we should put one
      #pool1 = tflearn.global_avg_pool (relu1,  name='pool1')
      pool1=slim.avg_pool2d(relu1,kernel_size=[int(relu1.shape[1]),int(relu1.shape[2])])
      flat = slim.flatten(pool1)
      if len(fc_type)>1:
        if fc_type[1]=='X':
          print('dropout mode')
          flat = slim.dropout(flat, keep_prob=0.2,is_training=is_training)
        fc_type = fc_type[0]
      if fc_type=='A':
        fc1 = flat
      else:
        #B-D
        #B
        fc1 = slim.fully_connected(flat, num_classes, name='pre_fc1')
        if fc_type=='C':
          fc1 = slim.batch_norm(fc1, center=False,scale=False, epsilon=2e-5, decay=bn_mom, scope='fc1')
        elif fc_type=='D':
          fc1 = slim.batch_norm(fc1,center=False,scale=False, epsilon=2e-5, decay=bn_mom, scope='fc1')
          fc1 = tflearn.prelu(fc1)
    return fc1

def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)

    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
      is_training=is_training):
    #print('in unit1')
      if bottle_neck:
          conv1 = slim.conv2d(data, int(num_filter*0.25), [1,1], stride=stride, padding="VALID",scope=name + '_conv1')
          bn1 = slim.batch_norm(conv1, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv2 = slim.conv2d(act1,  int(num_filter*0.25), [3,3], stride=1, padding="SAME",scope=name + '_conv2')
          bn2 = slim.batch_norm(conv2, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act2 = tflearn.prelu(bn2)
          conv3 = slim.conv2d(act2, num_filter, [1,1], stride=1, padding="VALID",scope=name + '_conv3')
          bn3 = slim.batch_norm(conv3, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')

          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn3,  name=name+'_se_pool1')
            body=slim.avg_pool2d(bn3,kernel_size=[int(bn3.shape[1]),int(bn3.shape[2])])
            body = slim.conv2d(body,  num_filter//16, [1,1],stride=1,padding="VALID", scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1],stride=1,padding="VALID",scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn3 = slim.multiply(bn3, body)
            #se end

          if dim_match:
            shortcut = data
          else:
            conv1sc = slim.conv2d(data,  num_filter, [1,1], stride=stride,biases_initializer=None,scope=name+'_conv1sc')
            shortcut = slim.batch_norm(conv1sc, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_sc')

            return tflearn.prelu(bn3 + shortcut)
      else:
          conv1 = slim.conv2d(data,  num_filter, [3,3], stride=stride,biases_initializer=None,scope=name + '_conv1')
          bn1 = slim.batch_norm(conv1, center=True,scale=True,decay=bn_mom, epsilon=2e-5, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv2 = slim.conv2d(act1, num_filter, [3,3], stride=1,biases_initializer=None, scope=name + '_conv2')
          bn2 = slim.batch_norm(conv2,center=True,scale=True,  decay=bn_mom, epsilon=2e-5, scope=name + '_bn2')
          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn2,name=name+'_se_pool1')
            body=slim.avg_pool2d(bn2,kernel_size=[int(bn2.shape[1]),int(bn2.shape[2])])
            body = slim.conv2d(body,num_filter//16, [1,1], stride=1,scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter,[1,1], stride=1,scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn2 = tf.multiply(bn2, body)
            #se end

          if dim_match:
              shortcut = data
          else:
              conv1sc = slim.conv2d(data, num_filter,[1,1],biases_initializer=None, stride=stride,scope=name+'_conv1sc')
              shortcut = slim.batch_norm(conv1sc, center=True,scale=True,decay=bn_mom, epsilon=2e-5, scope=name + '_sc')
          
          output=tf.concat([bn2 ,shortcut],3)
          return tflearn.prelu(output)

def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)
    #print('in unit1')
    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
      is_training=is_training):

      if bottle_neck:
          conv1 = slim.conv2d(data, int(num_filter*0.25),[1,1], stride=1,biases_initializer=None,scope=name + '_conv1')
          bn1 = slim.batch_norm(conv1, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv2 = slim.conv2d(act1, int(num_filter*0.25), [3,3], stride=1,biases_initializer=None,scope=name + '_conv2')
          bn2 = slim.batch_norm(conv2, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act2 = tflearn.prelu(bn2)
          conv3 = slim.conv2d(act2, num_filter, [1,1], stride=stride,biases_initializer=None,scope=name + '_conv3')
          bn3 = slim.batch_norm(conv3, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')

          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn3,name=name+'_se_pool1')
            body=slim.avg_pool2d(bn3,kernel_size=[int(bn3.shape[1]),int(bn3.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=1,scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body,num_filter,[1,1], stride=1,scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn3 = tf.multiply(bn3, body)
            #se end

          if dim_match:
              shortcut = data
          else:
              conv1sc = slim.conv2d(data, num_filter, [1,1], stride=stride,biases_initializer=None,scope=name+'_conv1sc')
              shortcut = slim.batch_norm(conv1sc,center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_sc')

          return tflearn.prelu(bn3 + shortcut)
      else:
          conv1 = slim.conv2d(data,num_filter, [3,3], stride=(1,1), biases_initializer=None,scope=name + '_conv1')
          bn1 = slim.batch_norm(conv1,center=True,scale=True,  decay=bn_mom, epsilon=2e-5, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv2 = slim.conv2d(act1,num_filter,[3,3], stride=stride,biases_initializer=None,scope=name + '_conv2')
          bn2 = slim.batch_norm(conv2,center=True,scale=True, decay=bn_mom, epsilon=2e-5, scope=name + '_bn2')
          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn2, name=name+'_se_pool1')
            body=slim.avg_pool2d(bn2,kernel_size=[int(bn2.shape[1]),int(bn2.shape[2])])
            body = slim.conv2d(body,num_filter//16, [1,1], stride=(1,1),scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1], stride=(1,1),scope=name+"_se_conv2")
            body = tf.sigmoid(body )
            bn2 = tf.multiply(bn2, body)
            #se end

          if dim_match:
              shortcut = data
          else:
              conv1sc = slim.conv2d(data, num_filter,[1,1], stride=stride,biases_initializer=None,scope=name+'_conv1sc')
              shortcut =slim.batch_norm(conv1sc, center=True,scale=True,decay=bn_mom, epsilon=2e-5, scope=name + '_sc')

          return tflearn.prelu(bn2 + shortcut )

def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)
    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
      is_training=is_training):
      #print('in unit2')
      if bottle_neck:
          # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
          bn1 = slim.batch_norm(data, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv1 = slim.conv2d(act1, int(num_filter*0.25), [1,1], stride=(1,1),biases_initializer=None,scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act2 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act2, int(num_filter*0.25), [3,3], stride=stride,biases_initializer=None,scope=name + '_conv2')
          bn3 = slim.batch_norm(conv2,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
          act3 = tflearn.prelu(bn3)
          conv3 = slim.conv2d(act3, num_filter,[1,1], stride=(1,1),biases_initializer=None,scope=name + '_conv3')
          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (conv3, name=name+'_se_pool1')
            body=slim.avg_pool2d(conv3,kernel_size=[int(conv3.shape[1]),int(conv3.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=(1,1),scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1], stride=(1,1),scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            conv3 = tf.multiply(conv3, body)
          if dim_match:
              shortcut = data
          else:
              shortcut = slim.conv2d(act1, num_filter,[1,1], stride=stride,biases_initializer=None,scope=name+'_sc')

          return conv3 + shortcut
      else:
          bn1 = slim.batch_norm(data,center=True,scale=True,  decay=bn_mom, epsilon=2e-5, scope=name + '_bn1')
          act1 = tflearn.prelu(bn1)
          conv1 = slim.conv2d(act1, num_filter, [3,3], stride=stride,biases_initializer=None,scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1, center=True,scale=True,decay=bn_mom, epsilon=2e-5, scope=name + '_bn2')
          act2 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act2, num_filter, [3,3],stride=1,biases_initializer=None,scope=name + '_conv2')
          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (conv2, name=name+'_se_pool1')
            body=slim.avg_pool2d(conv2,kernel_size=[int(conv2.shape[1]),int(conv2.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=(1,1), name=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1], stride=(1,1), scope=name+"_se_conv2")
            body = slim.sigmoid(body)
            conv2 = tf.multiply(conv2, body)
          if dim_match:
              shortcut = data
          else:
              shortcut = slim.conv2d(act1, num_filter, [1,1], stride=stride, biases_initializer=None,scope=name+'_sc')
          
          output=tf.concat([conv2,shortcut],3)
          return output

def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    use_se = kwargs.get('version_se', 0)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)
    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ],
      is_training=is_training):
      #print('in unit3')
      if bottle_neck:
          bn1 = slim.batch_norm(data,center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          conv1 = slim.conv2d(bn1, int(num_filter*0.25), [1,1], stride=(1,1),biases_initializer=None,padding="VALID",scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act1 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act1, int(num_filter*0.25), [3,3], stride=(1,1),biases_initializer=None,padding="SAME",scope=name + '_conv2')
          bn3 = slim.batch_norm(conv2, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
          act2 = tflearn.prelu(bn3)
          conv3 = slim.conv2d(act2, num_filter, [1,1], stride=stride, biases_initializer=None,padding="VALID",scope=name + '_conv3')
          bn4 = slim.batch_norm(conv3,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn4')

          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn4, name=name+'_se_pool1')
            body=slim.avg_pool2d(bn4,kernel_size=[int(bn4.shape[1]),int(bn4.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=(1,1), scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1], stride=(1,1), scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn4 = tf.multiply(bn4, body)
            #se end

          if dim_match:
              shortcut = data
          else:
              conv1sc = slim.conv2d(data,num_filter, [1,1], stride=stride,biases_initializer=None,padding="VALID",scope=name+'_conv1sc')
              shortcut = slim.batch_norm(conv1sc,center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_sc')

          return bn4 + shortcut
      else:
          bn1 = slim.batch_norm(data, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          conv1 = slim.conv2d(bn1,num_filter, [3,3], stride=(1,1),biases_initializer=None,scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act1 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act1, num_filter, [3,3], stride=stride, biases_initializer=None,scope=name + '_conv2')
          bn3 = slim.batch_norm(conv2, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
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
              conv1sc = slim.conv2d(data,num_filter, [1,1], stride=stride,biases_initializer=None,padding="VALID",scope=name+'_conv1sc')
              shortcut = slim.batch_norm(conv1sc,center=True,scale=True,  decay=bn_mom, epsilon=2e-5, scope=name + '_sc')

          return bn3 + shortcut

def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    print (data.shape)
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)

    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
      is_training=is_training):

      num_group = 32
      #print('in unit3')
      bn1 = slim.batch_norm(data, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
      #conv1 = slim.conv2d(bn1, int(num_filter*0.5), [1,1],padding="VALID",biases_initializer=None,scope=name + '_conv1')
      conv1 = tflearn.layers.conv.grouped_conv_2d(bn1, num_group, int(num_filter*0.5), kernel=(3,3), stride=(1,1),biases_initializer=None,scope=name + '_conv2')
      
      bn2 = slim.batch_norm(conv1, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
      act1 = tflearn.prelu(bn2)
      conv2 = slim.conv2d(act1, int(num_filter*0.5), [3,3],biases_initializer=None,scope=name + '_conv2')
      #conv2 = tfleaen.grouped_conv_2d(act1, num_group, int(num_filter*0.5), kernel=(3,3), stride=(1,1),biases_initializer=None,scope=name + '_conv2')
      
      bn3 = slim.batch_norm(conv2,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
      act2 = tflearn.prelu(bn3)
      conv3 = slim.conv2d(act2, num_filter,[1,1], stride=stride,padding="VALID",scope=name + '_conv3')
      bn4 = slim.batch_norm(conv3, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_bn4')
      print (bn4.shape)
      if use_se:
        #se begin
        #2 choice body = tflearn.global_avg_pool (bn4, name=name+'_se_pool1')
        body=slim.avg_pool2d(bn4,kernel_size=[int(bn4.shape[1]),int(bn4.shape[2])])
        print ("-->",body.shape)
        body = slim.conv2d(body, num_filter//16, [1,1],scope=name+"_se_conv1")
        body = tflearn.prelu(body)
        body = slim.conv2d(body,num_filter, [1,1], stride=(1,1),scope=name+"_se_conv2")
        body = tf.sigmoid(body)
        bn4 = tf.multiply(bn4, body)
        #se end

      if dim_match:
          shortcut = data
      else:
          conv1sc = slim.conv2d(data, num_filter, [1,1], stride=stride,biases_initializer=None,scope=name+'_conv1sc')
          shortcut = slim.batch_norm(conv1sc, center=True,scale=True,epsilon=2e-5, decay=bn_mom, scope=name + '_sc')

      print (bn4.shape )
      print (shortcut.shape )
      return bn4 + shortcut

def residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    is_training=kwargs.get('phase_train',True)
    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
      is_training=is_training):
      #print('in unit3')
      if bottle_neck:
          bn1 = slim.batch_norm(data, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          conv1 = slim.conv2d(bn1, int(num_filter*0.25), [1,1], stride=(1,1),biases_initializer=None,scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act1 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act1, int(num_filter*0.25), [3,3], stride=(1,1),biases_initializer=None,scope=name + '_conv2')
          bn3 = slim.batch_norm(conv2, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
          act2 = tflearn.prelu(bn3)
          conv3 = slim.conv2d(act2, num_filter, [1,1], stride=stride,biases_initializer=None,scope=name + '_conv3')
          bn4 = slim.batch_norm(conv3,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn4')

          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn4, name=name+'_se_pool1')
            body=slim.avg_pool2d(bn4,kernel_size=[int(bn4.shape[1]),int(bn4.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=(1,1),scope=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body, num_filter, [1,1], stride=(1,1),scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn4 = tf.multiply(bn4, body)
            #se end
          if dim_match:
              shortcut = data

              return bn4+shortcut
          else:
            return bn4

      else:
          bn1 = slim.batch_norm(data,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn1')
          conv1 = slim.conv2d(bn1, num_filter, [3,3], stride=(1,1),biases_initializer=None,scope=name + '_conv1')
          bn2 = slim.batch_norm(conv1,center=True,scale=True,  epsilon=2e-5, decay=bn_mom, scope=name + '_bn2')
          act1 = tflearn.prelu(bn2)
          conv2 = slim.conv2d(act1,num_filter, [3,3], stride=stride,biases_initializer=None,scope=name + '_conv2')
          bn3 = slim.batch_norm(conv2, center=True,scale=True, epsilon=2e-5, decay=bn_mom, scope=name + '_bn3')
          if use_se:
            #se begin
            #body = tflearn.global_avg_pool (bn3,name=name+'_se_pool1')
            body=slim.avg_pool2d(bn3,kernel_size=[int(bn3.shape[1]),int(bn3.shape[2])])
            body = slim.conv2d(body, num_filter//16, [1,1], stride=(1,1),scoep=name+"_se_conv1")
            body = tflearn.prelu(body)
            body = slim.conv2d(body,num_filter,[1,1], stride=(1,1),scope=name+"_se_conv2")
            body = tf.sigmoid(body)
            bn3 = tf.multiply(bn3, body)
            #se end

          if dim_match:
              shortcut = data

              return bn3+shortcut
          else:
           return bn3

def residual_unit(data, num_filter, stride, dim_match, name,bottle_neck, **kwargs):
  uv = kwargs.get('version_unit', 3)
  version_input = kwargs.get('version_input', 1)
  if uv==1:
    if version_input==0:
      return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==2:
    return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==4:
    return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  else:
    if version_input<=1:
      return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)

def resnet(images,units,num_stages, filter_list, bottle_neck, bottleneck_layer_size,**kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    print(version_se, version_input, version_output, version_unit)
    num_unit = len(units)
    is_training=kwargs.get("is_training",True)
    assert(num_unit == num_stages)
    with slim.arg_scope([slim.batch_norm],
      updates_collections=None,
      variables_collections= [ tf.GraphKeys.TRAINABLE_VARIABLES ]):

      if version_input==0:
        body = slim.conv2d(images, filter_list[0], [7, 7], stride=(2,2),biases_initializer=None,scope="conv0")
        body = slim.batch_norm(body,center=True,scale=True,epsilon=2e-5,decay=bn_mom,scope='bn0')
        body = tflearn.prelu(body)
        body = slim.max_pool2d(body, kernel_size=[3, 3], stride=(2,2),padding="SAME")
      else:
        body = images
        body = slim.conv2d(body, filter_list[0], [3,3], stride=(1,1),biases_initializer=None,scope="conv0")
        body = slim.batch_norm(body,center=True,scale=True,epsilon=2e-5,decay=bn_mom,scope='bn0')
        body = tflearn.prelu(body)

    
    for i in range(num_stages):
      if version_input==0:
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      else:
        body = residual_unit(body,filter_list[i+1],(2, 2),False,
          name='stage%d_unit%d'%(i+1,1),bottle_neck=bottle_neck,**kwargs)
      for j in range(units[i]-1):
        body = residual_unit(body,filter_list[i+1],(1,1),True,
          name='stage%d_unit%d'%(i+1,j+2),bottle_neck=bottle_neck,**kwargs)


    fc1 = get_fc1(body, bottleneck_layer_size, fc_type)
    return fc1

def inference(images,phase_train=True,num_layers=50,bottleneck_layer_size=256, **kwargs):
    end_points={}
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
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
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(images,
      units=units,
      num_stages=num_stages,
      filter_list=filter_list,
      bottle_neck=bottle_neck,
      bottleneck_layer_size=bottleneck_layer_size,
      **kwargs),end_points
