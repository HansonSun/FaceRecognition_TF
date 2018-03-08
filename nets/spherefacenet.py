import tflearn
import tensorflow as tf
import tensorflow.contrib.slim as slim

def layer_setup(num_layers):
    if num_layers==64:
        units = [3,8,16,3]
        filters = [64,128,256,512]
    elif num_layers==20:
        units = [1,2,4,1]
        filters = [64,128,256,512]
        #filters = [64, 256, 512, 1024]
    elif num_layers==36:
        units = [2,4,8,2]
        filters = [64,128,256,512]
        #filters = [64, 256, 512, 1024]
    elif num_layers==60:
        units = [3,8,14,3]
        filters = [64,128,256,512]
    elif num_layers==104:
        units = [3,8,36,3]
        filters = [64,128,256,512]
    return units,filters

def inference(images,keep_probability,phase_train=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=256,num_layers=20,reuse=None):
    units,filters=layer_setup(num_layers)
    end_poins={}
    body = images
    with tf.variable_scope('inference', [images], reuse=reuse):
        with slim.arg_scope([slim.conv2d],activation_fn=tflearn.prelu,
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(0.1) ):

            for i in xrange(len(units)):
                f = filters[i]

                body = slim.conv2d(body,f, [3, 3], stride=2,biases_initializer=None,scope= "conv%d_%d"%(i+1, 1),)
                idx = 2
                for j in xrange(units[i]):
                    _body = slim.conv2d(body,f, [3, 3], stride=1,biases_initializer=None,scope= "conv%d_%d"%(i+1, idx))
                    idx+=1
                    _body = slim.conv2d(_body, f, [3, 3], stride=1,biases_initializer=None,scope= "conv%d_%d"%(i+1, idx))
                    idx+=1
                    body = body+_body
            body=slim.flatten(body)
            body = slim.fully_connected(body, bottleneck_layer_size, activation_fn=None,scope='Bottleneck', reuse=False)
            return body,end_poins