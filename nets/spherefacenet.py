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

def inference(images,keep_probability=0.8,phase_train=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=512,num_layers=20,reuse=None):
    units,filters=layer_setup(num_layers)
    end_poins={}
    body = images
    with tf.variable_scope('inference', [images], reuse=reuse):
        with slim.arg_scope([slim.conv2d],activation_fn=tflearn.prelu,
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(0.1) ):

            for i in xrange(len(units)):
                f = filters[i]

                body = slim.conv2d(body,f, [3, 3], stride=2,scope= "conv%d_%d"%(i+1, 1))
                print body
                idx = 2
                for j in xrange(units[i]):
                    _body = slim.conv2d(body,f, [3, 3], stride=1,scope= "conv%d_%d"%(i+1, idx))
                    print _body
                    idx+=1
                    _body = slim.conv2d(_body, f, [3, 3], stride=1,scope= "conv%d_%d"%(i+1, idx))
                    print _body
                    idx+=1
                    body = body+_body
            body=slim.flatten(body)
            print body
            body = slim.fully_connected(body, bottleneck_layer_size,scope='Bottleneck', reuse=False)
            print body
            return body,end_poins