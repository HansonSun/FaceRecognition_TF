import tflearn
import tensorflow as tf
import tensorflow.contrib.slim as slim

def layer_setup():
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

def inference(images,is_training=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=256,num_layers=20):
    units,filters=layer_setup()
    end_poins={}
    with slim.arg_scope([slim.conv2d],activation_fn=tflearn.prelu,
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):

    for i in xrange(len(units)):
        f = filters[i]

        body = slim.conv2d(body,f, [3, 3], stride=2,scope= "conv%d_%d"%(i+1, 1),)
        body = tflearn.prelu(body)
        idx = 2
        for j in xrange(units[i]):
            _body = slim.conv2d(body,f, [3, 3], stride=1,scope= "conv%d_%d"%(i+1, idx))
            idx+=1
            _body = slim.conv2d(_body, f, [3, 3], stride=1,scope= "conv%d_%d"%(i+1, idx))
            idx+=1
            body = tf.concat([body,_body],3)

    return body,end_poins

'''
def inference(images,is_training=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=256):
    end_poins={}
    with slim.arg_scope([slim.conv2d],activation_fn=tflearn.prelu,
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):

        conv1_1=slim.conv2d(images,64,[3,3],stride=2,scope="conv1_1",padding="VALID")
        conv1_2=slim.conv2d(conv1_1,64,[3,3],stride=1,scope="conv1_2")
        conv1_3=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_3")
        res1_3=tf.concat([conv1_1,conv1_3],3)

        conv2_1=slim.conv2d(res1_3,128,[3,3],stride=2,scope="conv2_1")
        conv2_2=slim.conv2d(conv2_1,128,[3,3],stride=1,scope="conv2_2")
        conv2_3=slim.conv2d(conv2_2,128,[3,3],stride=1,scope="conv2_3")
        res2_3=tf.concat([conv2_1,conv2_3],3)

        conv2_4=slim.conv2d(res2_3,128,[3,3],stride=1,scope="conv2_4")
        conv2_5=slim.conv2d(conv2_4,128,[3,3],stride=1,scope="conv2_5")
        res2_5=tf.concat([res2_3,conv2_5],3)
        
        conv3_1=slim.conv2d(res2_5,256,[3,3],stride=2,scope="conv3_1")
        conv3_2=slim.conv2d(conv3_1,256,[3,3],stride=1,scope="conv3_2")
        conv3_3=slim.conv2d(conv3_2,256,[3,3],stride=1,scope="conv3_3")
        res3_3=tf.concat([conv3_1,conv3_3],3)


        conv3_4=slim.conv2d(res3_3,256,[3,3],stride=1,scope="conv3_4")
        conv3_5=slim.conv2d(conv3_4,256,[3,3],stride=1,scope="conv3_5")
        res3_5=tf.concat([res3_3,conv3_5],3)

        conv3_6=slim.conv2d(res3_5,256,[3,3],stride=1,scope="conv3_6")
        conv3_7=slim.conv2d(conv3_6,256,[3,3],stride=1,scope="conv3_7")
        res3_7=tf.concat([res3_5,conv3_7],3)

        conv3_8=slim.conv2d(res3_7,256,[3,3],stride=1,scope="conv3_8")
        conv3_9=slim.conv2d(conv3_8,256,[3,3],stride=1,scope="conv3_9")
        res3_9=tf.concat([res3_7,conv3_9],3)

        conv4_1=slim.conv2d(res3_9,512,[3,3],stride=2,scope="conv4_1")
        conv4_2=slim.conv2d(conv4_1,512,[3,3],stride=1,scope="conv4_2")
        conv4_3=slim.conv2d(conv4_2,512,[3,3],stride=1,scope="conv4_3")
        res4_3=tf.concat([conv4_3,conv4_1],3)
        res4_3=slim.flatten(res4_3)

        features=slim.fully_connected(res4_3,bottleneck_layer_size,activation_fn=None,scope="fc1")
        return features,end_poi

'''