import tensorflow as tf
import tensorflow.contrib.slim as slim

def MFM(net):
    net_channel=int (net.shape[-1]) 
    #Slice
    slice1,slice2 = tf.split(net,[net_channel/2,net_channel/2],int(net.shape.ndims)-1 )
    #eltwise max
    eltwise=tf.maximum(slice1,slice2)
    return eltwise

def inference(image):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):
        #conv1
        net=slim.conv2d(image,96,[5,5],stride=1,scope="conv1",padding="SAME")
        #MFM1
        net=MFM(net)
        #pool1
        net=slim.max_pool2d(net,[2,2],stride=2,scope="pool1")
        #conv2_X
        net_conv2_x=slim.repeat(net,2,slim.conv2d,48,[3,3],scope="conv2_x",padding="SAME")
        net = tf.concat([net, net_conv2_x], 2)
        #conv2a
        net=slim.conv2d(net,96,[1,1],stride=1,scope="conv2a",padding="SAME")
        #MFM2a
        net=MFM(net)
        #conv2
        net=slim.conv2d(net,192,[3,3],stride=1,scope="conv2",padding="SAME")
        #MFM2
        net=MFM(net)
        #pool2
        net=slim.max_pool2d(net,[2,2],stride=2,scope="pool2")
        #conv3_x
        net_conv3_x=slim.repeat(net,4,slim.conv2d,96,[3,3],scope="conv3_x",padding="SAME")
        net = tf.concat([net, net_conv3_x], 2)
        #conv3a
        net=slim.conv2d(net,192,[1,1],stride=1,scope="conv3a",padding="SAME")
        #MFM3a
        net=MFM(net)
        #conv3
        net=slim.conv2d(net,384,[3,3],stride=1,scope="conv3",padding="SAME")
        #MFM3
        net=MFM(net)
        #pool3
        net=slim.max_pool2d(net,[2,2],stride=2,scope="pool3")
        #conv4_x
        net_conv4_x=slim.repeat(net,6,slim.conv2d,192,[3,3],scope="conv4_x",padding="SAME")
        net = tf.concat([net, net_conv4_x], 2)
        #conv4a
        net=slim.conv2d(net,384,[1,1],stride=1,scope="conv4a")
        #MFM4a
        net=MFM(net)
        #conv4
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv4")
        #MFMF4
        net=MFM(net)
        #conv5_x
        net_conv5_x=slim.repeat(net,8,slim.conv2d,128,[3,3],scope="conv5_x",padding="SAME")
        net = tf.concat([net, net_conv5_x], 2)
        #conv5a
        net=slim.conv2d(net,256,[1,1],stride=1,scope="conv5a")
        #MFM5a
        net=MFM(net)
        #conv5
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv5")
        #MFMF5
        net=MFM(net)
        #pool4
        net=slim.max_pool2d(net,[2,2],stride=2,scope="pool4")
        #fc1
        net=slim.flatten(net)
        net=slim.fully_connected(net,512,activation_fn=None)
        #MFM_sfc1
        net=MFM(net)

    return net