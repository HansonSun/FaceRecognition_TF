import tensorflow as tf
import tensorflow.contrib.slim as slim

def MFM(net):
    net_channel=int (net.shape[-1]) 
    #Slice
    slice1,slice2 = tf.split(net,[net_channel/2,net_channel/2],int(net.shape.ndims)-1 )
    #eltwise max
    eltwise=tf.maximum(slice1,slice2)
    return eltwise

def block2(net):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):

        conv2_x_1=slim.conv2d(net,64,[1,1],scope="conv2_x_1",padding="SAME")
        conv2_x_2=slim.conv2d(conv2_x_1,64,[3,3],scope="conv2_x_2",padding="SAME")
        conv2_x_3=slim.conv2d(conv2_x_2,256,[1,1],scope="conv2_x_3",padding="SAME")
        net = tf.concat([net, conv2_x_3], 2)
    return net

def block3(net):
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):

        conv3_x_1=slim.conv2d(net,128,[1,1],scope="conv3_x_1",padding="VALID")
        conv3_x_2=slim.conv2d(conv3_x_1,128,[3,3],scope="conv3_x_2",padding="SAME")
        conv3_x_3=slim.conv2d(conv3_x_2,512,[1,1],scope="conv3_x_3",padding="SAME")
        net = tf.concat([net, conv3_x], 2)
    return net

def block4(net):
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):
        conv4_x_1=slim.conv2d(net,256,[1,1],scope="conv4_x_1",padding="VALID")
        conv4_x_2=slim.conv2d(conv4_x_1,256,[3,3],scope="conv4_x_2",padding="SAME")
        conv4_x_3=slim.conv2d(conv4_x_2,1024,[1,1],scope="conv4_x_3",padding="SAME")
        net = tf.concat([net, conv4_x_3], 2)
    return net

def block5(net):
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1) ):
        conv5_x_1=slim.slim(conv2d,512,[1,1],scope="conv5_x_1",padding="VALID")
        conv5_x_2=slim.slim(conv5_x_2,512,[3,3],scope="conv5_x_2",padding="SAME")
        conv5_x_3=slim.slim(conv5_x_3,2048,[1,1],scope="conv5_x_3",padding="SAME")
        net = tf.concat([net, conv5_x_3], 2)

    return net

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
        for i in range(3):
            net=block2(net)
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
        for i in range(4):
            net=block4(net)
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
        for i in range(6):
            net=block4(net)
        #conv4a
        net=slim.conv2d(net,384,[1,1],stride=1,scope="conv4a")
        #MFM4a
        net=MFM(net)
        #conv4
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv4")
        #MFMF4
        net=MFM(net)
        #conv5_x
        for i in range(3):
            net=block5(net)
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