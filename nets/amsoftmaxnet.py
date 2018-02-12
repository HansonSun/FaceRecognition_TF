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
                    weights_regularizer=slim.l2_regularizer(0.1),activation_fn=None ):
        #conv1
        net_conv_1_1=slim.conv2d(image,64,[3,3],stride=2,scope="conv1_1",padding="VALID")
        net=slim.conv2d(net_conv_1_1,64,[3,3],stride=1,scope="conv1_2",padding="VALID")
        net=slim.conv2d(net,64,[3,3],stride=1,scope="conv1_3",padding="VALID")

        tf.concat(net_conv_1_1,net)

        #conv2
        net_conv_2_1=slim.conv2d(net,128,[3,3],stride=2,scope="conv2_1",padding="SAME")
        net=slim.conv2d(net_conv_2_1,128,[3,3],stride=1,scope="conv2_2",padding="SAME")
        net=slim.conv2d(net,128,[3,3],stride=1,scope="conv2_3",padding="SAME")
        net_res_3_3=tf.concat(net_conv_2_1,net)

        net=slim.conv2d(net,128,[3,3],stride=1,scope="conv2_4",padding="SAME")
        net=slim.conv2d(net,128,[3,3],stride=1,scope="conv2_5",padding="SAME")
        #MFM2
        tf.concat(net_res_3_3,net)

        #conv3
        conv3_1=slim.conv2d(res_2_5,256,[3,3],stride=2,scope="conv3_1",padding="SAME")
        net=slim.conv2d(conv3_1,256,[3,3],stride=1,scope="conv3_2",padding="SAME")
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_3",padding="SAME")

        res3_3=tf.concat(conv3_1,net)

        net=slim.conv2d(res3_3,256,[3,3],stride=1,scope="conv3_4",padding="SAME")

        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_5",padding="SAME")
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_6",padding="SAME")
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_7",padding="SAME")
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_8",padding="SAME")
        net=slim.conv2d(net,256,[3,3],stride=1,scope="conv3_9",padding="SAME")

        #conv4
        net=slim.conv2d(net,512,[3,3],stride=2,scope="conv4_1")
        net=slim.conv2d(net,512,[3,3],stride=1,scope="conv4_2")
        net=slim.conv2d(net,512,[3,3],stride=1,scope="conv4_3")

        #fc1
        net=slim.flatten(net)
        net=slim.fully_connected(net,512,activation_fn=None)
        #MFM_sfc1
        net=MFM(net)

    return net