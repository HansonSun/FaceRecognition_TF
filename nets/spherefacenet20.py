import tflearn


def inference(images,is_training=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=256):
    with slim.arg_scope([slim.conv2d],activation_fn=tflearn.prelu,
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1),activation_fn=None ):

        conv1_1=slim.conv2d(images,64,[3,3],stride=2,padding="VALID",scope="conv1_1"    )
        conv1_2=slim.conv2d(conv1_1,64,[3,3],stride=1,scope="conv1_2")
        conv1_3=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_3")
        res1_3=tf.concat([conv1_1,conv1_3],3)

        conv2_1=slim.conv2d(res1_3,128,[3,3],stride=2,scope="conv2_1",padding="VALID")
        conv2_2=slim.conv2d(con2_1,128,[3,3],stride=1,scope="conv2_2")
        conv2_3=slim.conv2d(conv2_2,128,[3,3],stride=1,scope="conv2_3")
        res2_3.tf.concat(conv2_1,conv2_3)

        conv2_4=slim.conv2d(res2_3,128,[3,3],stride=1,scope="conv2_4")
        conv2_5=slim.conv2d(conv2_4,128,[3,3],stride=1,scope="conv2_5")
        res2_5.tf.concat(res2_3,conv2_5)
        
        conv3_1=slim.conv2d(res2_5,256,[3,3],stride=2,scope="conv3_1",padding="VALID")
        conv3_2=slim.conv2d(conv3_1,256,[3,3],stride=1,scope="conv3_2")
        conv3_3=slim.conv2d(conv3_2,256,[3,3],stride=1,scope="conv3_3")
        res3_3.tf.concat(conv3_1,conv3_3)


        conv3_4=slim.conv2d(res3_3,256,[3,3],stride=1,scope="conv3_4")
        conv3_5=slim.conv2d(conv3_4,256,[3,3],stride=1,scope="conv3_5")
        res3_5=tf.concat(res3_3,conv3_5)

        conv3_6=slim.conv2d(res3_5,256,[3,3],stride=1,scope="conv3_6")
        conv3_7=slim.conv2d(conv3_6,256,[3,3],stride=1,scope="conv3_7")
        res3_7.concat(res3_5,conv3_7)

        conv3_8=slim.conv2d(res3_7,,256,[3,3],stride=1,scope="conv3_8")
        conv3_9=slim.conv2d(conv3_8,256,[3,3],stride=1,scope="conv3_9")
        res3_9.concat(res3_7,conv3_9)

        conv4_1=slim.conv2d(res3_9,512,[3,3],stride=2,scope="conv4_1",padding="VALID")
        conv4_2=slim.conv2d(conv4_1,512,[3,3],stride=1,scope="conv4_2")
        conv4_3=slim.conv2d(conv4_2,512,[3,3],stride=1,scope="conv4_3")
        res4_3.concat(con4_3,conv4_1)
        res4_3=slim.flatten(res4_3)

        features=slm.fully_connected(res4_3,bottleneck_layer_size,activation_fn=None,scope="fc1")
        return features