import tflearn


def inference(image):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                    weights_regularizer=slim.l2_regularizer(0.1),activation_fn=None ):

        conv1_1=slim.conv2d(images,64,[3,3],stride=2,padding="VALID",scope="conv1_1")
        conv1_2=slim.conv2d(conv1_1,64,[3,3],stride=1,scope="conv1_2")
        conv1_3=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_3")
        res1_3=tf.concat(conv1_1,conv1_3)

        conv1_4=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_4")
        conv1_5=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_5")
        res1_5=tf.concat(res1_3,conv1_5)

        conv1_6=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_6")
        conv1_7=slim.conv2d(conv1_2,64,[3,3],stride=1,scope="conv1_7")
        res1_7=tf.concat(res1_5,conv1_7)


        conv2_1=slim.conv2d(res1_7,128,[3,3],stride=2,scope="conv2_1",padding="VALID")
        conv2_2=slim.conv2d(con2_1,128,[3,3],stride=1,scope="conv2_2")
        conv2_3=slim.conv2d(conv2_2,128,[3,3],stride=1,scope="conv2_3")
        res2_3=tf.concat(conv2_1,conv2_3)

        conv2_4=slim.conv2d(res2_3,128,[3,3],scope="conv2_4")
        conv2_5=slim.conv2d(conv2_4,128,[3,3],scope="conv2_5")
        res2_5=tf.concat(res2_3,conv2_5)

        conv2_6=slim.conv2d(res2_5,128,[3,3],scope="conv2_6")
        conv2_7=slim.conv2d(conv2_6,128,[3,3],scope="conv2_7")
        res2_7=tf.concat(res2_5,conv2_7)

        conv2_8=slim.conv2d(res2_7,128,[3,3],scope="conv2_8")
        conv2_9=slim.conv2d(conv2_8,128,[3,3],scope="conv2_9") 
        res2_9=tf.concat(res2_7,conv2_9)

        conv2_10=slim.conv2d(res2_9,128,[3,3],scope="conv2_10") 
        conv2_11=slim.conv2d(conv2_10,128,[3,3],scope="conv2_11")
        res2_11=tf.concat(res2_9,conv2_11)

        conv2_12=slim.conv2d(res2_11,128,[3,3],scope="conv2_12")
        conv2_13=slim.conv2d(conv2_12,128,[3,3],scope="conv2_13")
        res2_13=tf.concat(res2_11,conv2_13)

        conv2_14=slim.conv2d(res2_13,128,[3,3],scope="conv2_14")
        conv2_15=slim.conv2d(conv2_14,128,[3,3],scope="conv2_15")
        res2_15=tf.concat(res2_13,conv2_15)

        
        conv3_1=slim.conv2d(res2_15,256,[3,3],stride=2,scope="conv3_1",padding="VALID")
        conv3_2=slim.conv2d(conv3_1,256,[3,3],stride=1,scope="conv3_2")
        conv3_3=slim.conv2d(conv3_2,256,[3,3],stride=1,scope="conv3_3")
        res3_3=tf.concat(conv3_1,conv3_3)

        conv3_4=slim.conv2d(res3_3,256,[3,3],stride=1,scope="conv3_4")
        conv3_5=slim.conv2d(conv3_4,256,[3,3],stride=1,scope="conv3_5")
        res3_5=tf.concat(res3_3,conv3_5)

        conv3_6=slim.conv2d(res3_5,256,[3,3],stride=1,scope="conv3_6")
        conv3_7=slim.conv2d(conv3_6,256,[3,3],stride=1,scope="conv3_7")
        res3_7=tf.concat(res3_5,conv3_7)

        conv3_8=slim.conv2d(res3_7,256,[3,3],stride=1,scope="conv3_8")
        conv3_9=slim.conv2d(conv3_8,256,[3,3],stride=1,scope="conv3_9")
        res3_9=tf.concat(res3_7,conv3_9)

        conv3_10=slim.conv2d(res3_9,256,[3,3],stride=1,scope="conv3_10")
        conv3_11=slim.conv2d(conv3_10,256,[3,3],stride=1,scope="conv3_11")
        res3_11=tf.concat(res3_9,conv3_11)

        conv3_12=slim.conv2d(res3_11,256,[3,3],stride=1,scope="conv3_12")
        conv3_13=slim.conv2d(conv3_12,256,[3,3],stride=1,scope="conv3_13")
        res3_13=tf.concat(res3_11,conv3_13)

        conv3_14=slim.conv2d(res3_13,256,[3,3],stride=1,scope="conv3_14")
        conv3_15=slim.conv2d(conv3_14,256,[3,3],stride=1,scope="conv3_15")
        res3_15=tf.concat(res3_13,conv3_15)

        conv3_16=slim.conv2d(res3_15,256,[3,3],stride=1,scope="conv3_16")
        conv3_17=slim.conv2d(conv3_16,256,[3,3],stride=1,scope="conv3_17")
        res3_17=tf.concat(res_3_15,conv3_17)

        conv3_18=slim.conv2d(res3_3,256,[3,3],stride=1,scope="conv3_4")
        conv3_19=slim.conv2d(conv3_4,256,[3,3],stride=1,scope="conv3_5")
        res3_19=tf.concat(res3_17,conv3_19)

        conv3_20=slim.conv2d(res3_19,256,[3,3],stride=1,scope="conv3_20")
        conv3_21=slim.conv2d(conv3_20,256,[3,3],stride=1,scope="conv3_21")
        res3_21=concat(res3_19,conv3_21)

        conv3_22=slim.conv2d(res3_21,256,[3,3],stride=1,scope="conv3_22")
        conv3_23=slim.conv2d(conv3_22,256,[3,3],stride=1,scope="conv3_23")
        res3_23=tf.concat(conv3_20,conv3_23)

        conv3_24=slim.conv2d(res3_23,256,[3,3],stride=1,scope="conv3_24")
        conv3_25=slim.conv2d(conv3_24,256,[3,3],stride=1,scope="conv3_25")
        res3_25=tf.concat(res3_23,conv3_35)

        conv3_26=slim.conv2d(res3_25,256,[3,3],stride=1,scope="conv3_26")
        conv3_27=slim.conv2d(conv3_26,256,[3,3],stride=1,scope="conv3_27")
        res3_27=tf.concat(conv3_26 ,conv3_27)

        conv3_28=slim.conv2d(res3_27,256,[3,3],stride=1,scope="conv3_28")
        conv3_29=slim.conv2d(conv3_28,256,[3,3],stride=1,scope="conv3_29")
        res3_29=tf.concat(res3_27,conv3_29)

        conv3_30=slim.conv2d(res3_29,256,[3,3],stride=1,scope="conv3_30")
        conv3_31=slim.conv2d(conv3_30,256,[3,3],stride=1,scope="conv3_31")
        res3_31=tf.concat(res3_29  ,conv3_11)

        conv3_32=slim.conv2d(res3_31,256,[3,3],stride=1,scope="conv3_32")
        conv3_33=slim.conv2d(conv3_32,256,[3,3],stride=1,scope="conv3_33")
        res3_33=tf.concat(res3_31,conv3_33)

        conv4_1=slim.conv2d(res3_33,512,[3,3],stride=2,scope="conv4_1",padding="VALID")
        conv4_2=slim.conv2d(conv4_1,512,[3,3],stride=1,scope="conv4_2")
        conv4_3=slim.conv2d(conv4_2,512,[3,3],stride=1,scope="conv4_3")
        res4_3=tf.concat()

        conv4_4=slim.conv2d(res3_9,512,[3,3],stride=2,scope="conv4_4")
        conv4_5=slim.conv2d(conv4_1,512,[3,3],stride=1,scope="conv4_5")
        res4_5=tf.concat(res4_3,conv4_5)

        conv4_6=slim.conv2d(conv4_2,512,[3,3],stride=1,scope="conv4_6")
        conv4_7=slim.conv2d(conv4_2,512,[3,3],stride=1,scope="conv4_7")
        res4_7=.concat(res4_5,conv4_7)

        res4_7=slim.flatten(res4_7)
        features=slim.fully_connected(res4_7,bottle_necl_size,activation_fn=None,scope="fc1")