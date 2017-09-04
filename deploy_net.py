import tensorflow as tf
import numpy as np
import solver




def Deploy_Net(images):
    x_image = tf.reshape(images, [-1,solver.test_input_width,solver.test_input_height,solver.test_input_channel])  

    #Conv1
    W_conv1=tf.get_variable( name="w_conv1",shape=[9,9,1,96],initializer=tf.contrib.layers.xavier_initializer() )
    b_conv1=tf.get_variable( name="b_conv1",initializer=tf.constant(0.1,shape=[96]) )
             
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')+ b_conv1
    print "pass conv1"
    #Pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #Slice1
    slice1_1,slice1_2 = tf.split(pool1,[48,48],3)

    #eltwise max
    eltwise1=tf.maximum(slice1_1,slice1_2)

    #Conv2
    W_conv2=tf.get_variable( name="w_conv2",shape=[5, 5, 48, 192],initializer=tf.contrib.layers.xavier_initializer() )
    b_conv2=tf.get_variable( name="b_conv2",initializer=tf.constant(0.1,shape=[192]) )

    conv2 = tf.nn.conv2d(eltwise1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')+ b_conv2
    #Pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    #Slice2
    slice2_1,slice2_2 = tf.split(pool2,[96,96],3)

    #eltwise max
    eltwise2=tf.maximum(slice2_1,slice2_2)

    #Conv3
    W_conv3=tf.get_variable( name="w_conv3",shape=[5, 5, 96, 256],initializer=tf.contrib.layers.xavier_initializer() )
    b_conv3=tf.get_variable( name="b_conv3",initializer=tf.constant(0.1,shape=[256]) )


    conv3 = tf.nn.conv2d(eltwise2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')+ b_conv3
    #Pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    #Slice3
    slice3_1,slice3_2 = tf.split(pool3,[128,128],3)

    #eltwise max 3
    eltwise3=tf.maximum(slice3_1,slice3_2)

    #Conv4 
    W_conv4=tf.get_variable( name="w_conv4",shape=[4, 4, 128, 384],initializer=tf.contrib.layers.xavier_initializer() )
    b_conv4=tf.get_variable( name="b_conv4",initializer=tf.constant(0.1,shape=[384]) )

    conv4 = tf.nn.conv2d(eltwise3, W_conv4, strides=[1, 1, 1, 1], padding='VALID')+ b_conv4
    #Pool4
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    #Slice4
    slice4_1,slice4_2 = tf.split(pool4,[192,192],3)

    #eltwise max 4
    eltwise4=tf.maximum(slice4_1,slice4_2)

    #FC1
    W_fc1=tf.get_variable( name="w_fc1",shape=[solver.FINAL_PIC_SIZE * solver.FINAL_PIC_SIZE*192,512],initializer=tf.contrib.layers.xavier_initializer() )
    b_fc1=tf.get_variable( name="b_fc1",initializer=tf.constant(0.1,shape=[512]) )

    fc1_flat = tf.reshape(eltwise4, [-1, solver.FINAL_PIC_SIZE * solver.FINAL_PIC_SIZE*192])
    fc1=tf.add(tf.matmul(fc1_flat, W_fc1) , b_fc1)

    #Slice5
    slice5_1,slice5_2 = tf.split(fc1,[256,256],1)

    #eltwise max 5
    eltwise5=tf.maximum(slice5_1,slice5_2)


    return eltwise5

