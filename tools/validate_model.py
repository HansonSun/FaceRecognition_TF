import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import importlib
import tensorflow as tf
import cv2
import sys
sys.path.append("../nets")
sys.path.append("../")
import numpy as np
import time
import argparse
import config

def test(net,img_w,img_h,img_c):
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_input_placeholder= tf.placeholder(tf.float32,shape=[1,img_h,img_w,img_c],name="input")
    network = importlib.import_module(net)
    prelogits,_ = network.inference(image_input_placeholder,
                                    phase_train=phase_train_placeholder,
                                    w_init=tf.random_normal_initializer(seed=666))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if img_c==1:
            img=cv2.imread("time.jpg",0)
        else:
            img=cv2.imread("time.jpg")

        img=img.astype(np.float32)
        img=cv2.resize(img,(img_w,img_h))
        img=np.reshape(img,(1,img_h,img_w,img_c))
        print sess.run(prelogits,feed_dict={image_input_placeholder:img,phase_train_placeholder:False})

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--net', type=str, help='net', default='vgg19')
    parser.add_argument('-ih','--height', type=int, help='image height', default=112)
    parser.add_argument('-iw','--width',type=int, help='image width' , default=112)
    parser.add_argument('-ic','--channel',type=int, help='image channel' , default=3)
    args=parser.parse_args(sys.argv[1:])

    test(args.net,args.height,args.width,args.channel)
