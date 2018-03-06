import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import importlib
import tensorflow as tf
import cv2
import config
import sys
sys.path.append("./nets")
import numpy as np
import time

def single_model_runtime(net,img_w,img_h,img_c,device="cpu"):

    if device=="gpu":
        tf_device="/gpu:0"
    else:
        tf_device="/cpu:0"

    with tf.device(tf_device):
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train') 
        image_input_placeholder= tf.placeholder(tf.float32,shape=[1,img_h,img_w,img_c],name="input")
        network = importlib.import_module(net)
        prelogits,_ = network.inference(image_input_placeholder,phase_train=phase_train_placeholder)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            img=cv2.imread("time.jpg")
            img=img.astype(np.float32)
            img=cv2.resize(img,(img_w,img_h))
            img=np.reshape(img,(1,img_w,img_h,img_c))
            total_time=0.0
            for i in range(10):
                start_t=time.time()
                sess.run(prelogits,feed_dict={image_input_placeholder:img,phase_train_placeholder:False})
                end_t=time.time()
                if i!=0:
                    total_time+=(end_t-start_t)
            print "model:%s runtime:%.3f(ms) imgshape:(%d,%d,%d) device:%s"%(net,(total_time/9)*1000,img_w,img_h,img_c,device)

def test_all_model(net_dir):
    for net in os.listdir(net_dir):
        if net.endswith("py"):
            print net.split(".")[0]

single_model_runtime("squeezenet",112,112,3)
#test_all_model("nets")