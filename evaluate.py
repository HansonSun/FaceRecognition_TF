import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
import tensorflow as tf
import numpy as np
import cv2
from facetools.lfwreader import *
import importlib
from tensorflow.python.framework import graph_util
class test():
    def __init__(self):
        with tf.Graph().as_default():
            self.sess=tf.Session()
            ckpname=str(tf.train.latest_checkpoint("model") )
            print ckpname
            saver = tf.train.import_meta_graph(ckpname+".meta")
            saver.restore(self.sess, ckpname )

            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


    def compare2face(self,path_l,path_r):

        images = np.zeros((2, 128, 128, 3))
        for index,i in enumerate([path_l,path_r]):
            img=cv2.imread(i)
            img=cv2.resize(img,(128,128))
            images[index,:,:,:] = img

        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
        result = self.sess.run(self.embeddings ,feed_dict=feed_dict)
        diff = np.subtract(result[0], result[1])
        similarity = np.sum(np.square(diff))
        return  similarity

demo=test()
print demo.compare2face("75-1.jpg","75-2.jpg")
#lfw=lfwreader("/home/hanson/valid_dataset/lfw_align_128x128")
#lfw.get_result(demo)
