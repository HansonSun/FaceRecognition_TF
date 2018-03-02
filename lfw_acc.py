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
import config

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

    def normimage(self,x,normtype=config.normtype):
        if normtype==0:
            mean = np.mean(x)
            std = np.std(x)
            std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
            y = np.multiply(np.subtract(x, mean), 1/std_adj)
            return y 
        elif normtype==1:
            y=(x-127.5)/128
            return y
        elif normtype==2:
            y=x/255.0
            return y

    def compare2face(self,path_l,path_r):

        images = np.zeros((2, 160, 160, 3))
        for index,i in enumerate([path_l,path_r]):
            img=cv2.imread(i)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(160,160))
            img=img.astype(np.float32)
            images[index,:,:,:] = self.normimage(img)

        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
        result = self.sess.run(self.embeddings ,feed_dict=feed_dict)
        diff = np.subtract(result[0], result[1])
        similarity = np.sum(np.square(diff))
        return  similarity


def test_lfw():
    print ("|---------------ready to test lfw------------------------|")
    demo=test()
    lfw=lfwreader("/home/hanson/valid_dataset/lfw_align_128x128",imgformat="jpg")
    lfw.load_detector(demo)
    lfw.find_best_threshold()
    print ("|-------------------test lfw done-----------------------------|")

if __name__ == "__main__":
    test_lfw()

