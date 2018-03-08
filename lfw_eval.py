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
import toolfunc


class LFWtest():
    def __init__(self):
        with tf.Graph().as_default():
            self.sess=tf.Session()
            model_dir_exp = os.path.expanduser(config.models_dir)
            meta_file, ckpt_file = toolfunc.get_model_filenames(model_dir_exp)
            
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))

            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def preprocess_image(self,x,normtype=config.normtype):
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
        if config.random_crop==1:
            img_w=config.crop_width
            img_h=config.crop_height
            img_ch=config.train_input_channel
        else:
            img_w=config.train_input_width
            img_h=config.train_input_height
            img_ch=config.train_input_channel

        images = np.zeros((2, img_w, img_h, img_ch))
        for index,i in enumerate([path_l,path_r]):
            img=cv2.imread(i)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(img_w, img_h))
            img=img.astype(np.float32)
            images[index,:,:,:] = self.preprocess_image(img)

        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
        result = self.sess.run(self.embeddings ,feed_dict=feed_dict)
        diff = np.subtract(result[0], result[1])
        similarity = np.sum(np.square(diff))
        return  similarity


def test_lfw():
    print ("|---------------ready to test lfw------------------------|")
    demo=LFWtest()
    lfw=lfwreader("/home/hanson/valid_dataset/LFW/lfw_align_160x160",imgformat="png")
    lfw.load_detector(demo)
    lfw.find_best_threshold()
    print ("|-------------------test lfw done-----------------------------|")

if __name__ == "__main__":
    test_lfw()

