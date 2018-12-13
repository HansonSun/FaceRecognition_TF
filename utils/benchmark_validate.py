from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
sys.path.append("../")
sys.path.append("/home/hanson/facetools/lib/FaceRecognition/method/tensorflow")
sys.path.append("/home/hanson/facetools/lib")
sys.path.append("/home/hanson/facetools/benchmark/FaceRecognition")
import tensorflow as tf
import numpy as np
import cv2
from fr_benchmark_test import  fr_benchmark_test
import config
import faceutils as fu
import scipy
from facerecognize_base import facerecognize_base

class model_inference(facerecognize_base):
    def __init__(self,
                model_dir,
                input_img_width=112,
                input_img_height=112,
                feature_normlize=0,
                img_preprocess_type=1,
                feature_flip=0,
                distance_metric="euclidean"):
        with tf.Graph().as_default():
            self.sess=tf.Session()
            model_dir_exp = os.path.expanduser(model_dir)
            ckpt=tf.train.get_checkpoint_state(model_dir)
            ckpt_file=os.path.basename(ckpt.model_checkpoint_path)
            meta_file=ckpt_file+".meta"

            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))
            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            self.input_img_width=input_img_width
            self.input_img_height=input_img_height
            self.feature_normlize=feature_normlize
            self.img_preprocess_type=img_preprocess_type
            self.feature_flip=feature_flip
            self.distance_metric=distance_metric

    def getFeature(self,img_data):
        images = self.load_data([img_data])
        feed_dict = { self.images_placeholder:images ,self.phase_train_placeholder:False}
        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
        emb_array=emb_array.flatten()
        emb_array=emb_array.reshape((1,emb_array.shape[0]))
        if self.feature_normlize==1:
            return fu.normlize_feature(emb_array)
        else:
            return emb_array



def test_benchmark(conf,model_dir):
    model=model_inference(model_dir,
                input_img_width=conf.input_img_width,
                input_img_height=conf.input_img_height,
                feature_normlize=conf.feature_normlize,
                img_preprocess_type=conf.img_preprocess_type,
                feature_flip=conf.feature_flip,
                distance_metric=conf.distance_metric)

    benchmark=fr_benchmark_test(conf.benchmark_dict)
    return benchmark.top_accurate(model)

if __name__ == "__main__":
    conf=config.get_config()
    test_benchmark(conf,"/home/hanson/work/FaceRecognition_TF/saved_models/20181213-131022/models")
