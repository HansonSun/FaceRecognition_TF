import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
sys.path.append("/home/hanson/work/facetools_install/facetools/fr_method/tensorflow")
sys.path.append("/home/hanson/work/facetools_install/facetools")
import tensorflow as tf
import numpy as np
import cv2
from fr_benchmark_test import  fr_benchmark_test
import config
import faceutils as fu
import scipy
from facerecognize_base import facerecognize_base

class benchmark_validate(facerecognize_base):
    def __init__(self,model_dir):
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

def test_benchmark(model_dir):
    demo=benchmark_validate(model_dir)
    benchmark=fr_benchmark_test(test_lfw=config.test_lfw,
                                lfw_path=config.lfw_dateset_path,
                                test_cfp=config.test_cfp,
                                cfp_path=config.cfp_dateset_path)
    return benchmark.top_accurate(demo)


if __name__ == "__main__":
    test_benchmark("/home/hanson/work/Facerecognize_TF/models/20180327-100239")
