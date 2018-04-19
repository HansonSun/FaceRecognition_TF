import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
sys.path.append("./facetools")
import tensorflow as tf
import numpy as np
import cv2
from fr_validate import  fr_validate
import config
import faceutils as fu
import scipy

class benchmark_validate():
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

    def preprocess_image(self,x,normtype=config.process_type):
        if config.process_type==0:
            mean = np.mean(x)
            std = np.std(x)
            std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
            y = np.multiply(np.subtract(x, mean), 1/std_adj)
            return y
        elif config.process_type==1:
            y=(x-127.5)/128
            return y
        elif config.process_type==2:
            y=x/255.0
            return y

    def compare2face(self,path_l,path_r):
        img_w=config.input_img_width
        img_h=config.input_img_height

        images = np.zeros((2, img_w, img_h, 3))
        for index,i in enumerate([path_l,path_r]):
            img=cv2.imread(i)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(img_w, img_h))
            img=img.astype(np.float32)
            images[index,:,:,:] = self.preprocess_image(img)

        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
        feature_array = self.sess.run(self.embeddings ,feed_dict=feed_dict)

        normlized_feature_array=fu.normlize_feature(feature_array)
        similarity_distance=scipy.spatial.distance.euclidean(normlized_feature_array[0], normlized_feature_array[1])
        return similarity_distance


def test_benchmark(model_dir):
    demo=benchmark_validate(model_dir)
    benchmark=fr_validate(test_lfw=config.test_lfw,lfwpath=config.lfw_root_path,test_cfp=config.test_cfp,cfppath=config.cfp_root_path)
    return benchmark.top_accurate(demo)


if __name__ == "__main__":
    test_benchmark("/home/hanson/work/Facerecognize_TF/models/20180327-100239")
