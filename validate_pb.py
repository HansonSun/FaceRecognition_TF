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
import argparse

class pb_validation():
    def __init__(self,img_w,img_h,pb_file,preprocess_type ):
        self.preprocess_type=preprocess_type
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess=tf.Session(graph=self.detection_graph)
        self.images_placeholder = self.detection_graph.get_tensor_by_name("input:0")
        self.embeddings = self.detection_graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.detection_graph.get_tensor_by_name("phase_train:0")

        self.image_w=img_w
        self.image_h=img_h

    def preprocess_image(self,x,preprocess_type):
        if preprocess_type==0:
            mean = np.mean(x)
            std = np.std(x)
            std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
            y = np.multiply(np.subtract(x, mean), 1/std_adj)
            return y
        elif preprocess_type==1:
            y=(x-127.5)/128
            return y
        elif preprocess_type==2:
            y=x/255.0
            return y

    def compare2face(self,path_l,path_r):

        images = np.zeros((2, self.image_w, self.image_h, 3))
        for index,i in enumerate([path_l,path_r]):
            img=cv2.imread(i)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(self.image_w, self.image_h))
            img=img.astype(np.float32)
            images[index,:,:,:] = self.preprocess_image(img,self.preprocess_type)

        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
        feature_array = self.sess.run(self.embeddings ,feed_dict=feed_dict)

        normlized_feature_array=fu.normlize_feature(feature_array)
        similarity_distance=scipy.spatial.distance.euclidean(normlized_feature_array[0], normlized_feature_array[1])
        return similarity_distance


def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("-ih","--img_height",type=int,help='input image height',default=112)
    parser.add_argument("-iw","--img_weight",type=int,help='input image weight',default=112)
    parser.add_argument("-pb","--pb_file",type=str,help='pd file',default="first.pb")
    parser.add_argument("--preprocess_type",type=int,help='preprocess type',default=0)
    parser.add_argument("--lfwpath",type=str,help='lfw path',default="/home/hanson/valid_dataset/LFW/lfw_align_112x112")
    parser.add_argument("--cfppath",type=str,help='cfp path',default="/home/hanson/valid_dataset/CFP/Images_112x112")
    args=parser.parse_args(argv)
    demo=pb_validation(img_w=args.img_weight,
                       img_h=args.img_height,
                       pb_file=args.pb_file,
                       preprocess_type=args.preprocess_type)
    benchmark=fr_validate(test_lfw=1,lfwpath=args.lfwpath,test_cfp=1,cfppath=args.cfppath)
    return benchmark.top_accurate(demo)


if __name__ == "__main__":
    main(sys.argv[1:])
