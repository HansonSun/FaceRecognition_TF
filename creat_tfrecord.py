import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
import cv2
import itertools
import os
import tensorflow as tf
import PIL.Image
import numpy as np
import random



def create_example( img_file, label ):
    img = PIL.Image.open(img_file)
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': fu.tf_bytes_feature(img.tobytes() ),
        'img_width':fu.tf_int_feature(img.width),
        'img_height':fu.tf_int_feature(img.height),
        'label': fu.tf_int_feature(label),
    }))
    return example




if __name__ == '__main__':  
    img_dataset=fu.dataset.ImageDatasetReader( "/home/hanson/dataset/test" )
    img_paths,labels=img_dataset.paths_and_labels(shuffle=True)

 
    with tf.python_io.TFRecordWriter("tfrecord_dataset/train.tfrecords") as writer:
        for path,label in zip(img_paths,labels):
            tf_example = create_example(path,label)
            writer.write(tf_example.SerializeToString())

