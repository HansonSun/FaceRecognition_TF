import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import sys
sys.path.append("/home/hanson/facetools/lib")
sys.path.append("../")
import faceutils as fu
import math
import config

class TFRecordDataset(fu.TFRecordDataset):
    def __init__(self,easyconfig):
        self.conf=easyconfig 

    def data_parse_function(self,example_proto):
        features = {'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                    }

        features = tf.parse_single_example(example_proto, features)
        img = tf.image.decode_jpeg(features['image_raw'],channels=3)
        img=self.tfimgprocess(img)
        label = tf.cast(features['label'], tf.int64)
        return img,label



    '''
    def data_parse_function(self,example_proto):
        features = {'img_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'img_width':tf.FixedLenFeature([], tf.int64),
                    'img_height':tf.FixedLenFeature([], tf.int64)}

        features = tf.parse_single_example(example_proto, features)
        img_width=tf.cast(features['img_width'], tf.int64)
        img_height=tf.cast(features['img_height'], tf.int64)
        img = tf.decode_raw(features['img_raw'],tf.uint8)
        img = tf.reshape(img, shape=(img_height,img_width,3))
        img=self.tfimgprocess(img)
        label = tf.cast(features['label'], tf.int64)
        return img, label
    '''


def read_tfrecord_test():
    demo=TFRecordDataset( config.get_config() )
    iterator,next_element=demo.generateDataset(test_mode=1,batch_size=1)
    print (demo.nrof_classes)
    sess = tf.Session()

    # begin iteration
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, label= sess.run(next_element)
                print (label)
                resultimg= images[0]
                resultimg=cv2.cvtColor(resultimg,cv2.COLOR_RGB2BGR)
                cv2.imshow('test', resultimg)
                cv2.waitKey(0)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break


if __name__ == '__main__':
    read_tfrecord_test()