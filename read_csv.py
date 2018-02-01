import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import facetools.faceutils as fu
from facetools.dataset import DatasetReader
import cv2
import numpy as np
from scipy import misc
import config 
  
def random_rotate_image(image,lowangle,highangle):
    angle = np.random.uniform(lowangle,highangle)
    return misc.imrotate(image, angle, 'bicubic') 

def resize_image(image,resize_w,resize_h):
    image=cv2.resize(image,(resize_w,resize_h))
    return image

def cvtcolor_image(image):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

def GetPathsandLabels(datasetcsvpath):

    filename_queue=tf.train.string_input_producer([datasetcsvpath],2)

    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    record_defaults = [[""], [1.0]] 
    col1, col2 = tf.decode_csv(value, record_defaults=record_defaults) 

    value=tf.read_file(col1)
    image=tf.image.decode_image(value,channels=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
             while(1):
                #x_, y_,line_key = sess.run([col1, col2,key])
                #print (x_,y_,line_key)
                result=sess.run(image)
                print result.shape
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

       

if __name__ == '__main__':
    GetPathsandLabels( "dataset.csv")  
