import tensorflow as tf
import facetools.faceutils as fu
from facetools.dataset import DatasetReader
import cv2
import numpy as np
from scipy import misc

  
def random_rotate_image(image):
    image=cv2.resize(image,(500,500))
    return image
    #return misc.imrotate(image, 90, 'bicubic')


def get_batch2(datesetpath):

    dataset=DatasetReader(datesetpath)
    img_paths,labels=dataset.paths_and_labels()

    img_paths = tf.cast(img_paths, tf.string)
    labels = tf.cast(labels, tf.int32)


    input_queue=tf.train.slice_input_producer([img_paths,labels],num_epochs=1)

    value=tf.read_file(input_queue[0])
    image=tf.image.decode_image(value,channels=3)


    image = tf.py_func(random_rotate_image, [image], tf.uint8)
    image.set_shape((500,500, 3))

    image_batch=tf.train.batch([image],batch_size= 5,num_threads= 64,capacity = 200)
    image_batch = tf.cast(image_batch, tf.float32)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threads=tf.train.start_queue_runners(sess=sess)

    k=0
    while 1:
        image_data = sess.run(image_batch)
        print image_data.shape
        for index,i in enumerate(image_data):
            #print i.shape
            #cv2.imshow("d",i)
            #cv2.waitKey(0)
            cv2.imwrite("test/%d_%d.png"%(k,index),i )
        k+=1

    sess.close()

if __name__ == '__main__':
	get_batch2(["/home/hanson/dataset/test_align_144x144"])