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

def GetPathsandLabels(datasetpath):

	dataset=DatasetReader(datasetpath)
	img_paths,labels=dataset.paths_and_labels()

	img_paths = tf.cast(img_paths, tf.string)
	labels = tf.cast(labels, tf.int32)


	input_queue=tf.train.slice_input_producer([img_paths,labels],config.epochs)
	value=tf.read_file(input_queue[0])
	image=tf.image.decode_image(value,channels=3)
	labels=input_queue[1]

	if config.train_input_channel==1:
		image = tf.py_func(cvtcolor_image, [image], tf.uint8)

	if config.random_crop==1:
		if(config.train_input_channel==1):
			image = tf.random_crop(image, [config.crop_width, config.crop_height])
		else:
			image = tf.random_crop(image, [config.crop_width, config.crop_height,config.train_input_channel])
		config.train_input_width=config.crop_width
		config.train_input_height=config.crop_height

	if  config.resize_image==1:
		image = tf.py_func(resize_image, [image,config.train_input_width,config.train_input_height], tf.uint8)

	if config.random_flip==1:
		image = tf.image.random_flip_left_right(image)

	if config.random_rotate==1:
		image = tf.py_func(random_rotate_image, [image,config.rotate_angle_range[0],config.rotate_angle_range[1]], tf.uint8)

	if config.train_input_channel==1:
		image.set_shape((config.train_input_width,config.train_input_height))
	else :
		image.set_shape((config.train_input_width,config.train_input_height,config.train_input_channel))

	min_after_dequeue = 1000  
	capacity = min_after_dequeue + config.train_batch_size  

	image_batch,label_batch=tf.train.shuffle_batch([image,labels],batch_size=config.train_batch_size,num_threads= 10,capacity = capacity,min_after_dequeue=min_after_dequeue)
	image_batch = tf.cast(image_batch, tf.float32)
	label_batch = tf.reshape(label_batch, [config.train_batch_size])

	return image_batch,label_batch,dataset.total_identity
if __name__ == '__main__':
	GetPathsandLabels(["/home/hanson/dataset/test_align_144x144"])