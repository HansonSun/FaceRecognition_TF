import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import os
import config
import input_data_old
from facetools.dataset import *
import time


def random_rotate_image(image,lowangle,highangle):
	angle = np.random.uniform(lowangle,highangle)
	return misc.imrotate(image, angle, 'bicubic')

def resize_image(image,resize_w,resize_h):
	image=cv2.resize(image,(resize_w,resize_h))
	return image

def cvtcolor_image(image):
	image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	return image

def text_parse_function(imgpath, label):
	value=tf.read_file(imgpath)
	img=tf.image.decode_image(value,channels=3)
	#resize the image
	img = tf.py_func(resize_image, [img,config.dataset_img_width,config.dataset_img_height], tf.uint8)

	#random crop the image
	if config.random_crop==1:
		if config.crop_img_height>config.dataset_img_height or config.crop_img_width>config.dataset_img_width:
			print ("crop size must <= input size")
			exit()
		img = tf.random_crop(img, [config.crop_img_height,config.crop_img_width,3])

	#random the image
	if config.random_flip==1:
		img = tf.image.random_flip_left_right(img)

	#random rotate the image
	if config.random_rotate==1:
		img = tf.py_func(random_rotate_image, [img,config.rotate_angle_range[0],config.rotate_angle_range[1]], tf.uint8)

	#random color brightness
	if config.random_color_brightness==1:
		img=tf.image.random_brightness(img,config.max_brightness)

	#random color saturation
	if config.random_color_saturation==1:
		img=tf.image.random_saturation(img,lower=config.saturaton_range[0],upper=config.saturaton_range[1])

	#random color hue
	if config.random_color_hue==1:
		img=tf.image.random_hue(img,config.max_hue)

	#random color contrast
	if config.random_color_contrast==1:
		img=tf.image.random_contrast(img,lower=config.contrast_range[0],upper=config.contrast_range[1])

	if config.input_test_flag==0:
		img = tf.cast(img, tf.float32)
		# standardize the image
		if config.process_type==0:
			img=tf.img.per_image_standardization(img)
		elif config.process_type==1:
			img = tf.subtract(img,127.5)
			img=tf.div(img,128.0)
		elif config.process_type==2:
			img=tf.div(img,255.0)

	return img, label

def tfrecord_parse_function(example_proto):
	features = {'image_raw': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.int64)}
	features = tf.parse_single_example(example_proto, features)
	# You can do more image distortion here for training data
	img = tf.image.decode_jpeg(features['image_raw'])
	img = tf.reshape(img, shape=(112,112,3))
	r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
	img = tf.concat([b, g, r], axis=-1)

	#resize the image
	img = tf.py_func(resize_image, [img,config.dataset_img_width,config.dataset_img_height], tf.uint8)
	#random crop the image
	if config.random_crop==1:
		if config.crop_img_height>config.dataset_img_height or config.crop_img_width>config.dataset_img_width:
			print ("crop size must <= input size")
			exit()
		img = tf.random_crop(img, [config.crop_img_height,config.crop_img_width,3])

	#random the image
	if config.random_flip==1:
		img = tf.image.random_flip_left_right(img)

	#random rotate the image
	if config.random_rotate==1:
		img = tf.py_func(random_rotate_image, [img,config.rotate_angle_range[0],config.rotate_angle_range[1]], tf.uint8)

	#random color brightness
	if config.random_color_brightness==1:
		img=tf.image.random_brightness(img,config.max_brightness)

	#random color saturation
	if config.random_color_saturation==1:
		img=tf.image.random_saturation(img,lower=config.saturaton_range[0],upper=config.saturaton_range[1])

	#random color hue
	if config.random_color_hue==1:
		img=tf.image.random_hue(img,config.max_hue)

	#random color contrast
	if config.random_color_contrast==1:
		img=tf.image.random_contrast(img,lower=config.contrast_range[0],upper=config.contrast_range[1])

	if config.input_test_flag==0:
		img = tf.cast(img, tf.float32)
		# standardize the image
		if config.normtype==0:
			img=tf.img.per_image_standardization(img)
		elif config.normtype==1:
			img = tf.subtract(img,127.5)
			img=tf.div(img,128.0)
		elif config.normtype==2:
			img=tf.div(img,255.0)

	label = tf.cast(features['label'], tf.int64)
	return img, label

def img_input_data(dataset_path,batch_size):
	img_dataset=fileDatasetReader([dataset_path])
	img_paths,labels=img_dataset.paths_and_labels()
	dataset=tf.data.Dataset.from_tensor_slices((img_paths,labels))
	dataset = dataset.map(text_parse_function)
	dataset = dataset.shuffle(buffer_size=20000)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	return iterator,next_element

def tfrecord_input_data(record_path,batch_size):
	record_dataset = tf.data.TFRecordDataset(record_path)
	record_dataset = record_dataset.map(tfrecord_parse_function)
	record_dataset = record_dataset.shuffle(buffer_size=20000)
	record_dataset = record_dataset.batch(batch_size)
	iterator = record_dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	return iterator,next_element


def read_text_test():
	config.input_test_flag=1
	iterator,next_element=img_input_data("/home/hanson/dataset/test_align_144x144",2)
	sess = tf.Session()

	for i in range(10):
		sess.run(iterator.initializer)
		while True:
			try:
				images, labels = sess.run(next_element)
				cv2.imshow('test', images[1, ...])
				cv2.waitKey(0)
				print labels

			except tf.errors.OutOfRangeError:
				print("End of dataset")
				break


def read_tfrecord_test():
	config.input_test_flag=1
	iterator,next_element=tfrecord_input_data('tfrecord_dataset/ms1m_train.tfrecords',2)
	sess = tf.Session()

	# begin iteration
	for i in range(1000):
		sess.run(iterator.initializer)
		while True:
			try:
				images, labels = sess.run(next_element)
				cv2.imshow('test', images[1, ...])
				cv2.waitKey(0)
				print labels
			except tf.errors.OutOfRangeError:
				print("End of dataset")


if __name__ == '__main__':
	read_text_test()
