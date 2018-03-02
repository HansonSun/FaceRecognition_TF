import tensorflow as tf
import facetools.faceutils as fu
from facetools.dataset import *
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

def GetPLFromImg(datasetpath):

	dataset=fileDatasetReader(datasetpath)
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


def GetPLFromCsv(datasetpath):

    dataset=fileDatasetReader(datasetpath)
    csvpath=dataset.save2csv(shuffle=1)

    filename_queue=tf.train.string_input_producer([csvpath],config.epochs)

    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    record_defaults = [[""], [1]] 
    imgpath, label = tf.decode_csv(value, record_defaults=record_defaults) 

    value=tf.read_file(imgpath)
    image=tf.image.decode_image(value,channels=3)

    #convert the color
    if config.train_input_channel==1:
        image = tf.py_func(cvtcolor_image, [image], tf.uint8)

     #resize the image
    image = tf.py_func(resize_image, [image,config.train_input_width,config.train_input_height], tf.uint8)

    #random crop the image
    if config.random_crop==1:
        if(config.train_input_channel==1):
            image = tf.random_crop(image, [config.crop_height,config.crop_width])
        else:
            image = tf.random_crop(image, [config.crop_height, config.crop_width,config.train_input_channel])
        config.train_input_width=config.crop_width
        config.train_input_height=config.crop_height


    #random the image
    if config.random_flip==1:
        image = tf.image.random_flip_left_right(image)

    #random rotate the image
    if config.random_rotate==1:
        image = tf.py_func(random_rotate_image, [image,config.rotate_angle_range[0],config.rotate_angle_range[1]], tf.uint8)

    #random color brightness
    if config.random_color_brightness==1:
        image=tf.image.random_brightness(image,config.max_brightness)

    #random color saturation
    if config.random_color_saturation==1:
        image=tf.image.random_saturation(image,lower=config.saturaton_range[0],upper=config.saturaton_range[1])

    #random color hue
    if config.random_color_hue==1:
        image=tf.image.random_hue(image,config.max_hue)

    #random color contrast 
    if config.random_color_contrast==1:
        image=tf.image.random_contrast(image,lower=config.contrast_range[0],upper=config.contrast_range[1])

    #tell tf tensor's shape
    if config.train_input_channel==1:
        image.set_shape((config.train_input_height,config.train_input_width))
    else :
        image.set_shape((config.train_input_height,config.train_input_width,config.train_input_channel))

    image = tf.cast(image, tf.float32)
    # standardize the image
    if config.normtype==0:
        image=tf.image.per_image_standardization(image)
    elif config.normtype==1:
        image = tf.subtract(image,127.5)
        image=tf.div(image,128.0)
    elif config.normtype==2:
        image=tf.div(image,255.0)

    min_after_dequeue = 1000  
    capacity = min_after_dequeue + config.train_batch_size  

    image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size=config.train_batch_size,num_threads= 10,capacity = capacity,
        min_after_dequeue=min_after_dequeue,allow_smaller_final_batch=True)
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [config.train_batch_size])

    return image_batch,label_batch,dataset.total_identity,dataset.total_img_num

  
def runtest(reader_func):
    image_batch,label_batch,_,_=reader_func( ["/home/hanson/dataset/test_align_128x128"] )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord=tf.train.Coordinator()
        thread=tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            while(1):
                results=sess.run(image_batch)
                for i in results:
                    print i
                    i=i.astype(np.uint8)
                    i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
                    cv2.imshow("fd",i)
                    cv2.waitKey(0)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

if __name__ == '__main__':
    runtest(GetPLFromCsv)