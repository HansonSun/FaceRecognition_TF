import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
import numpy as np
import tensorflow as tf
import input_data
import model
import importlib
import config
import input_data


def run_training():

	image_batch,label_batch = input_data.GetPathsandLabels( )      

	network = importlib.import_module("lightcnn_a")

	train_logits = network.inference(image_batch)

    #train_loss = model.losses(train_logits, train_label_batch)      

    #train_op = model.trainning(train_loss, learning_rate)
 
	sess = tf.Session()

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
	try:
		for step in np.arange(config.max_iter):
			if coord.should_stop():
				break
			result = sess.run(train_logits)
			print result.shape
               

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		coord.request_stop()
        
	coord.join(threads)
	sess.close()


run_training()