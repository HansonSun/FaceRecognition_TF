from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import sys
sys.path.append("custom_nets")
import random
import tensorflow as tf
import numpy as np
import importlib
import tools_func
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import cv2
import config
from benchmark_validate import *

def resize_image(image,resize_w,resize_h):
    image=cv2.resize(image,(resize_w,resize_h))
    return image


def main( ):

    network = importlib.import_module(config.train_net)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.logs_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    models_dir = os.path.join(os.path.expanduser(config.models_dir), subdir)
    if not os.path.isdir(models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(models_dir)
    topn_models_dir = os.path.join(models_dir,"topn")
    if not os.path.isdir(topn_models_dir):  # Create the topn model directory if it doesn't exist
        os.makedirs(topn_models_dir)

    seed=666
    np.random.seed(seed=seed)
    random.seed(seed)
    train_set = tools_func.get_dataset(config.training_dateset)
    nrof_classes = len(train_set)

    print('Model directory: %s' % models_dir)
    print('Log directory: %s' % log_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, label_list = tools_func.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(config.batch_size*config.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                image = tf.py_func(resize_image, [image,config.dataset_img_width,config.dataset_img_height], tf.uint8)

                if config.random_rotate:
                    image = tf.py_func(tools_func.random_rotate_image, [image], tf.uint8)
                if config.random_crop:
                    image = tf.random_crop(image, [config.input_img_height,config.input_img_width,3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, config.input_img_height,config.input_img_width)
                if config.random_flip:
                    image = tf.image.random_flip_left_right(image)

                #pylint: disable=no-member
                image.set_shape((config.input_img_height,config.input_img_width,3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(config.input_img_height,config.input_img_width,3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * config.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, config.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=config.embedding_size,
            weight_decay=config.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if config.Centerloss_lambda>0.0:
            prelogits_center_loss, _ = tools_func.center_loss(prelogits, label_batch, config.Centerloss_alpha, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * config.Centerloss_lambda)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            config.learning_rate_decay_epochs*config.epoch_size, config.learning_rate_decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = tools_func.train(total_loss, global_step, config.optimizer,
            learning_rate, config.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < config.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // config.epoch_size
                # Train for one epoch
                train( sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, config.learning_rate_schedule_file)

                filename = os.path.join(models_dir, "%d.cpkt"%step)
                saver.save(sess, filename)
                if config.test_lfw==1 :
                    acc_dict=test_benchmark(os.path.join(models_dir))
                    if acc_dict["lfw_acc"]>config.topn_threshold:
                        topn_file=open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")
                        topn_file.write("%s %s\n"%(os.path.join(topn_models_dir, "%d.ckpt"%step),str(acc_dict)) )
                        shutil.copyfile(os.path.join(models_dir, "%d.ckpt.meta"%step),os.path.join(topn_models_dir, "%d.ckpt.meta"%step))
                        shutil.copyfile(os.path.join(models_dir, "%d.ckpt.index"%step),os.path.join(topn_models_dir, "%d.ckpt.index"%step))
                        shutil.copyfile(os.path.join(models_dir, "%d.ckpt.data-00000-of-00001"%step),os.path.join(topn_models_dir, "%d.ckpt.data-00000-of-00001"%step))
                        topn_file.close()
    return models_dir


def train( sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if config.learning_rate>0.0:
        lr = config.learning_rate
    else:
        lr = tools_func.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < config.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:config.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tlr %f\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, config.epoch_size, lr,duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


if __name__ == '__main__':
    main( )
