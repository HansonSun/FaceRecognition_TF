import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import importlib
import tensorflow.contrib.slim as slim
import sys
sys.path.append("./nets")
sys.path.append("lossfunc")
import numpy as np
import tools_func

# Get MNIST Data
mnist = input_data.read_data_sets('classification_dataset/MNIST/', one_hot=False)
# Variables
class_num=10
batch_size = 200
total_steps = 10000
bottleneck_layer_size=128
draw_feature_flag=0
# Build Model
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
img_placeholder=tf.placeholder(tf.float32, [batch_size, 28,28,1])
label_placeholder = tf.placeholder(tf.int64, [batch_size,])
global_step=tf.Variable(0,trainable=False,name='global_step')

#-----------------------------------modify here--------------------------------------------------##
#load network
network = importlib.import_module("lightcnn_b")
lossfunc=importlib.import_module("AngularMargin")
#lossfunc=importlib.import_module("AdditiveMargin")
#lossfunc=importlib.import_module("AdditiveAngularMargin")
#lossfunc=importlib.import_module("LargeMarginCosine")
features,end_points = network.inference(img_placeholder,
                                        phase_train=phase_train_placeholder,
                                        bottleneck_layer_size=bottleneck_layer_size,
                                        weight_decay=5e-5)
#logits = slim.fully_connected(features, class_num,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),scope='Logits_end',reuse=False)
logits,custom_loss=lossfunc.cal_loss(features,label_placeholder,class_num)

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#centerloss, centers, centers_update_op=lossfunc.center_loss(features,label_placeholder,0.9,10)
#total_loss=softmaxloss+centerloss*0.01
total_loss=tf.add_n([custom_loss]+regularization_losses)

# Loss
optimizer = tf.train.AdamOptimizer(0.001)
grads = optimizer.compute_gradients(total_loss)
update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
##------------------------------------------------------------------------------------##

# Prediction
correct_prediction = tf.equal(tf.argmax(logits,1),label_placeholder )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train 10000 steps
    for step in range(total_steps + 1):
        images, labels = mnist.train.next_batch(batch_size)
        images=np.reshape(images,(batch_size,28,28,1))
        _,trainloss,acc=sess.run([train_op,total_loss,accuracy],feed_dict={img_placeholder:images,label_placeholder:labels ,phase_train_placeholder: True})

        if step%100==0:
            print ("step %d ,total loss %.4f , train accuracy %f"%(step,trainloss,acc) )
        if step%1000==0:
            images=np.reshape(mnist.test.images,(10000,28,28,1))
            labels=mnist.test.labels
            total_acc=0
            test_features=[]
            test_iter=int(10000/batch_size)
            for i in range(test_iter):
                fd={img_placeholder:images[i*batch_size:(i+1)*batch_size],label_placeholder:labels[i*batch_size:(i+1)*batch_size],phase_train_placeholder: False}
                test_features_tmp,acc=sess.run([features,accuracy],feed_dict=fd)
                total_acc+=acc
                test_features.append((test_features_tmp) )
            print ("test accuracy %.4f"%(total_acc*1.0/test_iter) )
#draw feature
if draw_feature_flag==1:
    test_features=(np.array(test_features) ).reshape( (10000,bottleneck_layer_size) )
    tools_func.draw_features(test_features,mnist.test.labels)
