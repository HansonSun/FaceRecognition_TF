from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

def cal_loss(features,labels,nrof_classes,centrloss_alpha=0.6,centerloss_lambda=0.01,w_init=tf.constant_initializer(0)):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)

        Args:

        features: embeddings feature 
        labels:  batch labels
        nrof_classes: number of classes
        centrloss_alpha: center loss hyparamter 
        centerloss_lambda:center loss hyparameter 
        w_init_:initialize the center
    """
    #get feature length
    nrof_features = features.get_shape()[1]
    #create centers
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,initializer=w_init, trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(features - centers_batch)
    #ready to update centers
    diff =(centers_batch - features)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = centrloss_alpha * diff
    centers = tf.scatter_sub(centers, labels, diff)
    return centerloss_lambda*loss, centers

    
def cal_loss_test( use_tfe=1):
    if use_tfe:
        print ("eager model enable")
        tfe.enable_eager_execution()
        #synthesis features which batch size is 5 ,lenghtn is 16
        features=tf.get_variable(name="features",dtype=tf.float32,shape=[6,2],initializer=tf.random_normal_initializer(seed=223))
        print (features)
        labels=np.array((0,0,0,1,1,2))
        print ("labels:",labels)
        nrof_classes=3
        w_init_method=tf.random_normal_initializer(seed=666)
        loss,center=cal_loss(features, labels, nrof_classes,centrloss_alpha=0.1,centerloss_lambda=0.2,w_init=w_init_method)
        print (loss,center )
    else:
        print ("eager model disable")
        #synthesis features which batch size is 5 ,lenghtn is 16
        features=tf.get_variable(name="features",dtype=tf.float32,shape=[6,2],initializer=tf.random_normal_initializer(seed=223))
        labels=np.array((0,0,0,1,1,2))
        print ("labels:",labels)
        nrof_classes=3
        w_init_method=tf.random_normal_initializer(seed=666)
        loss,center=cal_loss(features, labels, nrof_classes,centrloss_alpha=0.1,centerloss_lambda=0.2,w_init=w_init_method)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        embeddings_v,loss_v= sess.run([features,loss])
        print (loss_v,embeddings_v )
        sess.close()
if __name__ == "__main__":
    cal_loss_test()
