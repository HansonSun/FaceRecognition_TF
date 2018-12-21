import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("TrainingNet")
sys.path.append("TrainingLoss")
import numpy as np
import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim
import time
from  trainFR_base import trainFR_base


class trainFR(trainFR_base):
    def __init__(self):
        #super(trainFR,self).__init__()
        trainFR_base.__init__(self)

    def make_model(self):
        #2.load dataset and define placeholder
        self.phase_train  = tf.placeholder(tf.bool, name='phase_train')
        self.images_input = tf.placeholder(name='input', shape=[None, self.conf.input_img_height,self.conf.input_img_width, 3], dtype=tf.float32)
        self.labels_input = tf.placeholder(name='labels', shape=[None, ], dtype=tf.int64)

        #3.load model and inference
        network = importlib.import_module(self.conf.fr_model_def)
        print ("trianing network:%s"%self.conf.fr_model_def)

        self.prelogits = network.inference(
            self.images_input,
            dropout_keep_prob=0.8,
            phase_train=self.phase_train,
            weight_decay=self.conf.weight_decay,
            feature_length=self.conf.feature_length)
    

    def make_loss(self):
        logits = slim.fully_connected(self.prelogits,
                                      self.nrof_classes,
                                      activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(5e-4),
                                      scope='Logits',
                                      reuse=False)
        self.softmaxloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_input),name="loss")

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(self.prelogits)+eps, ord=1, axis=1))
        tf.add_to_collection('losses', prelogits_norm * 5e-4)
        tf.add_to_collection('losses', self.softmaxloss)

        custom_loss=tf.get_collection("losses")
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss=tf.add_n(custom_loss+regularization_losses,name='total_loss')

        embeddings = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')
        self.predict_labels=tf.argmax(logits,1)

        #add loss summary
        tf.summary.scalar("softmax_loss",self.softmaxloss)
        tf.summary.scalar("total_loss",self.total_loss)

        print ("lr stratege : %s"%self.conf.lr_type)


    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            for epoch in range(self.conf.max_nrof_epochs):
                sess.run(self.traindata_iterator.initializer)
                while True:
                    self.use_time=0
                    try:
                        images_train, labels_train = sess.run(self.traindata_next_element)

                        start_time=time.time()
                        input_dict={self.phase_train:True,self.images_input:images_train,self.labels_input:labels_train}
                        result_op=[self.train_op,self.global_step,self.learning_rate,self.total_loss,self.predict_labels,self.labels_input]
                        _,self.step,lr,train_loss,predict_labels,real_labels ,summary_str= sess.run(result_op+[self.summary_op],feed_dict=input_dict)
                        self.summary_writer.add_summary(summary_str, global_step=self.step)
                        end_time=time.time()
                        self.use_time+=(end_time-start_time)
                        train_acc=np.equal(predict_labels,real_labels).mean()
                        self.train_display()
                        
                    except tf.errors.OutOfRangeError:
                        print("End of epoch ")
                        break


if __name__=="__main__":
    fr=trainFR()
    fr.run()