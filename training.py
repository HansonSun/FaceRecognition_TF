import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("./nets")
import numpy as np
import tensorflow as tf
import input_data
import importlib
import config
import input_data
import tensorflow.contrib.slim as slim
import time
import lossfunc



def trainning(loss,learning_rate,global_step):

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op


def run_training():
    
    if not os.path.exists(config.log_path):   # Create the log directory if it doesn't exist
        os.mkdir(config.log_path)
    if not os.path.exists(config.model_path): # Create the log directory if it doesn't exist
        os.mkdir(config.model_path)


    image_batch,label_batch,class_num,total_img_num = input_data.GetPLFromCsv( config.training_dateset ) 
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')     
    
    #load model and inference
    network = importlib.import_module("lightcnn_b")
    image_batch = tf.identity(image_batch, 'input')
    prelogits = network.inference(image_batch,is_training=phase_train_placeholder)
    logits = slim.fully_connected(prelogits, class_num, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(5e-5),
                    scope='Logits', reuse=False)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    predict_labels=tf.argmax(logits,1)


    #adjust learning rate 
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate,global_step,config.decay_step,config.decay_rate,staircase=True)

    #cal loss and update
    centerloss, _ = lossfunc.center_loss(prelogits, label_batch, config.centerloss_alpha, class_num)
    softmaxloss = lossfunc.softmax_loss(logits, label_batch)  
    total_loss = softmaxloss + config.centerloss_lambda * centerloss
    train_op = trainning(total_loss,learning_rate,global_step)

    saver=tf.train.Saver(max_to_keep=50)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #restore the check point
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
            cur_iter=int(checkpoint.model_checkpoint_path.split("-")[-1])+1

            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        iter=0
        use_time=0
        display_loss=0
        dispaly_acc=0
        try:
            for step in np.arange(config.max_iter):
                if coord.should_stop():
                    break
                start_time=time.time()
                lr,train_loss,_,pre_labels,real_labels = sess.run([learning_rate,total_loss,train_op,predict_labels,label_batch],feed_dict={phase_train_placeholder:True})
                end_time=time.time()
                iter+=1
                use_time+=(end_time-start_time)
                display_loss+=train_loss
                train_acc=np.equal(pre_labels,real_labels).mean()
                dispaly_acc+=train_acc

                if(iter%config.display_iter==0):
                    epoch_scale=iter*config.train_batch_size*1.0/total_img_nu
                    display_loss=display_loss/config.display_iter
                    dispaly_acc=dispaly_acc/config.display_iter

                    print "iterator:%d lr:%f time:%f total_loss:%f acc:%.3f epoch:%.3f"%(iter,lr,use_time,display_loss,dispaly_acc,epoch_scale)
                    use_time=0
                    display_loss=0.0
                    dispaly_acc=0.0

                if (iter % config.snapshot==0): 
                    saver.save(sess, "model/recognize",global_step = iter)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)


run_training()