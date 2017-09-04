import tensorflow as tf
import cv2	
from input_data import input_data
from input_data import input_data_thread
import numpy as np
import solver
import train_test_net

def Training():

    #loading train and test dataset
    print "loading train data"
    train_data= input_data("train_list.txt")

    print "loading test data"
    test_data = input_data("test_list.txt")

    x  = tf.placeholder(tf.float32,[None,None,None,None], name='x-input')

    y  = tf.placeholder(tf.int32,[None], name='y-input')
    phase=tf.placeholder(tf.int32,(1),name="phase")

    #train and get loss
    logits = train_test_net.Train_Test_Net(x,phase)
    predict_labels=tf.argmax(logits,1)
    logits=tf.cast(logits, tf.float32 )
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    #decay the learning rate
    global_step=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(solver.base_lr,global_step=global_step,decay_steps=solver.decay_step,decay_rate=solver.decay_rate)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    saver=tf.train.Saver(max_to_keep=50)
    variable_init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables() )

    cur_iter=1
    train_loss_array=0
    train_acc_array=0

    with tf.Session() as sess :
        sess.run(variable_init)

        #restore the check point
        checkpoint = tf.train.get_checkpoint_state("saved_model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
            cur_iter=int(checkpoint.model_checkpoint_path.split("-")[-1])+1

        else:
            print ('Could not find old network weights')


        for iter in range(cur_iter,solver.max_iter+1):
            train_img,train_label=train_data.next_batch( )
            train_feed = { x: train_img , y: train_label , phase:solver.phase_train}
            pred_value,_,loss_value=sess.run([pred_label,train_op,loss],feed_dict=train_feed )

            train_loss+=loss_value.mean()
            train_acc+=np.equal(pred_value,train_label).mean()
            
            if(iter%solver.display_iter==0):
                print "iterator:%d train loss:%f accuracy:%f"%(iter,train_loss/solver.display_iter,train_acc/solver.display_iter)

            
            if (iter % solver.snapshot==0): 
                saver.save(sess, "saved_model/recognize",global_step = iter)


            if(iter %test_interval==0 ):

                print "ready to test"
                print "run Test mode"
                test_img,test_label=test_data.get_all_data_by_batchs( )
                print "test label shape:",test_label.shape
                print "test img shape:",test_img.shape
                test_loss_array=np.zeros((len(test_label)),np.float32)
                test_acc_array=np.zeros((len(test_label)),np.float32)

                for index,test_data_batch in enumerate(test_img):
                    test_feed={x:test_data_batch,y:test_label[index],phase:solver.phase_train}
                    test_pred_value,_,test_loss_value=sess.run([pred_label,train_op,loss],feed_dict=test_feed ) 
                    test_loss_array[index]=test_loss_value.mean()
                    test_acc_array[index]=np.equal(test_pred_value,test_label[index]).mean()
                print " ---->test loss:%f accuracy:%f"%(test_loss_array.mean(),test_acc_array.mean())


def Testing():
     #loading test dataset
    print "loading test data"
    test_data = input_data("test_list.txt")


    x  = tf.placeholder(tf.float32,[None,solver.test_input_width,solver.test_input_height,solver.test_input_channel], name='x-input')  #shape : [?,128,128,1]
    y  = tf.placeholder(tf.int32,[None], name='y-input') #shape : [?]
    phase=tf.placeholder(tf.int32,(1),name="phase")

    #train and get loss
    logits = train_test_net.Train_Test_Net(x, phase)
    predict_labels=tf.argmax(logits,1)
    logits=tf.cast(logits, tf.float32 )
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)


    saver=tf.train.Saver(max_to_keep=50)
    variable_init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables() )

    cur_iter=1
    with tf.Session() as sess :
        sess.run(variable_init)

        #restore the check point
        checkpoint = tf.train.get_checkpoint_state("saved_model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
            cur_iter=int(checkpoint.model_checkpoint_path.split("-")[-1])+1

        else:
            print ('Could not find old network weights')


        print "run Test mode"
        test_img,test_label=test_data.test_all_data(solver.batch_size)
        print "test label shape:",test_label.shape
        print "test img shape:",test_img.shape
        test_loss_array=np.zeros((len(test_label)),np.float32)
        test_acc_array=np.zeros((len(test_label)),np.float32)

        for index,test_data_div in enumerate(test_img):
            test_pred_value,_,test_loss_value=sess.run([pred_label,train_op,loss],feed_dict={x:test_data_div,y:test_label[index],phase:solver.test } ) 
            test_loss_array[index]=test_loss_value.mean()
            test_acc_array[index]=np.equal(test_pred_value,test_label[index]).mean()
        print " train loss:%f accuracy:%f"%(test_loss_array.mean(),test_acc_array.mean())


if __name__=="__main__":

    Training()
