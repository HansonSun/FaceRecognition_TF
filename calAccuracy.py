import os.path
import time
import numpy
import tensorflow as tf
import cv2
import numpy as np
import solver
import deploy_net
import scipy 


class LFW_Reader():
	def __init__(self):
		print "ready to load lfw data set..."
		f_path=open("Path_lfw2.txt","r")
		self.path_dict={}
		self.total_pair=6000
		for index,img_path in enumerate(f_path.readlines()):
			self.path_dict[index+1]=os.path.join("hand_lfw-ALIGN-GRAY-128x128",img_path[:-2] )
		print "find %d picture"%len(self.path_dict)
		f_path.close()

	def get_pos_list(self):
		pos_pair=3000
		img_array_l=np.zeros( (pos_pair,128,128,1) )
		img_array_r=np.zeros( (pos_pair,128,128,1) )

		pos_f=open("postive_pairs.txt","r")
		for index,pos_text in enumerate( pos_f.readlines() ):
			l_num,r_num= pos_text.split("   ")
			l_num=int(l_num)
			r_num=int(r_num[:-1])

			img_l=cv2.imread(self.path_dict[l_num],0)

			img_l=img_l.astype(np.float32)
			img_l=img_l/255
			img_array_l[index,...,0]=img_l

			img_r=cv2.imread(self.path_dict[r_num],0)
			img_r=img_r.astype(np.float32)
			img_r=img_r/255
			img_array_r[index,...,0]=img_r

		pos_f.close()
		print img_array_l.shape, img_array_r.shape
		return pos_pair,img_array_l,img_array_r
		

	def get_neg_list(self):
		neg_pair=3000

		img_array_l=np.zeros( (neg_pair,128,128,1) )
		img_array_r=np.zeros( (neg_pair,128,128,1) )


		neg_f=open("negative_pairs.txt","r")
		for index,neg_text in enumerate( neg_f.readlines() ):
			l_num,r_num= neg_text.split("   ")
			l_num=int(l_num)
			r_num=int(r_num[:-1])

			img_l=cv2.imread(self.path_dict[l_num],0)
			img_l=img_l.astype(np.float32)
			img_l=img_l/255
			img_array_l[index,...,0]=img_l

			img_r=cv2.imread(self.path_dict[r_num],0)
			img_r=img_r.astype(np.float32)
			img_r=img_r/255
			img_array_r[index,...,0]=img_r

		neg_f.close()
		print img_array_l.shape, img_array_r.shape
		return neg_pair,img_array_l,img_array_r



def is_same_person(feature_l,feature_r):
	th=0.2
	similar = 1 - scipy.spatial.distance.cosine(feature_l,feature_r)   # similarity of same person
	if similar>th:
		return 1
	else :
		return 0



def run_prediction( ):
	right_cnt=0
	lfw_reader=LFW_Reader( )

	img_list=tf.placeholder( tf.float32,[None,128,128,1],name="input" ) 
	
	logits = deploy_net.Deploy_Net(img_list)

	init_op=tf.global_variables_initializer()

	saver = tf.train.Saver(tf.global_variables() )

	with tf.Session() as sess:
		sess.run(init_op)
		checkpoint = tf.train.get_checkpoint_state("saved_model")
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
		else:
			print ('Could not find old network weights')

		
		pos_pair,pos_img_array_l,pos_img_array_r=lfw_reader.get_pos_list()
		for i in range(pos_pair):
			single_array_l=np.reshape(pos_img_array_l[i],(1,128,128,1))
			
			single_array_r=np.reshape(pos_img_array_r[i],(1,128,128,1))
			#cv2.imshow("l",pos_img_array_l[i])
			#cv2.imshow("r",pos_img_array_r[i])
			#cv2.waitKey(3000)
	

			l_pos_img_attr=sess.run(logits,feed_dict={img_list:single_array_l} )
			r_pos_img_attr=sess.run(logits,feed_dict={img_list:single_array_r} )
			pos_simi=is_same_person(l_pos_img_attr,r_pos_img_attr)
			
			if pos_simi==1:
				#print "right"
				right_cnt+=1
			else :
				print "wrong%f"%( 1 - scipy.spatial.distance.cosine(l_pos_img_attr,r_pos_img_attr) )
		

		neg_pair,neg_img_array_l,neg_img_arrat_r=lfw_reader.get_neg_list()
		for i in range(neg_pair):
			single_array_l=np.reshape(neg_img_array_l[i],(1,128,128,1))
			single_array_r=np.reshape(neg_img_arrat_r[i],(1,128,128,1))
			l_neg_img_attr=sess.run(logits,feed_dict={img_list:single_array_l} )
			r_neg_img_attr=sess.run(logits,feed_dict={img_list:single_array_r} )
			neg_simi=is_same_person(l_neg_img_attr,r_neg_img_attr)
			
			if neg_simi==0:
				#print "right"
				right_cnt+=1
			else :
				print "wrong%f"%( 1 - scipy.spatial.distance.cosine(l_neg_img_attr,r_neg_img_attr) )

		print "final accuracy is %f"%(right_cnt/6000.0)




if __name__ =="__main__":
	run_prediction()



