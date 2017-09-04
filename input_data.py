import tensorflow as tf
import cv2
import numpy as np
import random
import solver
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool


class input_data() :

	def __init__(self,file_path):

		self.label_list=[]
		self.img_list=[]
		self.total_file_cnt=0
		self.cur_file_pos=0

		print "loading data"
		path_file=open(file_path)
		for line in  path_file.readlines():
			line=line[:-1]
			img_path=line.split(' ')[0]
			img_label=int(line.split(' ')[1])


			self.img_list.append(img_path)
			self.label_list.append(img_label)
			
		self.all_data_array = np.array([self.img_list, self.label_list])
		self.all_data_array = self.all_data_array.transpose()
		np.random.shuffle(self.all_data_array)

		self.total_file_cnt=self.all_data_array.shape[0]
		self.cur_file_pos=0

		print "total:%d"%self.total_file_cnt


	
	def next_batch(self,batch_size=solver.train_batch_size,img_width=solver.train_input_width,img_height=solver.train_input_height,img_channel=solver.train_input_channel):
		
		start_pos=self.cur_file_pos
		end_pos=0

		if (start_pos+batch_size)<self.total_file_cnt:
			end_pos=start_pos+batch_size
			self.cur_file_pos+=batch_size
			
			img_result=np.zeros( (batch_size,solver.train_input_width,solver.train_input_height,solver.train_input_channel),dtype=np.float32 )
			output_data=self.all_data_array[start_pos:end_pos]

			for i in range(batch_size):
				tmp_img=cv2.imread(output_data[i,0],0 )
				tmp_img=cv2.resize(tmp_img,(img_width,img_height))
				tmp_img=tmp_img.astype(np.float32)
				tmp_img=tmp_img/255
				img_result[i,...,0]=tmp_img


			label_result=output_data[...,1]
			label_result=label_result.astype(np.int32)

			return img_result, label_result
		
		else:
			np.random.shuffle(self.all_data_array)
			self.cur_file_pos=0
			start_pos=self.cur_file_pos
			end_pos=start_pos+batch_size
			self.cur_file_pos+=batch_size

			img_result=np.zeros( (batch_size,solver.train_input_width,solver.train_input_height,solver.train_input_channel),dtype=np.float32 )
			output_data=self.all_data_array[start_pos:end_pos]

			for i in range(batch_size):
				tmp_img=cv2.imread(output_data[i,0],0 )
				tmp_img=cv2.resize(tmp_img,(img_width,img_height))
				tmp_img=tmp_img.astype(np.float32)
				tmp_img=tmp_img/255
				img_result[i,...,0]=tmp_img


			label_result=output_data[...,1]
			label_result=label_result.astype(np.int32)

			return img_result, label_result #shape : [batch_size,img_width,img_height,img_channel]


	def get_all_data_by_batchs(self,batch_size=solver.test_batch_size,img_width=solver.test_input_width,img_height=solver.test_input_height,img_channel=solver.test_input_channel):

		div_cnt=self.total_file_cnt/batch_size
		print div_cnt


		img_result=np.zeros( (div_cnt,batch_size,img_width,img_height,img_channel),dtype=np.float32 )

		for x in range(div_cnt):
			for  y in range(batch_size):
				tmp_img=cv2.imread(self.all_data_array[x*batch_size+y,0],0 )
				tmp_img=cv2.resize(tmp_img,(solver.test_input_width,solver.test_input_height))
				tmp_img=tmp_img.astype(np.float32)
				tmp_img=tmp_img/255
				img_result[x,y,...,0]=tmp_img


		label_result=np.zeros( (div_cnt,batch_size),dtype=np.float32 )
		for x in range(div_cnt):
			for  y in range(batch_size):
				label_result[x,y]=self.all_data_array[x*batch_size+y,1]
		
		label_result=label_result.astype(np.int32)

		return img_result, label_result  #return shape : [batchs,batch_size,img_width,img_height,img_channel]
		


class input_data_thread() :

	def __init__(self,file_path):

		self.label_list=[]
		self.img_list=[]
		self.total_file_cnt=0
		self.cur_file_pos=0

		print "loading data"
		path_file=open(file_path)
		for line in  path_file.readlines():
			line=line[:-1]
			img_path=line.split(' ')[0]
			img_label=int(line.split(' ')[1])


			self.img_list.append(img_path)
			self.label_list.append(img_label)
			
		self.all_data_array = np.array([self.img_list, self.label_list])
		self.all_data_array = self.all_data_array.transpose()
		np.random.shuffle(self.all_data_array)

		self.total_file_cnt=self.all_data_array.shape[0]
		self.cur_file_pos=0

		print "total:%d"%self.total_file_cnt


	def read_img_to_list(self,path_and_label):
		tmp_img=cv2.imread(path_and_label[1],0 )
		tmp_img=tmp_img.astype(np.float32)
		tmp_img=tmp_img/255
		self.img_result[ path_and_label[0],...,0]=tmp_img



	def next_batch(self,batch_size):
		self.thread_deal_list=[]
		start_pos=self.cur_file_pos
		end_pos=0

		if (start_pos+batch_size)<self.total_file_cnt:
			end_pos=start_pos+batch_size
			self.cur_file_pos+=batch_size
			
			self.img_result=np.zeros( (batch_size,solver.train_input_width,solver.train_input_height,solver.train_input_channel),dtype=np.float32 )
			output_data=self.all_data_array[start_pos:end_pos]

			label_result=output_data[...,1]
			label_result=label_result.astype(np.int32)


			output_data=list(output_data)

			for index,i in enumerate(output_data):
				self.thread_deal_list.append( (index,i[0],int(i[1]) ) )

			if(batch_size<=1000):
				max_pool=batch_size
			else:
				max_pool=1000
			pool = ThreadPool(max_pool)
			return_data = pool.map(self.read_img_to_list, self.thread_deal_list)
			pool.close()
			pool.join()


			return self.img_result, label_result


	def all_data_by_batch(self,batch_size):
		self.thread_deal_list=[]
		div_cnt=self.total_file_cnt/batch_size
		print div_cnt
		#img_result=np.zeros( (div_cnt,batch_size,solver.train_input_width,solver.train_input_height,solver.train_input_channel),dtype=np.float32 )

		for index,i in enumerate(self.all_data_array):
					self.thread_deal_list.append( (index,i[0],int(i[1]) ) )


		if(batch_size<=1000):
			max_pool=batch_size
		else:
			max_pool=1000
		pool = ThreadPool(max_pool)
		return_data = pool.map(self.read_img_to_list, self.thread_deal_list)
		pool.close()
		pool.join()


	

if __name__ =="__main__":

	start=time.time()
	test=input_data("test_list.txt")
	a,b=test.test_all_data(64)
	end=time.time()
	print "use time:%d"%(end-start)
	print a.shape 
	print b.shape
	print a.dtype
	print b.dtype
	print b[0,0]
	print b.shape[0]
	cv2.imshow("fs",a[0,0,...,0])
	cv2.waitKey(0)