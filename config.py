import tensorflow as tf


##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = ["/home/hanson/dataset/CASIA/CASIA-WebFace_align_128x128"]
train_batch_size=2
train_input_width=300
train_input_height=300
train_input_channel=3
#max epochs
epochs=1000
display_iter=10
max_iter=1000000
snapshot=10
saver_path="./model"
##--------------------------------------------------------------##


##---------------test process parameter-------------------------##
#testing data path list
testing_dataset  = ["/home/hanson/dataset/test_align_128x128"]
test_batch_size=128
test_input_width=128
test_input_height=128
test_input_channel=1
test_interval= 1000
##--------------------------------------------------------------##


##--------------------hyper parameter---------------------------##
learning_rate=0.001
decay_step=1000
decay_rate=0.96
# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[0]
##--------------------------------------------------------------##


##---------------------Data Augment-----------------------------##
#open random crop
random_crop=1
crop_width=50
crop_height=50
#random rotate
random_rotate=0
rotate_angle_range=[-90,90]
#random flip 
random_flip=0
#resize image
resize_image=0
##----------------------------------------------------------------##


##-----------------------center loss------------------------------##
centerloss_lambda=1e-2
centerloss_alpha=0.5
##----------------------------------------------------------------##
