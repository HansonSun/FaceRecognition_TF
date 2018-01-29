import tensorflow as tf


##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = ["/home/hanson/dataset/CASIA/CASIA-WebFace_align_128x128"]
train_batch_size=64
train_input_width=128
train_input_height=128
train_input_channel=3
#max epochs
epochs=100000
display_iter=10
max_iter=1000000
snapshot=1000
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
base_lr=0.01
decay_step=10000
decay_rate=0.96
# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[0]
##--------------------------------------------------------------##


##---------------------Data Augment-----------------------------##
#open random crop
random_crop=0
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

