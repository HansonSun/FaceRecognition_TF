import tensorflow as tf


##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = ["/home/hanson/dataset/CASIA/CASIA-WebFace_align_zoom0.10"]
train_batch_size=128
train_input_width=144
train_input_height=144
train_input_channel=3    #rgb:3  gray:1
#max epochs
epochs=1000
display_iter=10
max_iter=1000000
snapshot=1000
model_path="./model"
log_path="./log"

##--------------------------------------------------------------##


##---------------test process parameter-------------------------##
#testing data path list
testing_dataset  = ["/home/hanson/dataset/test_align_128x128"]
test_batch_size=128
test_input_width=128
test_input_height=128
test_input_channel=3
test_interval= 1000
##--------------------------------------------------------------##


##--------------------hyper parameter---------------------------##
learning_rate=0.001
decay_step=1000
decay_rate=0.96
# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[0]

moving_average_decay=0.9999
weight_decay=5e-5 
##--------------------------------------------------------------##


##---------------------Data Augment-----------------------------##
#open random crop
random_crop=1
crop_width=128
crop_height=128

#random rotate
random_rotate=0
rotate_angle_range=[-90,90]

#random flip 
random_flip=1

#random brigtness
random_color_brightness=1
max_brightness=0.2

#random hue
random_color_hue=0
max_hue=0

#random contrast 
random_color_contrast=0
contrast_range=[0.5,1.5]

#random saturation
random_color_saturation=0
saturaton_range=[0.5,1.5]

#normtype
normtype=0

##----------------------------------------------------------------##

##-----------------------center loss------------------------------##
centerloss_lambda=0.003
centerloss_alpha=0.9
##----------------------------------------------------------------##
