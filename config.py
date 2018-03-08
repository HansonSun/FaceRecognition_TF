import tensorflow as tf


##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = ["/home/hanson/dataset/VGGFACE2/train_align"]
train_batch_size=64
train_input_width=170
train_input_height=170
train_input_channel=3    #rgb:3  gray:1
#max epochs
epochs=1000
display_iter=10
max_iter=1000000
snapshot=1000
models_dir=""
logs_dir=""
train_net="squeezenet"
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

##--------------benchmark test----------------------------------##
lfw_root_path="/home/hanson/valid_dataset/LFW/lfw_align_160x160"
agedb_root_path="/home/hanson/valid_dataset/AGEDB"
cfp_root_path="/home/hanson/valid_dataset/CFP"
youtube_root_path="home/hanson/valid_dataset/YOUTUBE"
##--------------------------------------------------------------##

##--------------------hyper parameter---------------------------##
learning_rate=0.001
decay_step=1000
decay_rate=0.96
# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[2]

moving_average_decay=0.9999
weight_decay=5e-5 
##--------------------------------------------------------------##


##---------------------Data Augment-----------------------------##
#open random crop
random_crop=1
crop_width=160
crop_height=160

#random rotate
random_rotate=0
rotate_angle_range=[-90,90]

#random flip 
random_flip=1

#random brigtness
random_color_brightness=0
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
normtype=-1

##----------------------------------------------------------------##

##-----------------------center loss------------------------------##
centerloss_lambda=0.003
centerloss_alpha=0.9
##----------------------------------------------------------------##
