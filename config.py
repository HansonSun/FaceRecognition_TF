import tensorflow as tf

##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = ["/home/hanson/dataset/VGGFACE2/train_align"]
train_batch_size=128
train_input_width=128
train_input_height=128
train_input_channel=3    #rgb:3  gray:1
#max epochs
epochs=100
display_iter=10
max_iter=1000000
snapshot=1000
models_dir=""
logs_dir=""
train_net="squeezenet"
emb_size=256
##--------------------------------------------------------------##

##--------------benchmark test----------------------------------##
test_lfw=1
lfw_root_path="/home/hanson/valid_dataset/LFW/lfw_align_112x112"
test_agedb=0
agedb_root_path="/home/hanson/valid_dataset/AGEDB"
test_cfp=0
cfp_root_path="/home/hanson/valid_dataset/CFP/Images_112x112"
test_youtubeface=0
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
crop_width=112
crop_height=112

inference_image_width=crop_width if random_crop else train_input_width
inference_image_height=crop_height if random_crop else train_input_height

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
centerloss_lambda=0
centerloss_alpha=0.9
##----------------------------------------------------------------##
