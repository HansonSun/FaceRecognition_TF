import tensorflow as tf
##-----------------train process parameter-----------------------##
#training dataset path list
training_dateset = "/home/hanson/dataset/CASIA/CASIA-WebFace_align_96x112"
num_output=10575

batch_size=128
dataset_img_width=128
dataset_img_height=128
display_iter=10
save_iter=1000

max_nrof_epochs=1000
epoch_size=1000
models_dir="models/"
logs_dir="logs/"
train_net="squeezenet"
embedding_size=128

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
learning_rate=0.01  #if learning_rate is -1,use learning_rate schedule file
learning_rate_decay_step=1000
learning_rate_decay_rate=0.96
learning_rate_schedule_file="lr_schedule/learning_rate_schedule_classifier_casia.txt"
# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[3]
keep_probability=0.8
moving_average_decay=0.9999
weight_decay=5e-5
gpu_memory_fraction=1
nrof_preprocess_threads=4
##--------------------------------------------------------------##

##---------------------Data Augment-----------------------------##
#open random crop
random_crop=1
crop_img_width=112
crop_img_height=112

input_img_width=crop_img_width if random_crop else dataset_img_width
input_img_height=crop_img_height if random_crop else dataset_img_height

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

#image preprocess type
process_type=1
##----------------------------------------------------------------##

##-----------------------center loss------------------------------##
center_loss_lambda=0
center_loss_alpha=0.9
##----------------------------------------------------------------##
