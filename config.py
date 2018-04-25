import tensorflow as tf
##-----------------train process parameter-----------------------##
#training dataset path list,if the input dataset is image dataset ,you needn't set the nrof_classes
training_dateset = "/home/hanson/dataset/VGGFACE2/train_align"
dataset_img_width=182
dataset_img_height=182

nrof_classes=-1
batch_size=100
display_iter=10
save_iter=10
max_nrof_epochs=1000
epoch_size=1000
models_dir="saved_models/"
logs_dir="logs/"
train_net="inception_resnet_v1"
embedding_size=128
input_test_flag=0
topn_threshold=98

##--------------benchmark test----------------------------------##
test_lfw=1  #topn save must set lfw test flage to 1
lfw_root_path="/home/hanson/valid_dataset/LFW/lfw_align"
test_agedb=0
agedb_root_path="/home/hanson/valid_dataset/AGEDB"
test_cfp=0
cfp_root_path="/home/hanson/valid_dataset/CFP/Images_112x112"
test_youtubeface=0
youtube_root_path="home/hanson/valid_dataset/YOUTUBE"

##--------------------hyper parameter---------------------------##
lr_type_list=['exponential_decay','piecewise_constant','manual_modify']
lr_type=lr_type_list[1]
learning_rate=-1  #if learning_rate is -1,use learning_rate schedule file
learning_rate_decay_epochs=100
learning_rate_decay_step=1000
learning_rate_decay_rate=0.96
learning_rate_schedule_file="lr_schedule/learning_rate_schedule_classifier_casia.txt"
boundaries = [10000, 100000,500000]
values = [0.1, 0.01, 0.001,0.0001]

# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[3]
keep_probability=0.8
moving_average_decay=0.9999
weight_decay=5e-5
gpu_memory_fraction=1
nrof_preprocess_threads=4

##---------------------Data Augment-----------------------------##
#open random crop,crop image size must less than dataset image size
random_crop=1
crop_img_width=160
crop_img_height=160
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
process_type=0

##-----------------------center loss------------------------------##
loss_type_list=['softmax','Centerloss','AdditiveAngularMargin','AdditiveMargin','AngularMargin','LargeMarginCosine']
loss_type=1

#Centerloss param
Centerloss_lambda=1e-2
Centerloss_alpha=0.6
#AdditiveAngularMargin param
AdditiveAngularMargin_s=64.0
AdditiveAngularMargin_m=0.5
#AdditiveMargin param
AdditiveMargin_m=0.35
AdditiveMargin_s=30
#AngularMargin param
AngularMargin_m=2
#LargeMarginCosine param
LargeMarginCosine_m=0.4
LargeMarginCosine_s=30.0
