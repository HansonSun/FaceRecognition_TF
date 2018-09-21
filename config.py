import tensorflow as tf
##-----------------train process parameter-----------------------##
#training dataset path list,if the input dataset is image dataset ,you needn't set the nrof_classes
training_dateset_path = "/home/hanson/dataset/glasses_fr_dataset"
dataset_img_width=128
dataset_img_height=128
##-----------------finetune process parameter-----------------------------------##
finetune_dataset_path="/home/hanson/dataset/glasses_fr_dataset"
finetune_model_dir="ToBeConvertModels"
finetune_nrof_classes=-1

nrof_classes=-1  #the code can auto infernce from dataset path
batch_size=90
display_iter=10
test_save_iter=5000
max_nrof_epochs=1000
models_dir="saved_models/"
fr_model_def="vgg11"         #train facerecognize model
classification_model_def="vgg11"   #loss test model
embedding_size=512 #feature output size
input_test_flag=0  #used to test input data
topn_threshold=98
distance_metric=0  #0:euclidean distance 1: cosine distance
feature_flip=0     #when set feature filp to 1 ,it will get twice size of feature
feature_normlize=0


##--------------benchmark test----------------------------------##
test_lfw=1  #topn save must set lfw test flage to 1
lfw_dateset_path="/home/hanson/valid_dataset/FaceRecognize/LFW/lfw_facecroponly_zoom0.10"
lfw_format="jpg"

test_agedb=0
agedb_dateset_path="/home/hanson/valid_dataset/FaceRecognize/AGEDB"
agedb_format="jpg"

test_cfp=0
cfp_dateset_path="/home/hanson/valid_dataset/FaceRecognize/CFP/Images_112x112"
cfp_format="png"

test_ytf=0
ytf_dateset_path="/home/hanson/valid_dataset/FaceRecognize/YTF/youtube/frame_images_DB"
ytf_format="png"

test_sllfw=0
sllfw_dateset_path="home/hanson/valid_dataset/FaceRecognize/sllfw"
sllfw_format="jpg"

test_calfw=0
calfw_dateset_path="/home/hanson/valid_dataset/FaceRecognize/CALFW"
calfw_format="jpg"

test_cplfw=0
cplfw_dateset_path="/home/hanson/valid_dataset/FaceRecognize/CPLFW"
cplfw_format="png"
##--------------------hyper parameter---------------------------##
lr_type_list=['exponential_decay','piecewise_constant','manual_modify']
lr_type=lr_type_list[0]
learning_rate=0.05  #if learning_rate is -1,use learning_rate schedule file
#expontial decay
learning_rate_decay_step=1000
learning_rate_decay_rate=0.98
#piecewise constant
boundaries = [10000, 100000,500000] #the num means iters
values = [0.1, 0.01, 0.001,0.0001]  #the number means learning rate
#manual_modify
modify_step=100

# optimizer func
optimizer_list=['ADAGRAD','ADADELTA','ADAM','RMSPROP','MOM']
optimizer=optimizer_list[2]
moving_average_decay=0.9999
weight_decay=5e-4
gpu_memory_fraction=1

##---------------------Data Augment-----------------------------##
#open random crop,crop image size must less than dataset image size
random_crop=1
crop_img_width=112
crop_img_height=112
input_img_width  = crop_img_width  if random_crop else dataset_img_width
input_img_height = crop_img_height if random_crop else dataset_img_height
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
# 0: image=(image-mean)/std
# 1: image=(image-127.5)/128.0
# 2: image=image/255.0
img_preprocess_type=1



##-----------------------loss function paramter------------------------------##
loss_type_list=['softmax','Centerloss','AdditiveAngularMargin','AdditiveMargin','AngularMargin','LargeMarginCosine']
loss_type=0

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
