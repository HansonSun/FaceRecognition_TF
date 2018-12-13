import tensorflow as tf
from easydict import EasyDict as edict  

def get_config( ):
    conf = edict()
    ##-----------------train process parameter-----------------------##
    #training dataset path list,if the input dataset is image dataset ,you needn't set the nrof_classes
    conf.training_dateset_path = "/home/hanson/dataset/VGGFACE2/train_fc_zoom_filter_130x130"
    conf.dataset_img_width=128
    conf.dataset_img_height=128
    ##-----------------finetune process parameter-----------------------------------##
    conf.finetune_dataset_path="/home/hanson/dataset/glasses_fr_dataset"
    conf.finetune_model_dir="ToBeConvertModels"
    conf.finetune_nrof_classes=-1

    conf.test_mode=0

    conf.nrof_classes=-1  #the code can auto infernce from dataset path
    conf.batch_size=120
    conf.display_iter=10
    conf.test_save_iter=5000
    conf.max_nrof_epochs=1000
    conf.fr_model_def="inception_resnet_v1"         #train facerecognize model
    conf.classification_model_def="vgg11"   #loss test model
    conf.feature_length=512 #feature output size
    conf.test_mode=0  #used to test input data
    conf.topn_threshold=98
    conf.distance_metric="euclidean"  #euclidean distance | cosine distance
    conf.feature_flip=0     #when set feature filp to 1 ,it will get twice size of feature
    conf.feature_normlize=0


    ##--------------benchmark test----------------------------------##
    conf.benchmark_dict={
    "test_lfw":1,  #topn save must set lfw test flage to 1
    "lfw_path":"/home/hanson/valid_dataset/FaceRecognize/LFW/lfw_facecrop_112x112",
    "lfw_format":"jpg",

    "test_agedb":0,
    "agedb_path":"/home/hanson/valid_dataset/FaceRecognize/AGEDB",
    "agedb_format":"jpg",

    "test_cfp":0,
    "cfp_path":"/home/hanson/valid_dataset/FaceRecognize/CFP/Images_112x112",
    "cfp_format":"png",

    "test_ytf":0,
    "ytf_path":"/home/hanson/valid_dataset/FaceRecognize/YTF/youtube/frame_images_DB",
    "ytf_format":"png",

    "test_sllfw":0,
    "sllfw_path":"home/hanson/valid_dataset/FaceRecognize/sllfw",
    "sllfw_format":"jpg",

    "test_calfw":0,
    "calfw_path":"/home/hanson/valid_dataset/FaceRecognize/CALFW",
    "calfw_format":"jpg",

    "test_cplfw":0,
    "cplfw_path":"/home/hanson/valid_dataset/FaceRecognize/CPLFW",
    "cplfw_format":"png"}
    ##--------------------hyper parameter---------------------------##
    lr_type_dict={0:'exponential_decay',1:'piecewise_constant',2:'manual_modify'}
    conf.lr_type=lr_type_dict[1]
    conf.learning_rate=0.05  #if learning_rate is -1,use learning_rate schedule file
    #expontial decay
    learning_rate_decay_step=1000
    learning_rate_decay_rate=0.98
    #piecewise constant
    conf.boundaries = [2,16,32] #the dataset epoch 
    conf.values     = [0.01, 0.01, 0.001,0.0001]  #the number means learning rate
    assert len(conf.values)-len(conf.boundaries)==1
    #manual_modify
    modify_step=100

    # optimizer func
    optimizer_dict={0:'ADAGRAD',1:'ADADELTA',2:'ADAM',3:'RMSPROP',4:'MOM'}
    conf.optimizer=optimizer_dict[2]
    conf.moving_average_decay=0.9999
    conf.weight_decay=5e-4
    conf.gpu_memory_fraction=1

    ##---------------------Data Augment-----------------------------##
    #open random crop,crop image size must less than dataset image size
    conf.random_crop=1
    conf.crop_img_width=112
    conf.crop_img_height=112
    conf.input_img_width  = conf.crop_img_width  if conf.random_crop else  conf.dataset_img_width
    conf.input_img_height = conf.crop_img_height if conf.random_crop else  conf.dataset_img_height
    #random rotate
    conf.random_rotate=0
    conf.rotate_angle_range=[89,91]
    #random flip
    conf.random_flip=1
    #blur image
    conf.blur_image=0
    conf.blur_ratio_range=[0.5,1]  #0 to 1.0
    #random brigtness
    conf.random_color_brightness=0
    conf.brightness_range=[-0.1,0.5]  #0.0 to 1.0
    #random hue
    conf.random_color_hue=0
    conf.hue_range=[0,0.05]              #0 to 0.5
    #random contrast
    conf.random_color_contrast=0
    conf.contrast_range=[0.8,1.5]
    #random saturation
    conf.random_color_saturation=0
    conf.saturaton_range=[0.6,1.5]
    #image preprocess type
    # 0: image=(image-mean)/std
    # 1: image=(image-127.5)/128.0
    # 2: image=image/255.0
    conf.img_preprocess_type=1



    ##-----------------------loss function paramter------------------------------##
    loss_type_dict={0:'softmax',1:'Centerloss',2:'AdditiveAngularMargin',3:'AdditiveMargin',4:'AngularMargin',5:'LargeMarginCosine'}
    conf.loss_type=0

    #Centerloss param
    conf.Centerloss_lambda=1e-4
    conf.Centerloss_alpha=0.6
    #AdditiveAngularMargin param
    conf.AdditiveAngularMargin_s=64.0
    conf.AdditiveAngularMargin_m=0.5
    #AdditiveMargin param
    conf.AdditiveMargin_m=0.35
    conf.AdditiveMargin_s=30
    #AngularMargin param
    conf.AngularMargin_m=2
    #LargeMarginCosine param
    conf.LargeMarginCosine_m=0.15
    conf.LargeMarginCosine_s=64.0


    return conf


conf=get_config
if __name__=="__main__":
    print (get_config())