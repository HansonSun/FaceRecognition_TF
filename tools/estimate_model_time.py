import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import importlib
import tensorflow as tf
import cv2
import sys
sys.path.append("../custom_nets")
sys.path.append("../")
import numpy as np
import time
import argparse
import config

def single_model_runtime(model_def,img_w,img_h,img_c,device):
    if device=="gpu":
        tf_device="/gpu:0"
    else:
        tf_device="/cpu:0"

    with tf.device(tf_device):
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_input_placeholder= tf.placeholder(tf.float32,shape=[1,img_h,img_w,img_c],name="input")
        network = importlib.import_module(model_def)
        prelogits,_ = network.inference(image_input_placeholder,
                                        phase_train=phase_train_placeholder)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if img_c==1:
                img=cv2.imread("time.jpg",0)
            else:
                img=cv2.imread("time.jpg")

            img=img.astype(np.float32)
            img=cv2.resize(img,(img_w,img_h))
            img=np.reshape(img,(1,img_h,img_w,img_c))
            total_time=0.0
            for i in range(21):
                start_t=time.time()
                sess.run(prelogits,feed_dict={image_input_placeholder:img,phase_train_placeholder:False})
                end_t=time.time()
                if i!=0:
                    total_time+=(end_t-start_t)
            print "model_def:%s runtime:%.3f(ms) imgshape:(%d,%d,%d) device:%s"%(model_def,(total_time/20)*1000,img_w,img_h,img_c,device)

def all_model_runtime(model_def_dir):
    for net in os.listdir(net_dir):
        if net.endswith("py"):
            print net.split(".")[0]


def single_pb_runtime(pb_file,img_w,img_h,img_c,device):
    if device=="gpu":
        tf_device="/gpu:0"
    else:
        tf_device="/cpu:0"
    with tf.device(tf_device):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        images_placeholder = detection_graph.get_tensor_by_name("input:0")
        embeddings = detection_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = detection_graph.get_tensor_by_name("phase_train:0")

        with tf.Session(graph=detection_graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if img_c==1:    
                img=cv2.imread("time.jpg",0)
            else:
                img=cv2.imread("time.jpg")

            img=img.astype(np.float32)
            img=cv2.resize(img,(img_w,img_h))
            img=np.reshape(img,(1,img_h,img_w,img_c))
            total_time=0.0
            for i in range(101):
                start_t=time.time()
                sess.run(embeddings, { images_placeholder:img, phase_train_placeholder:False })
                end_t=time.time()
                if i!=0:
                    total_time+=(end_t-start_t)
            print "pb file:%s runtime:%.3f(ms) imgshape:(%d,%d,%d) device:%s"%(pb_file,(total_time/100)*1000,img_w,img_h,img_c,device)
            

def all_pb_runtime(pb_file_dir):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_def', type=str, help='model file', default='')
    parser.add_argument('-md','--model_def_dir', type=str, help='model file dir', default='')
    parser.add_argument('-p','--pb_file', type=str, help='pb file', default='')
    parser.add_argument('-pd','--pb_file_dir', type=str, help='pb file dir', default='')
    parser.add_argument('-ih','--height', type=int, help='image height', default=112)
    parser.add_argument('-iw','--width',type=int, help='image width' , default=112)
    parser.add_argument('-ic','--channel',type=int, help='image channel' , default=3)
    parser.add_argument('-d','--device', type=str, help='device' , default='cpu')
    args=parser.parse_args(sys.argv[1:])

    if args.model_def!="":
        single_model_runtime(args.model_def,args.height,args.width,args.channel,args.device)
    elif args.model_def_dir!="":
        all_model_runtime(args.model_def_dir)
    elif args.pb_file!="":
        single_pb_runtime(args.pb_file,args.height,args.width,args.channel,args.device)
    elif args.pb_file_dir!="":
        all_pb_runtime(args.pb_file_dir)