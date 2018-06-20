from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("../")
sys.path.append("/home/hanson/work/facetools_install/facetools/")
import tensorflow as tf
import numpy as np
import cv2
from fr_benchmark_test import fr_benchmark_test
import config
import faceutils as fu
import scipy
import argparse
from fr_method.tensorflow.facerecognize_base import facerecognize_base as face_fr


def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("-ih","--input_img_height",type=int,help='input image height',default=112)
    parser.add_argument("-iw","--input_img_weight",type=int,help='input image weight',default=112)
    parser.add_argument("-p","--pb_file",type=str,help='pd file',default="112x112.pb")
    parser.add_argument("--preprocess_type",type=int,help='preprocess type',default=0)
    parser.add_argument("--lfw_path",type=str,help='lfw path',default="/home/hanson/valid_dataset/FaceRecognize/LFW/lfw_facenet")
    parser.add_argument("--cfp_path",type=str,help='cfp path',default="/home/hanson/valid_dataset/FaceRecognize/CFP/Images_112x112")
    parser.add_argument("--cplfw_path",type=str,help='cplfw path',default="/home/hanson/valid_dataset/FaceRecognize/CFP/Images_112x112")
    parser.add_argument("--sllfw_path",type=str,help='sllfw path',default="/home/hanson/valid_dataset/FaceRecognize/CFP/Images_112x112")
    args=parser.parse_args(argv)
    demo=face_fr(input_img_width=args.input_img_height,
                 input_img_height=args.input_img_weight,
                 pb_file=args.pb_file)
    benchmark=fr_benchmark_test(test_lfw=1,lfw_path=args.lfw_path,test_cfp=1,cfp_path=args.cfp_path)
    return benchmark.top_accurate(demo)


if __name__ == "__main__":
    main(sys.argv[1:])
