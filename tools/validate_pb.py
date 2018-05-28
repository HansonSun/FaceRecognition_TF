import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("../")
sys.path.append("/home/hanson/work/facetools_install/facetools/")
import tensorflow as tf
import numpy as np
import cv2
from facerecognize_validate import facerecognize_validate
import config
import faceutils as fu
import scipy
import argparse
import fr_method.tensorflow.facerecognize_base as face_fr


def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("-ih","--img_height",type=int,help='input image height',default=112)
    parser.add_argument("-iw","--img_weight",type=int,help='input image weight',default=112)
    parser.add_argument("-p","--pb_file",type=str,help='pd file',default="faceidentify_160x160.pb")
    parser.add_argument("--preprocess_type",type=int,help='preprocess type',default=0)
    parser.add_argument("--lfw_path",type=str,help='lfw path',default="/home/hanson/valid_dataset/LFW/lfw_facenet")
    parser.add_argument("--cfp_path",type=str,help='cfp path',default="/home/hanson/valid_dataset/CFP/Images_112x112")
    parser.add_argument("--cplfw_path",type=str,help='cplfw path',default="/home/hanson/valid_dataset/CFP/Images_112x112")
    parser.add_argument("--sllfw_path",type=str,help='sllfw path',default="/home/hanson/valid_dataset/CFP/Images_112x112")
    args=parser.parse_args(argv)
    demo=face_fr(image_width=args.img_weight,
               image_height=args.img_height,
               pb_file=args.pb_file)
    benchmark=facerecognize_validate(test_lfw=1,lfw_path=args.lfw_path,test_cfp=1,cfp_path=args.cfp_path)
    return benchmark.top_accurate(demo)


if __name__ == "__main__":
    main(sys.argv[1:])
