import tensorflow as tf

train_batch_size=2
train_input_width=144
train_input_height=144
train_input_channel=3


test_batch_size=128
test_input_width=128
test_input_height=128
test_input_channel=1


CLASS_NUM=10575

max_iter=100



base_lr=0.0001
decay_step=10000
decay_rate=0.96



snapshot=1000


test_interval= 1000

display_iter=10



phase_train=10
phase_test=11

#open random crop
random_crop=0
crop_width=50
crop_height=50

#random rotate
random_rotate=1
rotate_angle_range=[-90,90]

#random flip 
random_flip=0


#max epochs
epochs=1

#resize image
resize_image=0

#training dataset dir
train_dir = ["/home/hanson/dataset/test_align_144x144"]
logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'