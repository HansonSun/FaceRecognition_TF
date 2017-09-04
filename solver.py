import tensorflow as tf

train_batch_size=128
train_input_width=152
train_input_height=152
train_input_channel=1


test_batch_size=128
test_input_width=128
test_input_height=128
test_input_channel=1



FINAL_PIC_SIZE=5

CLASS_NUM=10575

max_iter=3000000



base_lr=0.0001
decay_step=10000
decay_rate=0.96



snapshot=1000


test_interval= 1000

display_iter=10



phase_train=10
phase_test=11


