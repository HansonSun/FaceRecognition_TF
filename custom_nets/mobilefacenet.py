import tensorflow as tf
import tflearn
import tensorflow.contrib.slim as slim

def bottleneck_2(inputs,num_outputs,name):
    with tf.variable_scope(name):
        output=slim.conv2d(inputs,num_outputs,[1,1],stride=1,scope="conv3_3")
        output=slim.separable_convolution2d(output,num_outputs,[3,3],1,stride=2,scope="dwconv_3_3")
        output=slim.conv2d(output,num_outputs,[1,1],stride=1,activation_fn=None,scope="lconv3_3")
        return output 

def bottleneck_1(inputs,num_outputs,name):
    with tf.variable_scope(name):
        inputs_res=slim.conv2d(inputs,num_outputs,[1,1],stride=1,scope="conv3_3")
        inputs_res=slim.separable_convolution2d(inputs_res,num_outputs,[3,3],1,stride=1,scope="dwconv_3_3")
        inputs_res=slim.conv2d(inputs_res,num_outputs,[1,1],stride=1,activation_fn=None,scope="lconv3_3")
        return inputs_res+inputs

def inference(images, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    end_points = {}
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.00001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]
    }

    with tf.variable_scope("inference"):
        with slim.arg_scope([slim.conv2d , slim.separable_convolution2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=tflearn.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            #input 112x112x3
            inputs=slim.conv2d(images,64,[3,3],stride=2,scope="conv3_3")
            #input 56x56x64
            inputs=slim.separable_convolution2d(inputs,64,[3,3],1,stride=1,scope="dwconv_3_3")
            #input 56x56x64
            inputs=bottleneck_2(inputs,64,"bottleneck_1_1")
            inputs=bottleneck_1(inputs,64,"bottleneck_1_2")
            #input 28x28x64
            inputs=bottleneck_2(inputs,128,"bottleneck_2_1")
            inputs=bottleneck_1(inputs,128,"bottleneck_2_2")
            inputs=bottleneck_1(inputs,128,"bottleneck_2_3")
            inputs=bottleneck_1(inputs,128,"bottleneck_2_4")
            #input 14x14x128
            inputs=bottleneck_1(inputs,128,"bottleneck_3_1")
            inputs=bottleneck_1(inputs,128,"bottleneck_3_2")
            #input 14x14x128
            inputs=bottleneck_2(inputs,128,"bottleneck_4_1")
            inputs=bottleneck_1(inputs,128,"bottleneck_4_2")
            inputs=bottleneck_1(inputs,128,"bottleneck_4_3")
            inputs=bottleneck_1(inputs,128,"bottleneck_4_4")
            #input 7x7x128
            inputs=bottleneck_1(inputs,128,"bottleneck_5_1")
            inputs=bottleneck_1(inputs,128,"bottleneck_5_2")

            #input 7x7x128
            inputs=slim.conv2d(inputs,512,[1,1],stride=1,scope="conv1_1")
            print inputs
            #input 7x7x512
            inputs=slim.separable_convolution2d(inputs,512,[7,7],1,stride=1,scope="gdconv7_7")
            #input 1x1x512
            inputs=slim.conv2d(inputs,bottleneck_layer_size,[1,1],stride=1,activation_fn=None,scope="lconv1_1")

    return inputs,end_points