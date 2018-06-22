import tensorflow as tf
import tensorflow.contrib.slim as slim

def channel_shuffle(X, groups):
    height, width, in_channels = X.shape.as_list()[1:]
    in_channels_per_group = int(in_channels/groups)

    # reshape
    shape = tf.stack([-1, height, width, groups, in_channels_per_group])
    X = tf.reshape(X, shape)

    # transpose
    X = tf.transpose(X, [0, 1, 2, 4, 3])

    # reshape
    shape = tf.stack([-1, height, width, in_channels])
    X = tf.reshape(X, shape)

    return X

def group_conv_1x1(inputs,num_output,groups):
    #get inputs channels
    inputs_channels = inputs.shape.as_list()[3]

    inputs_channels_per_group = int(inputs_channels/groups)
    output_channels_per_group = int(num_output/groups)
    # split channels
    inputs_channels_split = tf.split(inputs, [inputs_channels_per_group]*groups, axis=3)
    results = []
    # do convolution for each split
    for i in range(groups):
        inputs_split = inputs_channels_split[i]
        results += [slim.conv2d(inputs_split,output_channels_per_group,kernel_size=1)]
    return tf.concat(results, 3)

def shufflenet_unit(inputs, groups, stride):
    #get inputs channels
    inputs_channels = inputs.shape.as_list()[3]

    outputs = group_conv_1x1(inputs,inputs_channels,groups)
    outputs = slim.batch_norm(outputs)
    outputs = tf.nn.relu(outputs)
    outputs = channel_shuffle(outputs, groups)
    outputs = slim.separable_conv2d(outputs, None,kernel_size=3,depth_multiplier=1,stride=stride)
    outputs = slim.batch_norm(outputs )
    outputs = group_conv_1x1(outputs,inputs_channels,groups)
    outputs = slim.batch_norm(outputs)

    if stride < 2:
        result = tf.add(inputs, outputs)
    else:
        output_bypass = slim.avg_pool2d(inputs,kernel_size=3,padding='SAME')
        result = tf.concat([output_bypass, outputs], 3)

    result = tf.nn.relu(result)

    return result

def first_shufflenet_unit(inputs, groups, num_outputs):
    #get inputs channel
    inputs_channels = inputs.shape.as_list()[3]
    out_channels = num_outputs-inputs_channels

    outputs = slim.conv2d(inputs,out_channels,kernel_size=1)

    outputs = slim.batch_norm(outputs)
    outputs = tf.nn.relu(outputs)

    outputs = slim.separable_conv2d(outputs, None,kernel_size=3,depth_multiplier=1,stride=2)

    outputs = slim.batch_norm(outputs)

    outputs = slim.conv2d(outputs,out_channels,kernel_size=1)

    outputs = slim.batch_norm(outputs)

    output_bypass = slim.avg_pool2d(inputs,kernel_size=3,padding='SAME')

    result = tf.concat([output_bypass, outputs], 3)
    return result

def inference(images, 
                phase_train=True,
                groups=1,
                complexity_scale_factor=1,
                bottleneck_layer_size=128,
                weight_decay=0.0, 
                reuse=None):
    end_points = {}

    
    if groups == 1:
        stage2_out_channels = 144
    elif groups == 2:
        stage2_out_channels = 200
    elif groups == 3:
        stage2_out_channels = 240
    elif groups == 4:
        stage2_out_channels = 272
    elif groups == 8:
        stage2_out_channels = 384

    stage2_out_channels = int(stage2_out_channels * complexity_scale_factor)
    with tf.variable_scope("inference"):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.995,
                            epsilon=0.00001,
                            updates_collections=None,
                            variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                            is_training=phase_train):

            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                
                inputs = slim.conv2d(images, 24, kernel_size=3, stride=2,scope="Conv1")

                inputs = slim.batch_norm(inputs)    
                inputs = tf.nn.relu(inputs)
                inputs = slim.max_pool2d(inputs,kernel_size=3,stride=2,padding='SAME',scope="Maxpool1")
                
                
                with tf.variable_scope('stage2'):
                    with tf.variable_scope("unit1"):
                        inputs = first_shufflenet_unit(inputs,groups,stage2_out_channels)
                    for i in range(3):
                        with tf.variable_scope('unit' + str(i + 2)):
                            inputs = shufflenet_unit(inputs, groups,stride=1)

                with tf.variable_scope('stage3'):
                    with tf.variable_scope('unit1'):
                        inputs = shufflenet_unit(inputs, groups, stride=2)
                    for i in range(7):
                        with tf.variable_scope('unit' + str(i + 2)):
                            inputs = shufflenet_unit(inputs, groups, stride=1)

                with tf.variable_scope('stage4'):
                    with tf.variable_scope('unit1'):
                        inputs = shufflenet_unit(inputs, groups, stride=2)

                    for i in range(3):
                        with tf.variable_scope('unit' + str(i + 2)):
                            inputs = shufflenet_unit(inputs, groups, stride=1)


                h,w=inputs.shape.as_list()[1:3]
                inputs = slim.avg_pool2d(inputs,kernel_size=[h,w],stride=1,padding='VALID')
                inputs=slim.flatten(inputs)
                return inputs,end_points
                