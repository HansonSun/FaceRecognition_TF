import tensorflow as tf
import tensorflow.contrib.slim as slim


def MFM(net):
	net_channel=int (net.shape[-1]) 
	#Slice
	slice1,slice2 = tf.split(net,[int(net_channel/2),int(net_channel/2)],int(net.shape.ndims)-1 )
	#eltwise max
	eltwise=tf.maximum(slice1,slice2)
	return eltwise


def inference(image,dropout_keep_prob=0.8,is_training=True,scope="inference",weight_decay=0.0,bottleneck_layer_size=256):
	end_poins=[]
	with tf.variable_scope(scope, 'inference', [image]):
		with slim.arg_scope([slim.conv2d, slim.fully_connected],activation_fn=None,
						weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
						weights_regularizer=slim.l2_regularizer(weight_decay) ):
			#conv1
			net=slim.conv2d(image,96,[5,5],stride=1,scope="conv1",padding="SAME")
			#MFM1
			net=MFM(net)
			#pool1
			net=slim.max_pool2d(net,[2,2],stride=2,scope="pool1")
			#conv2a
			net=slim.conv2d(net,96,[1,1],stride=1,scope="conv2a",padding="SAME")
			#MFM2a
			net=MFM(net)
			#conv2
			net=slim.conv2d(net,192,[3,3],stride=1,scope="conv2",padding="SAME")
			#MFM2
			net=MFM(net)
			#pool2
			net=slim.max_pool2d(net,[2,2],stride=2,scope="pool2")
			#conV3a
			net=slim.conv2d(net,192,[1,1],stride=1,scope="conv3a",padding="SAME")
			#MFM3a
			net=MFM(net)
			#conv3
			net=slim.conv2d(net,384,[3,3],stride=1,scope="conv3",padding="SAME")
			#MFM3
			net=MFM(net)
			#pool3
			net=slim.max_pool2d(net,[2,2],stride=2,scope="pool3")
			#conv4a
			net=slim.conv2d(net,384,[1,1],stride=1,scope="conv4a",padding="SAME")
			#MFM4a
			net=MFM(net)
			#conv4
			net=slim.conv2d(net,256,[3,3],stride=1,scope="conv4",padding="SAME")
			#MFM4
			net=MFM(net)
			#conv5a
			net=slim.conv2d(net,256,[1,1],stride=1,scope="conv5a",padding="SAME")
			#MFM5a
			net=MFM(net)
			#conv5
			net=slim.conv2d(net,256,[3,3],stride=1,scope="conv5")
			#pool4
			net=slim.max_pool2d(net,[2,2],stride=2,scope="pool4")
			#fc1
			net=slim.flatten(net)
			#droupout
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='Dropout')
			net=slim.fully_connected(net,bottleneck_layer_size*2,activation_fn=None,scope="fc1")
			#MFM_FC1
			net=MFM(net)

			net=slim.flatten(net)
	return  net,end_poins