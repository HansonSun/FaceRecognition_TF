import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

reader=pywrap_tensorflow.NewCheckpointReader("./ToBeConvertModels/10.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()
f=open("node.txt","w")
for key in var_to_shape_map:
	print( key )
	f.write(key+"\n")
	#print("tensor_name: ", key)
	#print(reader.get_tensor(key))
#output=chkp.print_tensors_in_checkpoint_file("./vgg_16.ckpt",tensor_name='', all_tensors="True",all_tensor_names=True)
#print (len(output))
