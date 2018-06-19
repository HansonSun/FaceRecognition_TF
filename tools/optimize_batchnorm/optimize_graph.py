import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
import re
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from collections import defaultdict
from tensorflow.python.framework import tensor_util
import numpy as np
import math
from tensorflow.core.framework import attr_value_pb2


def get_input_node_map(input_graph_def):
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map.keys():
            input_node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)
    return input_node_map

def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name

def get_control_map(input_graph_def):
    control_map = defaultdict(list)
    for node in input_graph_def.node:
        for inp in node.input:
             if inp.startswith('^'):
                 control_map[inp[1:]].append(node.name)

    return control_map

def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Node named '%s' should be a Const op for values_from_const." %
        node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value

def remove_assert_depend(input_graph_def):
    names_to_remove = {}
    for node in input_graph_def.node:
        if node.op == 'Assert':
            names_to_remove[node.name] = True

    nodes_after_removal = []
    for node in input_graph_def.node:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)

    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.node.extend(nodes_after_removal)
    return output_graph_def

def remove_identity(input_graph_def, protected_nodes=None):
    nodes = input_graph_def.node
    types_to_splice = {"Identity": True}
    names_to_splice = {}
    control_map = get_control_map(input_graph_def)
    for node in nodes:
        if node.op in types_to_splice and node.name not in protected_nodes:

            if node.name in control_map:
                continue
            has_control_edge = False
            for input_name in node.input:
                if re.match(r"^\^", input_name):
                    has_control_edge = True
            if not has_control_edge:
                names_to_splice[node.name] = node.input[0]

    nodes_after_splicing = []
    for node in nodes:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:

            input_name = re.sub(r"^\^", "", full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub(r"^\^", "", full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph


def bn_fold(input_graph_def, conv_name, weight_name, mean_name,
            var_name, beta_name, gamma_name, epsilon_name, add_name):
    input_node_map = get_input_node_map(input_graph_def)

    skip_ops = [conv_name, weight_name, mean_name,
                var_name, beta_name, gamma_name, epsilon_name, add_name]
    skip_ops.extend([])

    try:
        conv_op = input_node_map[conv_name]
        weights_op = input_node_map[weight_name]
        mean_op = input_node_map[mean_name]
        var_op = input_node_map[var_name]
        beta_op = input_node_map[beta_name]
        gamma_op = input_node_map[gamma_name]
        epsilon_op = input_node_map[epsilon_name]
        add_op = input_node_map[add_name]
    except KeyError as e:
        print("node %s not in graph"%e)
        return [],[]

    weights = values_from_const(weights_op)
    mean_value = values_from_const(mean_op)
    var_value = values_from_const(var_op)
    beta_value = values_from_const(beta_op)
    gamma_value = values_from_const(gamma_op)
    variance_epsilon_value = values_from_const(epsilon_op)

    new_ops = []

    scale_value = (
        (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
        gamma_value)

    offset_value = (-mean_value * scale_value) + beta_value
    scaled_weights = np.copy(weights)
    it = np.nditer(
        scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

      if conv_op.op == "DepthwiseConv2dNative":
        current_scale = scale_value[it.multi_index[2]]
      else:
        current_scale = scale_value[it.multi_index[3]]
      it[0] *= current_scale
      it.iternext()
    scaled_weights_op = node_def_pb2.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = weights_op.name
    scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            scaled_weights, weights.dtype.type, weights.shape)))

    new_conv_op = node_def_pb2.NodeDef()
    new_conv_op.CopyFrom(conv_op)
    offset_op = node_def_pb2.NodeDef()
    offset_op.op = "Const"
    offset_op.name = conv_op.name + "_bn_offset"
    offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
    offset_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            offset_value, mean_value.dtype.type, offset_value.shape)))

    new_add_op = node_def_pb2.NodeDef()
    new_add_op.CopyFrom(add_op)
    del new_add_op.input[:]
    new_add_op.input.extend([new_conv_op.name, offset_op.name])

    new_ops.extend([scaled_weights_op, new_conv_op, offset_op, new_add_op])
    return skip_ops,new_ops

if __name__ == "__main__":

    input_pb_file = '/opt/python-project/PycharmProjects/detect-test/models/face_mobile_8_18_512590_graph.pb'
    output_pb_file = '/opt/python-project/PycharmProjects/detect-test/models/opt_512590.pb'

    input_node_names = ["image_tensor"]
    output_node_names = ['detection_boxes','detection_scores','detection_classes', 'num_detections']
    input_graph_def = graph_pb2.GraphDef()
    with open(input_pb_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = remove_assert_depend(input_graph_def)

    output_graph_def = strip_unused_lib.strip_unused(
        output_graph_def, input_node_names=input_node_names,
        output_node_names=output_node_names,
        placeholder_type_enum=dtypes.uint8.as_datatype_enum)

    output_graph_def = remove_identity(output_graph_def, output_node_names)

    skip_ops = []
    new_ops = []


    conv_names = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_pointwise/Conv2D'%i for i in range(1,14)] +\
                ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_depthwise/depthwise'%i for i in range(1,14)]+\
                ["FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D",
                 "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2D",
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2D',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2D',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2D'
                 ]
    weight_names = ['FeatureExtractor/MobilenetV1/Conv2d_%s_pointwise/weights'%i for i in range(1,14)] +\
                ["FeatureExtractor/MobilenetV1/Conv2d_%s_depthwise/depthwise_weights"%i for i in range(1,14)]+\
                  ['FeatureExtractor/MobilenetV1/Conv2d_0/weights',
                    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/weights',
                   'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/weights',
                   'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/weights',
                   'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/weights'
                   ]
    mean_names = ['FeatureExtractor/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/moving_mean'%i for i in range(1,14)] +\
                ["FeatureExtractor/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/moving_mean"%i for i in range(1,14)]+\
                ['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_mean',
                    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/moving_mean',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/moving_mean',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/moving_mean',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/moving_mean'
                ]
    var_names = ['FeatureExtractor/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/moving_variance'%i for i in range(1,14)]+\
                ["FeatureExtractor/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/moving_variance"%i for i in range(1,14)]+\
            ['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_variance',
                'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/moving_variance',
             'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/moving_variance',
             'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/moving_variance',
             'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/moving_variance'
            ]
    beta_names = ['FeatureExtractor/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/beta'%i for i in range(1,14)] +\
                ["FeatureExtractor/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/beta"%i for i in range(1,14)] +\
                ['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/beta',
                    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/beta',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/beta',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/beta',
                 'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/beta'
                 ]
    gamma_names = ['FeatureExtractor/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/gamma'%i for i in range(1,14)] +\
                ["FeatureExtractor/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/gamma"%i for i in range(1,14)]+\
        ['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/gamma',
            'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/gamma',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/gamma',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/gamma',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/gamma'
         ]
    epsilon_names = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/batchnorm/add/y'%i for i in range(1,14)] +\
                    ["FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/batchnorm/add/y"%i for i in range(1,14)]+\
        ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add/y',
            'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/batchnorm/add/y',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/batchnorm/add/y',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/batchnorm/add/y',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/batchnorm/add/y'
         ]
    add_names = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_pointwise/BatchNorm/batchnorm/add_1'%i for i in range(1,14)] +\
                ["FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_%s_depthwise/BatchNorm/batchnorm/add_1"%i for i in range(1,14)]+\
        ["FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add_1",
            'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/batchnorm/add_1',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/batchnorm/add_1',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/batchnorm/add_1',
         'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/batchnorm/add_1'
         ]


    for conv_name, weight_name, mean_name,var_name, beta_name, gamma_name, epsilon_name, add_name in zip(conv_names,
                            weight_names, mean_names,var_names, beta_names, gamma_names, epsilon_names, add_names):

        skip_op, new_op = bn_fold(output_graph_def, conv_name, weight_name, mean_name,
                                var_name, beta_name, gamma_name, epsilon_name, add_name)
        skip_ops.extend(skip_op)
        new_ops.extend(new_op)

    result_graph_def = graph_pb2.GraphDef()
    for node in output_graph_def.node:
        if node.name in skip_ops:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])

    result_graph_def.node.extend(new_ops)
    output_graph_def = result_graph_def
    output_graph_def = strip_unused_lib.strip_unused(
        output_graph_def, input_node_names=input_node_names,
        output_node_names=output_node_names,
        placeholder_type_enum=dtypes.uint8.as_datatype_enum)

    with open(output_pb_file,'wb') as f:
        f.write(output_graph_def.SerializeToString())


