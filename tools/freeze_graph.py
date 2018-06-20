from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import sys
from six.moves import xrange

def main(intput_model_name,output_model_name):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            meta_file ="./ToBeConvertModels/%s.ckpt.meta"%intput_model_name
            cpkt_file="./ToBeConvertModels/%s.ckpt"%intput_model_name
            saver = tf.train.import_meta_graph(meta_file)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), cpkt_file)

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile("%s.pb"%output_model_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('inference') or node.name.startswith('embeddings') or node.name.startswith('phase_train') or node.name.startswith('Bottleneck') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--intput_model_name",type=str,help="intput model name",default="")
    parser.add_argument("-o","--output_model_name",type=str,help="output model name",default="result")
    args=parser.parse_args(sys.argv[1:])
    main( args.intput_model_name,args.output_model_name )
