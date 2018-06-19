#!/bin/sh
python optimize_for_inference.py \
--input=/opt/Tensorflow-Examples/examples/3_NeuralNetworks/train_dir/pb_file/classify_19999.pb \
--output=/opt/Tensorflow-Examples/examples/3_NeuralNetworks/train_dir/pb_file/optimized_classify_19999.pb \
--frozen_graph=True \
--input_names='input_tensor' \
--output_names='output_tensor'
