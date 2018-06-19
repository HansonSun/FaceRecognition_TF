#!/bin/sh
python optimize_for_inference.py \
--input=/opt/python-project/PycharmProjects/detect-test/models/eightth.pb \
--output=/opt/python-project/PycharmProjects/detect-test/models/optimized_eightth_new_25.pb \
--frozen_graph=True \
--input_names='input_tensor' \
--output_names='embeddings'
