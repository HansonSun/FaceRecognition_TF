#!/bin/sh
python optimize_for_inference.py \
--input=/opt/python-project/PycharmProjects/detect-test/models/183909.pb \
--output=/opt/python-project/PycharmProjects/detect-test/models/optimized_183909.pb \
--frozen_graph=True \
--input_names='image_tensor' \
--output_names='detection_boxes,detection_scores,detection_classes,num_detections'
