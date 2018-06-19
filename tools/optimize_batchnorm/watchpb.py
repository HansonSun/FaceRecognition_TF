import tensorflow as tf
from tensorflow.python.platform import gfile
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
with open("fjlsfjs.pb","r") as f:
    graphdef.ParseFromString(f.read())
    tf.import_graph_def(graphdef, name="")
    summary_write = tf.summary.FileWriter("./" , graph)


