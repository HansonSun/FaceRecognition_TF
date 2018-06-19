import tensorflow  as tf
graph=tf.get_default_graph()
test=tf.GraphDef()

with open("112x112_glasses_optimize.pb") as f:
    test.ParseFromString(f.read())
    tf.import_graph_def(test)
    tf.summary.FileWriter("./" , graph)

