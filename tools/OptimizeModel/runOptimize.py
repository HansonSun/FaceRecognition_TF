import os
import tensorflow as tf

for i in os.listdir("ConvertedModels"):
    if "pb" in i:
        print("please remove previous pb  file in ConvertedModels dir!!")
        exit()

os.system("python freeze_graph.py")

for i in os.listdir("ConvertedModels"):
    if "pb" in i:
        pbfile=i
        break

os.system("python optimize_for_inference.py --input=ConvertedModels/%s --output=ConvertedModels/%s.optimized --frozen_graph=True --input_names='input' --output_names='embeddings' "%(pbfile,pbfile))

print ("start to generate tfevents file...")
#generate the tfevents and node txt
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
with open("ConvertedModels/%s"%(pbfile+".optimized"),"r") as f:
    graphdef.ParseFromString(f.read())
    tf.import_graph_def(graphdef, name="")
    summary_write = tf.summary.FileWriter("./tfevents" , graph)
    with open("ConvertedModels/pbnode.txt",'w') as f:
        for node in graphdef.node:
            f.write("name: "+str(node.name)+"  op: "+str(node.op)+"\n" )

