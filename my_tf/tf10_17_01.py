import tensorflow as tf

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open('E:/XTF/Ccao/ConsoleApplication1/model/face28.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())

        tf.import_graph_def()

