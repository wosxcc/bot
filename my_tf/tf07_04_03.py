import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
from  my_tf.face72_smoll import face_net

os.environ['CUDA_VISIBLE_DEVICES']='0'  #设置GPU


model_path  = "E:/xbot/face_into/face72/face72/smoll/model.ckpt-98000" #设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前


def main():

    tf.reset_default_graph()

    input_node = tf.placeholder(tf.float32, shape=(96, 96, 3))
    input_node = tf.expand_dims(input_node, 0)
    flow = face_net(input_node)
    flow = tf.cast(flow, tf.uint8, 'out') #设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用

    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, model_path)

        #保存图
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        #把图和参数结构一起
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'out','save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

    print("done")

if __name__ == '__main__':
    main()