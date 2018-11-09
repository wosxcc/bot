import tensorflow as tf

import  numpy as np
import cv2 as cv
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def recognize(jpg_path, pb_file_path,img_w,img_h):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            image_batch = tf.placeholder(tf.float32, shape=(None, 160, 160, 3), name='image_batch')
            # label_batch = tf.placeholder(tf.int32, shape=(None,), name='batch_join')  'label_batch': label_batch,
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            input_map = {'image_batch': image_batch,  'phase_train': phase_train_placeholder}
            print('输出图')
            for node in output_graph_def.node:
                print(node.name)
            _ = tf.import_graph_def(output_graph_def, input_map=input_map, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            out_softmax = tf.get_default_graph().get_tensor_by_name("output/output:0")
            facess=[]
            for file in os.listdir(jpg_path):
                img_path = jpg_path + '/' + file
                img = cv.imread(img_path)
                img = cv.resize(img, (img_w, img_h), interpolation=cv.INTER_CUBIC)
                ximg = np.array(img, dtype='float32')
                ximg = (ximg - 127.5) / 128
                lfw_batch_size =1
                start_time = datetime.datetime.now()
                feed_dict = {image_batch: np.reshape(ximg, [1, img_w, img_h, 3]), labels: np.reshape(1, [1, 1])
                    , batch_size_placeholder: lfw_batch_size, phase_train_placeholder: False}
                img_out_softmax = sess.run(out_softmax, feed_dict=feed_dict)
                facess.append(img_out_softmax[0])
                print(img_out_softmax)
                print('耗时：', datetime.datetime.now() - start_time)

            for aface in facess:
                for bface in facess:
                    dist = np.sqrt(np.sum(np.square(aface -bface)))
                    print('距离为',dist)



recognize("E:/faceID/mface", "./botid/1012/botface.pb",160,160)
