import tensorflow as tf

import  numpy as np
import cv2 as cv
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
lfw_batch_size = 1

# def recognize(jpg_path, pb_file_path,img_w,img_h):
#
#     with tf.Graph().as_default():
#         output_graph_def = tf.GraphDef()
#         with open(pb_file_path, "rb") as f:
#             output_graph_def.ParseFromString(f.read())
#             _ = tf.import_graph_def(output_graph_def, name="") #
#
#         with tf.Session() as sess:
#             init = tf.global_variables_initializer()
#             sess.run(init)
#
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             for file in os.listdir(jpg_path):
#                 img_path = jpg_path + '/' + file
#                 img = cv.imread(img_path)
#                 img =cv.resize(img, (img_w,img_h), interpolation=cv.INTER_CUBIC)
#                 ximg =np.array(img,dtype='float32')
#                 ximg=(ximg-127.5) / 128
#
#                 feed_dict = {images_placeholder: ximg, phase_train_placeholder: False}
#                 emb_array= sess.run(embeddings, feed_dict=feed_dict)
#

def recognize(jpg_path, pb_file_path,img_w,img_h):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            print(str(output_graph_def)[-155000:])
            for node in  output_graph_def.node:
                print(node.name)
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for file in os.listdir(jpg_path):
                img_path = jpg_path + '/' + file
                img = cv.imread(img_path)
                img =cv.resize(img, (img_w,img_h), interpolation=cv.INTER_CUBIC)
                ximg =np.array(img,dtype='float32')
                ximg=(ximg-127.5) / 128

                start_time =datetime.datetime.now()
                feed_dict = {images_placeholder: np.reshape(ximg, [1, img_w, img_h, 3]),phase_train_placeholder: False}
                img_out_softmax= sess.run(embeddings, feed_dict=feed_dict)

                print(img_out_softmax)
                print('耗时：',datetime.datetime.now()-start_time)
                img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)


                img_out_softmax=img_out_softmax.reshape(1,-1)
                img_out_softmax[0] = (img_out_softmax[0] + 1) / 2

                img_out_softmax[0]
                print(img_out_softmax)
                for i in range(int(len(img_out_softmax[0]) / 2) - 1):
                    cv.circle(img, (int(img_out_softmax[0][i * 2] * img.shape[1]), int(img_out_softmax[0][i * 2 + 1] * img.shape[0])), 2,
                                     (0, 255, 255), -1)


                biaoq ='None'
                if img_out_softmax[0][0]>= 0.8 and img_out_softmax[0][0]<1.6:
                    biaoq = 'Smile'
                elif img_out_softmax[0][0]>=1.6:
                    biaoq = 'Laugh'
                biaoq+=':' + str(img_out_softmax[0][1])
                img = cv.putText(img, biaoq, (0, 30), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))

                print("img_out_softmax:", img_out_softmax[0][:2])
                cv.imshow('img',img)
                cv.waitKey()


recognize("E:/face72/trains", "./model/facenet.pb",160,160)
