import tensorflow as tf

import  numpy as np
import cv2 as cv
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lfw_batch_size = 1
def recognize(jpg_path, pb_file_path,img_w,img_h):
    image_batch = tf.placeholder(tf.float32, shape=(None, 160, 160, 3), name='batch_join')
    label_batch = tf.placeholder(tf.int32, shape=(None,), name='batch_join')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def,input_map = input_map, name="")
            print(str(output_graph_def)[:5000])
            print()
            print(str(output_graph_def)[-5000:])
            # print(_)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # input_x = sess.graph.get_tensor_by_name("fifo_queue:0")

            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')

            # out_softmax = sess.graph.get_tensor_by_name("embeddings:0")
            out_softmax = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            for file in os.listdir(jpg_path):
                img_path = jpg_path + '/' + file
                img = cv.imread(img_path)
                img =cv.resize(img, (img_w,img_h), interpolation=cv.INTER_CUBIC)
                ximg =np.array(img,dtype='float32')
                ximg=(ximg-127.5) / 128

                start_time =datetime.datetime.now()
                feed_dict = {batch_size_placeholder:lfw_batch_size, phase_train_placeholder: False
                        ,image_batch: np.reshape(ximg, [1, img_w, img_h, 3]),labels:np.reshape(1,[1, 1])}
                # feed_dict = {image_paths_placeholder: image_paths_array, labels: labels_array,
                #              phase_train_placeholder: False}
                img_out_softmax, lab = sess.run([out_softmax, labels], feed_dict=feed_dict)
                # img_out_softmax = sess.run(out_softmax, feed_dict={batch_size_placeholder:np.reshape(1,[ 1])
                #     ,image_batch:np.reshape(ximg,[1, img_w,img_h, 3]),label_batch:np.reshape(1,[1, 1])
                #     ,phase_train_placeholder: False})

                print('耗时：',datetime.datetime.now()-start_time)
                img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
                img_out_softmax=img_out_softmax.reshape(1,-1)
                for i in range(int(len(img_out_softmax[0]) / 2) - 1):
                    cv.circle(img, (int(img_out_softmax[0][2+i * 2] * img.shape[1]), int(img_out_softmax[0][2+i * 2 + 1] * img.shape[0])), 2,
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


recognize("E:/face72/trains", "./model/facenet.pb",160,160)  ##../face68/image_test   E:/face72/trainb    E:/xbot/face_into/face68/image_test
