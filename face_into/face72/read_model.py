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
            print(str(output_graph_def)[:5000])

            print(str(output_graph_def)[5000:])
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("input:0")
            print (input_x)
            out_softmax = sess.graph.get_tensor_by_name("output/output:0")
            # out_softmax = sess.graph.get_tensor_by_name("resnet_v2_50/output:0")
            print (out_softmax)
            # out_label = sess.graph.get_tensor_by_name("output:0")
            # print (out_label)

            for file in os.listdir(jpg_path):
                img_path = jpg_path + '/' + file
                img = cv.imread(img_path)
                img =cv.resize(img, (img_w,img_h), interpolation=cv.INTER_CUBIC)
                ximg =np.array(img,dtype='float32')
                # ximg=(ximg-127.5) / 256
                ximg /=255

                start_time =datetime.datetime.now()
                img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(ximg, [-1, img_w,img_h, 3])})
                print(img_out_softmax)
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






recognize("E:/xbot/face_into/face68/image_test", "./face72/facell/face72.pb",96,96)  ##../face68/image_test   E:/face72/trainb    E:/xbot/face_into/face68/image_test

## E:/xbot/face_into/res_face/facesmoll/log/face72.pb

# E:/xbot/face_into/res_face/faceres/log
#./face72/face_0802/face72.pb

#  ./face72/facell/face72.pb

## E:/xbot/face_into/face72/SmallB/face72.pb

# E:/xbot/face_into/res_face/faceres/log