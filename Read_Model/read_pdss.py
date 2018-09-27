import tensorflow as tf

import numpy as np
import cv2 as cv
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lfw_batch_size = 1
def get_face_data():
    face_txt = 'E:/Model/facenet-master/src/txt_face'
    face_data ={}
    for file in os.listdir(face_txt):
        if file[-3:] == 'txt':
            txt_open = open(face_txt+'/'+file)
            txt_read = txt_open.read()
            face_number = txt_read.split(' ')
            face_data[file[:-3]] = [float(i) for i in face_number[:-1]]
    return face_data


def recognize(jpg_path, pb_file_path, img_w, img_h):
    face_data = get_face_data()

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())

            print(str(output_graph_def)[:5000])
            print()
            print(str(output_graph_def)[-5000:])
            image_batch = tf.placeholder(tf.float32, shape=(None, 160, 160, 3), name='batch_join')
            label_batch = tf.placeholder(tf.int32, shape=(None,), name='batch_join')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            # print('出错钱')
            _ = tf.import_graph_def(output_graph_def, input_map=input_map, name="")
            # print('出错猴')

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            out_softmax = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            for file in os.listdir(jpg_path):
                img_path = jpg_path + '/' + file
                img = cv.imread(img_path)
                img = cv.resize(img, (img_w, img_h), interpolation=cv.INTER_CUBIC)
                ximg = np.array(img, dtype='float32')
                ximg = (ximg - 127.5) / 128
                start_time = datetime.datetime.now()
                feed_dict = {image_batch: np.reshape(ximg, [1, img_w, img_h, 3]), labels: np.reshape(1, [1, 1])
                    , phase_train_placeholder: False,batch_size_placeholder: lfw_batch_size}
                img_out_softmax = sess.run(out_softmax, feed_dict=feed_dict)
                print('耗时：', datetime.datetime.now() - start_time)
                print(img_out_softmax.shape)
                img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
                min_name = ''
                min_dist = 10.0

                for aface in face_data:
                    dist = np.sqrt(np.sum(np.square(face_data[aface] - img_out_softmax[0])))
                    if min_dist > dist:
                        min_dist = dist
                        min_name = aface
                print('最小距离为:', min_dist)
                if min_dist < 0.7:
                    cv.putText(img, min_name, (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


                cv.imshow('img', img)
                cv.waitKey()


recognize("E:/face72/trains", "./model/facenet.pb", 160,
          160)  ##../face68/image_test   E:/face72/trainb    E:/xbot/face_into/face68/image_test
