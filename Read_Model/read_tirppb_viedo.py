import tensorflow as tf

import numpy as np
import cv2 as cv
import datetime
import os
import math
from MY_Function.cv_function import dram_chinese

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
lfw_batch_size = 1
def get_face_data():
    face_txt = './txt_face'
    face_data ={}
    for file in os.listdir(face_txt):
        if file[-3:] == 'txt':
            txt_open = open(face_txt+'/'+file)
            txt_read = txt_open.read()
            face_number = txt_read.split(' ')
            face_data[file[:-4]] = [float(i) for i in face_number[:-1]]
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

            _ = tf.import_graph_def(output_graph_def, name="") #
            # print('出错猴')

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            cap = cv.VideoCapture(0)  # 'D:/bot_opencv/opencv_one/opencv_one/image/1.mp4' # 加载摄像头录制
            success, frame = cap.read()
            classifier = cv.CascadeClassifier(
                "E:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml")  # 确保此xml文件与该py文件在一个文件夹下，否则将这里改为绝对路径
            while success:
                success, frame = cap.read()
                size = frame.shape[:2]
                image = np.zeros(size, dtype=np.float16)
                image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cv.equalizeHist(image, image)
                divisor = 8
                h, w = size
                minSize = (w // divisor, h // divisor)
                faceRects = classifier.detectMultiScale(image, 1.2, 2, cv.CASCADE_SCALE_IMAGE, minSize)
                if len(faceRects) > 0:
                    for faceRect in faceRects:
                        x, y, w, h = faceRect
                        if w * h > 10000:


                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(x - 16, 0)
                            bb[1] = np.maximum(y - 16, 0)
                            bb[2] = np.minimum(x + w + 16, frame.shape[1])
                            bb[3] = np.minimum(y + h + 16, frame.shape[0])
                            cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                            cropped = cv.resize(cropped, (160, 160), interpolation=cv.INTER_CUBIC)
                            cropped = (np.array(cropped, dtype='float32') - 127.5) / 128
                            start_time = datetime.datetime.now()
                            feed_dict = {images_placeholder: np.reshape(cropped, [1, img_w, img_h, 3]),
                                         phase_train_placeholder: False}
                            img_out_softmax = sess.run(embeddings, feed_dict=feed_dict)
                            print('耗时：', datetime.datetime.now() - start_time)
                            print(img_out_softmax.shape)
                            # img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
                            min_name = ''
                            min_dist = 10.0

                            for aface in face_data:
                                # dist = np.sqrt(np.sum(np.square(face_data[aface] - img_out_softmax[0])))
                                dot_product = 0.0;
                                normA = 0.0;
                                normB = 0.0;
                                for a, b in zip(face_data[aface], img_out_softmax[0]):
                                    dot_product += a * b
                                    normA += a ** 2
                                    normB += b ** 2
                                if normA == 0.0 or normB == 0.0:
                                    print('居然都为0')
                                else:
                                    dist=  dot_product / ((normA * normB) ** 0.5)
                                    dist  = math.acos(dist)
                                    print('与', min_name, '最小距离为:', min_dist)
                                    if min_dist > dist:
                                        min_dist = dist
                                        min_name = aface
                            print('得到结果与',min_name,'最小距离为:', min_dist)
                            if min_dist < 0.9:
                                frame =dram_chinese(frame,min_name,bb[0],bb[1],30,(255, 0, 0))
                                # cv.putText(frame, min_name, (bb[0], bb[1]+30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


                cv.imshow('frame', frame)
                cv.waitKey(10)


recognize("E:/face72/trains", "./model/facenet.pb", 160,160)
