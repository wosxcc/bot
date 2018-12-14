import cv2 as cv
import numpy as np
import random






def file_to_data(x_data,y_data,size1,size2):


    x_read =[]
    y_read =[]
    for i in range(len(x_data)):
        img = cv.imread(x_data[i])
        img = cv.resize(img, (size1, size1), cv.INTER_CUBIC)

        x_rand = random.randint(0, (size1-size2)/2)
        y_rand = random.randint(0, (size1-size2)/2)
        img = img[y_rand:size1 - y_rand, x_rand:size1 - x_rand, :]


        img = cv.resize(img, (size2, size2), cv.INTER_CUBIC)

        x_read.append(img[:, :, ::-1].transpose((2, 0, 1)))
        y_read.append(y_data[i])
        # np.array(img[:, :, ::-1].transpose((2, 0, 1)), dtype=np.float32)
    x_read = ((np.array(x_read, dtype=np.float32) - 127.5) / 128)
    y_read = np.array(y_read, np.long)
    return x_read, y_read


# X_data_flie=open('dogcat.txt').read().split('\n')
#
# BATCH_SIZE =64
# batch_img = []
# batch_lab = []
# for ai in range(BATCH_SIZE):
#     xxx = random.randint(0, len(X_data_flie) - 1)
#     batch_img.append(X_data_flie[xxx].split(' ')[0])
#     batch_lab.append(X_data_flie[xxx].split(' ')[1])
# batch_x, batch_y = file_to_data(batch_img, batch_lab,124,112)
#
# print(batch_x.shape)
# print(batch_y.shape)