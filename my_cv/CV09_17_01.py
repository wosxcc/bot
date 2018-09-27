import numpy as np
import cv2 as cv
img =cv.imread('50002.jpg')

# img_ = img[:, :, :].transpose((2, 0, 1)) #(3, 560, 560)
# print(img_[:,:10,:10])

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

kernel =[[1,-1,1],
         [1,-1,1],
         [1,-1,1],]
kernel = np.array(kernel,dtype='float32')
stride = 1
kernel_size = kernel.shape[0]
img_con =np.zeros((int((img.shape[0] - kernel_size + 1) / stride),int((img.shape[1]-kernel_size+1)/stride)),np.float32)
for x in range(int((img.shape[1]-kernel_size+1)/stride)):
    for y in range(int((img.shape[0] - kernel_size + 1) / stride)):
        img_con[y,x] = np.sum(np.dot(img[y*stride:y*stride+kernel_size,x*stride:x*stride+kernel_size],kernel))

img_con = img_con - np.min(img_con)
img_con = img_con / np.max(img_con) * 255
img_con = np.array(img_con, dtype='int')

imgc =np.zeros((int((img.shape[0] - kernel_size + 1) / stride),int((img.shape[1]-kernel_size+1)/stride)),np.uint8)

imgc[:,:] =img_con
cv.imshow('img',img)
cv.imshow('imgc',imgc)
cv.waitKey()
