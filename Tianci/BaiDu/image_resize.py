import os
import cv2 as cv

start_path = 'E:/Model/deeplab/Database/SegmentationClass'
end_path =  'E:/Model/deeplab/Database640x320/SegmentationClass'


for imfile in os.listdir(start_path):
    img = cv.imread(start_path+'/'+imfile)


    # cv.imshow('img',img)

    imgresize = cv.resize(img,(640,320),cv.INTER_CUBIC)
    # cv.imshow('imgresize',imgresize)

    cv.imwrite(end_path+'/'+imfile,imgresize)
    # cv.waitKey()







