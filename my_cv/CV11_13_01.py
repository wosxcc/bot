import cv2 as cv
import numpy as np

zuozi =np.array([[[128,173],[137,300],[242,316],[434,316],[440,153],[268,165]]], dtype = np.int32)

xiansqi=np.array([[[205,182],[173,206],[175,227],[274,221],[292,198],[272,176]]], dtype = np.int32)

shubiao =np.array([[[228,236],[236,278],[302, 270],[299, 233]]], dtype = np.int32)


bitong =np.array([[[327,181],[327,211],[355,210],[357,180]]], dtype = np.int32)


cap =cv.VideoCapture('E:/Desk_Set/55.mp4')
while True:
    ret, fram = cap.read()
    fram=cv.resize(fram,(640,360),cv.INTER_CUBIC)
    cv.polylines(fram, zuozi, 1, (255,0,0),2)
    cv.polylines(fram, xiansqi, 1, (255,255,0), 2)
    cv.polylines(fram, shubiao, 1, (255,0 , 255), 2)
    cv.polylines(fram, bitong, 1, (0, 255, 255), 2)

    cv.imshow("imgs",fram)
    cv.waitKey(0)










