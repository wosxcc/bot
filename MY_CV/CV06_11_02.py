import cv2 as cv

import numpy as np
import  sys

(major_ver, minor_ver, subminor_ver)=(cv.__version__).split('.') ##获取版本信息

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW']
tracker_type = tracker_types[2]

if  int(minor_ver)>=3 and int(major_ver)>2:
    if tracker_type=='BOOSTING':
        tracker=cv.TrackerBoosting_create()

    if tracker_type=='MIL':
        tracker=cv.TrackerMIL_create()
    if tracker_type=='KCF':
        tracker=cv.TrackerKCF_create()
    if tracker_type=='TLD':
        tracker=cv.TrackerTLD_create()
    if tracker_type=='MEDIANFLOW':
        tracker=cv.TrackerMEDIANFLOW_create()

video= cv.VideoCapture(0) #'D:/pproject/ppop/image/2.mp4'
bbox = [(287, 23, 86, 320),(200,200,60,60)]
print(bbox)
while True:
    res, img = video.read()
    # k = cv.waitKey(0) & 0xFF
    # if k==ord('d'):
    #     cv.waitKey()
    #     bbox = cv.selectROI(img, False)
    timer =cv.getTickCount()

    ok,bbox=tracker.update(img)
    fps =cv.getTickFrequency()/(cv.getTickCount()-timer)

    if ok:
        p1 = (int(bbox[0]),int(bbox[1]))
        p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
        cv.rectangle(img, p1, p2 ,(255,0,0),2,1)

    cv.putText(img, tracker_type + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    # Display FPS on frame
    cv.putText(img, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    # Display result
    cv.imshow("Tracking", img)
    cv.waitKey(10)









