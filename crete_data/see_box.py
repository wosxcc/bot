import os
import numpy as np
import cv2 as cv


# paths='E:/BOT_Person/trainb'
paths='E:/BOT_Car/train'
# paths='./train'
# paths='./train'
for file in os.listdir(paths):
    if file[-4:]=='.txt': ## and file[:-4]>'00500' and file[:-4]>'E:/BOT_Person/trainb/000000480594.jpg' and file[:-4]>'000000480591'  and file[:-4]>'84000'
        new_box=''
        new_txt =open(paths+'/'+file)
        old_data = new_txt.read()
        print(paths+'/'+file[:-4]+'.jpg')
        img=cv.imread(paths+'/'+file[:-4]+'.jpg')
        for bbox in old_data.split('\n'):
            box=bbox.split(' ')
            if len(box)==5:
                box =[float(i) for i in box]

                img=cv.rectangle(img,(int((box[1]-box[3]/2)*img.shape[1]),int((box[2]-box[4]/2)*img.shape[0])),(int((box[1]+box[3]/2)*img.shape[1]),int((box[2]+box[4]/2)*img.shape[0])),(255,0,0),2 )
                img = cv.circle(img, (int((box[1]) * img.shape[1]), int((box[2] - box[4] / 2) * img.shape[0])), 2,
                                (255, 0, 255), -1)
        imgs = cv.resize(img ,(1280,720), interpolation=cv.INTER_CUBIC)
        cv.imshow(file,imgs)
        cv.waitKey()
        cv.destroyAllWindows()
