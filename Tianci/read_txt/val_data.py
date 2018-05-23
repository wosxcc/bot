import os
import numpy as np
import cv2 as cv


# paths='E:/BOT_Person/train'
paths='E:/BOT_Txt/train/'
for file in os.listdir(paths):
    if file[-4:]=='.txt'  : ## and file[:-4]>'00500'  and file[:-4]>'50100'
        new_box=''
        new_txt =open(paths+'/'+file)
        old_data = new_txt.read()
        print(paths+'/'+file[:-4]+'.jpg')
        img=cv.imread(paths+'/'+file[:-4]+'.jpg')
        for bbox in old_data.split('\n'):
            box=bbox.split(' ')
            if len(box)==5:
                box =[float(i) for i in box]

                # cv.circle(img, (int(box[1]*img.shape[1]), int(box[2]*img.shape[0])), 2, (0, 255, 0), -1)
                if box[0]==0:


                    cv.circle(img, (int(box[1] * img.shape[1]), int(box[2] * img.shape[0])), 5, (0, 255, 255), -1)
                    cv.circle(img, (int(box[3] * img.shape[1]), int(box[4] * img.shape[0])), 5, (0, 0, 255), -1)
                    print(int(box[1] * img.shape[1]) * 2 - int(box[3] * img.shape[1]),
                                    int(box[2] * img.shape[0]) * 2 - int(box[4] * img.shape[0]))
                    cv.circle(img, (int(box[1] * img.shape[1]) * 2 - int(box[3] * img.shape[1]),
                                    int(box[2] * img.shape[0]) * 2 - int(box[4] * img.shape[0])), 5, (0, 0, 255), -1)
                    # cv.line(img, (int(box[1] * img.shape[1]), int(box[2] * img.shape[0])),
                    #         (int(box[3] * img.shape[1]), int(box[4] * img.shape[0])), (0, 255, 255),2)
                else:
                    cv.circle(img, (int(box[1] * img.shape[1]), int(box[2] * img.shape[0])), 5, (0, 255, 255), -1)
                    cv.circle(img, (int(box[3] * img.shape[1]), int(box[4] * img.shape[0])), 5, (0, 255, 0), -1)
                    cv.circle(img, (int(box[1] * img.shape[1]) * 2 - int(box[3] * img.shape[1]),
                                    int(box[2] * img.shape[0]) * 2 - int(box[4] * img.shape[0])), 5, (0, 255, 0), -1)
                    # cv.line(img, (int(box[1] * img.shape[1]), int(box[2] * img.shape[0])),
                    #         (int(box[3] * img.shape[1]), int(box[4] * img.shape[0])), (255, 255, 0),2)
                # img=cv.rectangle(img,(int((box[1]-box[3]/2)*img.shape[1]),int((box[2]-box[4]/2)*img.shape[0])),(int((box[1]+box[3]/2)*img.shape[1]),int((box[2]+box[4]/2)*img.shape[0])),(255,0,0),2 )
        cv.imshow(file,img)
        cv.waitKey()
        cv.destroyAllWindows()
