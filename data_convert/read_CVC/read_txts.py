import os
import cv2 as cv
import numpy as np

paths='E:/person/CVC05-PartOcclusion/Annotations/txt'
imgpath='E:/person/CVC05-PartOcclusion/FramesPos'
pathss='E:/BOT_Person/trainb/'
now_count=8000
for file in os.listdir(paths):

    img =cv.imread(imgpath+'/'+file[:-4]+'.png')
    file_into = open(paths+'/'+file)
    ssh_inta=file_into.read()
    person_bbox=ssh_inta.split('person ')
    xxc =0
    out_txt=''
    for bbox in person_bbox:
        if  xxc >=1:
            box=bbox.split(' ')
            box=box[:-1]
            box =[int(i) for i in box]
            img_X=float((box[0] + box[2]/2)/img.shape[1])
            img_Y=float((box[0]+ box[3]/2 )/img.shape[0])
            img_W=float(box[2]/img.shape[1])
            img_H=float(box[3]/img.shape[0])
            out_txt+='0 '+str(img_X)+' '+str(img_Y)+' '+str(img_W)+' '+str(img_H)+'\n'
            img=cv.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),2)
        xxc+=1
    print()
    print(out_txt)
    if len(out_txt)>3:
        now_name=str(now_count)
        for i in range(5-len(now_name)):
            now_name='0'+now_name

        now_count+=1
        out_file=pathss+now_name
        out_txts = open(out_file+'.txt', 'w')
        out_txts.write(out_txt)
        out_txts.close()
        # image = cv.resize(img, (416,416), interpolation=cv.INTER_CUBIC)
        cv.imwrite(out_file+'.jpg',img)




