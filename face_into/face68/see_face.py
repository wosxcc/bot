import numpy as np
import cv2 as cv
import os

        ###一共194个关键点

file_path = 'E:/face68/txt'
count_img=0
output_txt = open('E:/face68/train.txt', 'w')
for file_txt in os.listdir(file_path):
    txt_open = open(file_path+'/'+file_txt,'r')
    txt_read = txt_open.read()
    txt_lines = txt_read.split('\n')
    # print(txt_lines)
    img =cv.imread('E:/face68/face/'+txt_lines[0]+'.jpg')
    # print(len(txt_lines))
    print(file_txt)

    xdata=[]
    ydata=[]
    for tline in txt_lines[1:-1]:
        if len(tline) >3:
            nx,ny=tline.split(',')
            xdata.append(int(float(nx)))
            ydata.append(int(float(ny)))

    minx=max(min(xdata),0)
    miny=max(min(ydata),0)
    maxx=min(max(xdata),img.shape[1])
    maxy=min(max(ydata),img.shape[0])

    pianchay=int((maxy-miny)/4)
    miny = max(0,miny-pianchay)
    maxy= min(maxy,img.shape[0])

    pianchax = int((maxy - miny) / 8)
    minx=max(0,minx-pianchax)
    maxx = min(maxx + pianchax, img.shape[1])
    imgxx=img[miny:maxy,minx:maxx,:]
    # print( 'minx：',minx,'maxx;',maxx,'img:',imgxx.shape[1])
    # print( 'miny：',minx,'maxy;',maxy,'img:',imgxx.shape[0])
    out_into=''

    for tline in txt_lines[59:115]:     # 嘴巴
        if len(tline) >3:
            nx,ny=tline.split(',')
            outx=(float(nx)-minx)/imgxx.shape[1]
            outy=(float(ny)-miny)/imgxx.shape[0]
            out_into+=' '+str(outx)+' '+str(outy)
            # img=cv.circle(img,(int(float(nx)),int(float(ny))),4,(0,0,255),-1)
    # for tline in txt_lines[115:155]:    # 眼睛
    #     if len(tline) >3:
    #         nx,ny=tline.split(',')
    #         img=cv.circle(img,(int(float(nx)),int(float(ny))),4,(0,255,0),-1)
    # for tline in txt_lines[155:]:       # 眉毛
    #     if len(tline) >3:
    #         nx,ny=tline.split(',')
    #         img=cv.circle(img,(int(float(nx)),int(float(ny))),4,(0,255,255),-1)

    out_imgname = str(count_img)+'.jpg'
    for i in range(8 - len(out_imgname)):
        out_imgname = '0' + out_imgname

    output_txt.write(out_imgname+ out_into+'\n')
    imgxx=cv.resize(imgxx,(96,96),interpolation=cv.INTER_CUBIC)
    cv.imwrite('E:/face68/trainb/'+out_imgname,imgxx)
    count_img+=1
    img=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
    # cv.imshow(txt_lines[0],img)
    # cv.imshow('simg',imgxx)
    # cv.waitKey()
    # cv.destroyAllWindows()
    txt_open.close()

output_txt.close()




