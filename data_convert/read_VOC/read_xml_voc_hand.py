from xml.etree import ElementTree as et
import  xml.dom.minidom
import os
import cv2 as cv


COUNT_person=0
COUNT_bicycle=0
COUNT_car=0
COUNT_motorbike=0
# xml_path='E:/xcc_download\VOC2007\Annotations'
# img_path='E:/xcc_download\VOC2007\JPEGImages'


xml_path='E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'
img_path='E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
now_count=0
for file_name in os.listdir(xml_path):
    print(xml_path+'/'+file_name)
    per=et.parse(xml_path+'/'+file_name)
    img=cv.imread(img_path+'/'+file_name[:-4]+'.jpg')
    cbndbox=per.findall('./object/part')
    input_txt=''
    for one_bndbox in cbndbox:
        for child in one_bndbox.getchildren():
            if str(child.text)=='hand':
                person_box= one_bndbox.findall('bndbox')
                nbbox=[]
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))

                img_X = float((nbbox[0] + nbbox[2]) / img.shape[1]/2)
                img_Y = float((nbbox[1] + nbbox[3]) / img.shape[0]/2)
                img_W = float((nbbox[2] - nbbox[0]) / img.shape[1])
                img_H = float((nbbox[3] - nbbox[1]) / img.shape[0])
                input_txt += '0 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'


    if len(input_txt)>3:
        now_name=str(now_count)
        for i in range(5-len(now_name)):
            now_name='0'+now_name

        now_count+=1
        out_file='E:/Hand/train/'+now_name
        out_txt = open(out_file+'.txt', 'w')
        out_txt.write(input_txt)
        out_txt.close()
        cv.imwrite(out_file+'.jpg',img)
        # cv.imshow('img',img)
        # cv.waitKey()
        # cv.destroyAllWindows()
