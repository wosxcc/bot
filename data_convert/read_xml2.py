from xml.etree import ElementTree as et
import  xml.dom.minidom
import os
import cv2 as cv


COUNT_person=0
COUNT_bicycle=0
COUNT_car=0
COUNT_motorbike=0
# xml_path='E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'
# img_path='E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
xml_path='E:/xcc_download\VOC2007\Annotations'
img_path='E:\xcc_download\VOC2007\JPEGImages'

for file_name in os.listdir('E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'):
    print(file_name)
    per=et.parse(xml_path+'/'+file_name)
    # img=cv.imread(img_path+'/'+file_name[:-4]+'.jpg')
    cbndbox=per.findall('./object')
    for one_bndbox in cbndbox:
        for child in one_bndbox.getchildren():
            if str(child.text)=='person':
                person_box= one_bndbox.findall('bndbox')
                nbbox=[]
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                COUNT_person+=1
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]),(255, 0, 0), 2)
                # img = cv.putText(img, 'person', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
            if str(child.text) == 'bicycle':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                COUNT_bicycle+=1
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 255, 0), 2)
                # img = cv.putText(img, 'bicycle', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (0, 255, 0))
            if str(child.text) == 'car' or str(child.text) =='bus' or str(child.text) == 'train':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                COUNT_car+=1
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 0, 255), 2)
                # img = cv.putText(img, 'car', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (0, 0, 255))
            if str(child.text) == 'motorbike':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                COUNT_motorbike+=1
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 255, 255), 2)
                # img=cv.putText(img,'motorbike',(nbbox[0], nbbox[1]),3,cv.FONT_HERSHEY_PLAIN,(0, 255, 255))
    # cv.imshow('img',img)
    # cv.waitKey()
    # cv.destroyAllWindows()

print("人有{0}个，自行车有{1}辆，汽车有{2}辆，摩托车有{3}辆".format(COUNT_person,COUNT_bicycle,COUNT_car,COUNT_motorbike))

# cbndbox=per.findall('./object/bndbox')
# for one_bndbox in cbndbox:
#     for child in one_bndbox.getchildren():
#         print(child.text)



# parts=per.findall('./object/part/bndbox')
# print('第二种')
# # print('cname',cname)
# for one_part in parts:
#     for child in one_part.getchildren():
#         print(child.text)
