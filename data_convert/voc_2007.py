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
now_count=2971
for file_name in os.listdir(xml_path):
    print(xml_path+'/'+file_name)
    per=et.parse(xml_path+'/'+file_name)
    img=cv.imread(img_path+'/'+file_name[:-4]+'.jpg')
    cbndbox=per.findall('./object')
    input_txt=''
    for one_bndbox in cbndbox:
        for child in one_bndbox.getchildren():
            if str(child.text)=='person':
                person_box= one_bndbox.findall('bndbox')
                nbbox=[]
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))

                img_X = float((nbbox[0] + nbbox[2]) / img.shape[1]/2)
                img_Y = float((nbbox[1] + nbbox[3]) / img.shape[0]/2)
                img_W = float((nbbox[2] - nbbox[0]) / img.shape[1]/2)
                img_H = float((nbbox[3] - nbbox[1]) / img.shape[0]/2)
                input_txt += '0 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'

                # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
                #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (0, 255, 255), 2)
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]),(255, 0, 0), 2)
                # img = cv.putText(img, 'person', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
            if str(child.text) == 'bicycle':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))

                img_X = float((nbbox[0] + nbbox[2])/img.shape[1]/2)
                img_Y = float((nbbox[1] + nbbox[3])/img.shape[0]/2)
                img_W = float((nbbox[2] - nbbox[0])/img.shape[1]/2)
                img_H = float((nbbox[3] - nbbox[1])/img.shape[0]/2)
                input_txt += '2 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'

                # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
                #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (0, 255, 255), 2)
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 255, 0), 2)
                # img = cv.putText(img, 'bicycle', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (0, 255, 0))
            if str(child.text) == 'car' or str(child.text) =='bus' or str(child.text) == 'train':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                img_X = float((nbbox[0] + nbbox[2]) / img.shape[1] / 2)
                img_Y = float((nbbox[1] + nbbox[3]) / img.shape[0] / 2)
                img_W = float((nbbox[2] - nbbox[0]) / img.shape[1] / 2)
                img_H = float((nbbox[3] - nbbox[1]) / img.shape[0] / 2)
                input_txt += '1 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'

                # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
                #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (0, 255, 255), 2)
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 0, 255), 2)
                # img = cv.putText(img, 'car', (nbbox[0], nbbox[1]), 3, cv.FONT_HERSHEY_PLAIN, (0, 0, 255))
            if str(child.text) == 'motorbike':
                person_box = one_bndbox.findall('bndbox')
                nbbox = []
                for one_bndbox in person_box:
                    for child in one_bndbox.getchildren():
                        nbbox.append(int(float(child.text)))
                img_X = float((nbbox[0] + nbbox[2]) / img.shape[1] / 2)
                img_Y = float((nbbox[1] + nbbox[3]) / img.shape[0] / 2)
                img_W = float((nbbox[2] - nbbox[0]) / img.shape[1] / 2)
                img_H = float((nbbox[3] - nbbox[1]) / img.shape[0] / 2)
                input_txt += '3 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'
                # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
                #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (0, 255, 255), 2)
                # img = cv.rectangle(img, (nbbox[0], nbbox[1]), (nbbox[2], nbbox[3]), (0, 255, 255), 2)
                # img=cv.putText(img,'motorbike',(nbbox[0], nbbox[1]),3,cv.FONT_HERSHEY_PLAIN,(0, 255, 255))
    # print(len(input_txt))
    if len(input_txt)>3:
        now_name=str(now_count)
        if now_count<10:
            now_name ='0000'+ str(now_count)
        elif now_count>9 and  now_count<100:
            now_name ='000'+ str(now_count)
        elif now_count>99 and  now_count<1000:
            now_name ='00'+ str(now_count)
        elif now_count>999 and  now_count<10000:
            now_name ='0'+ str(now_count)




        now_count+=1
        out_file='E:/BOT_train/train/'+now_name
        out_txt = open(out_file+'.txt', 'w')
        out_txt.write(input_txt)
        out_txt.close()
        image = cv.resize(img, (416,416), interpolation=cv.INTER_CUBIC)
        cv.imwrite(out_file+'.jpg',image)
        # cv.imshow('img',img)
        # cv.waitKey()
        # cv.destroyAllWindows()

# print("人有{0}个，自行车有{1}辆，汽车有{2}辆，摩托车有{3}辆".format(COUNT_person,COUNT_bicycle,COUNT_car,COUNT_motorbike))

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
