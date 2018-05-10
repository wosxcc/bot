import json
import cv2 as cv
import os
import numpy as np


# with open('voc_2017_data.txt','w') as file_txt:
with open("E:/Caltech/annotations/orter/annotations.json",'r') as load_f:
    load_dict = json.load(load_f)
    now_count=0
    # for data4 in load_dict['set10']['V008']['frames']:
    #     xxname=str(int(data4)+1)+'.jpg'
    #     img=cv.imread('E:/Caltech/set10/V008/'+xxname)
    #     for data5 in  load_dict['set10']['V008']['frames'][data4]:
    #         print (data5['pos'])
    #         bbox=[int(i) for i in data5['pos']]
    #         img=cv.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),2)
    #     cv.imshow('img',img)
    #     cv.waitKey()


    # for hhhh in  load_dict['set03']['V008']['frames']:
    #     # for sss in load_dict['set03']['V008'][hhhh]:
    #     #     print(load_dict['set03']['V008'][hhhh][sss])
    #     print(hhhh)
        # print(load_dict['set03']['V008'][hhhh])
        # print(load_dict['set03']['V008']['nFrame'])
    for data2 in load_dict:
        if data2>'set05':
            for data3 in load_dict[data2]:
                for data4 in load_dict[data2][data3]['frames']:
                    xxname = str(int(data4) + 1) + '.jpg'
                    img = cv.imread('E:/Caltech/'+data2+'/'+data3+'/' + xxname)
                    for data5 in load_dict[data2][data3]['frames'][data4]:
                        bbox = [int(i) for i in data5['pos']]
                    img = cv.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
                    cv.imshow('img',img)
                    cv.waitKey()
                print(data3)
                print(load_dict[data2][data3])



    # for date2 in load_dict['annotations']:
    #     out_file_name = ''
    #     out_txt_into = ''
    #     ###保存人数据
    #     if date2['category_id']==1:
    #         out_file_name =str(date2['image_id'])
    #         # print(len(str(date2['image_id'])))
    #         if len(str(date2['image_id']))<12:
    #             for i in range(12-len(str(date2['image_id']))):
    #                 out_file_name='0'+out_file_name
    #         # print('行人', out_file_name)
    #         img=cv.imread('E:/xcc_download/train2017/'+out_file_name+'.jpg')
    #         box=date2['bbox']
    #         box = [int(i) for i in box]
    #         img_X = float((box[0] + box[2] / 2) / img.shape[1] )
    #         img_Y = float((box[1] + box[3] / 2) / img.shape[0])
    #         img_W = float((box[2] / 2) / img.shape[1])
    #         img_H = float((box[3] / 2) / img.shape[0])
    #
    #         out_txt_into+= '0 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'
    #
    #     if len(out_txt_into)>3:
    #         file_path = 'E:/BOT_COCO/train/'+out_file_name
    #         image = cv.resize(img, (416, 416), interpolation=cv.INTER_CUBIC)
    #         cv.imwrite(file_path + '.jpg', image)
    #
    #         new_txt =out_txt_into
    #         txt_name =file_path+'.txt'
    #         if os.path.exists(txt_name):
    #             print('第{0}张图片：{1}\t\t文本已存在'.format(now_count, file_path))
    #             exist_txt = open(txt_name)
    #             old_data = exist_txt.read()
    #             new_txt = old_data + new_txt
    #             exist_txt.close()
    #             write_txt = open(txt_name, 'w')
    #             write_txt.write(new_txt)
    #             write_txt.close()
    #         else:
    #             print('第{0}张图片：{1}\t\t文本已创建'.format(now_count, file_path))
    #             write_txt = open(txt_name, 'w')
    #             write_txt.write(new_txt)
    #             write_txt.close()
    #         now_count += 1

