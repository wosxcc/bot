import json
import cv2 as cv
import os
import numpy as np


# with open('voc_2017_data.txt','w') as file_txt:
with open("E:/xcc_download/annotations_trainval2017 (2)/annotations/instances_train2017.json",'r') as load_f:
    load_dict = json.load(load_f)

    # print('info')
    # print()
    # for date2 in  load_dict['info']:
    #     print(date2)

    # print('images')
    # print()
    # for date2 in load_dict['images']:
    #     print(date2)



    # print('licenses')
    # print()
    # for date2 in load_dict['licenses']:
    #     print(date2)


    # print('categories')
    # print()
    # for date2 in load_dict['categories']:
    #     print(date2)
    # print('annotations')
    # print()

    # {'supercategory '：'人'，'身份证'：1，名称：“人”}
    # {'supercategory '：'汽车'，'身份证'：2，名称：“自行车”}
    # {'supercategory '：'汽车'，'身份证'：3，名称：“汽车”}
    # {'supercategory '：'汽车'，'身份证'：4，名称：“摩托车”}
    # {'supercategory '：'汽车'，'身份证'：6，名称：“公交”}
    # {'supercategory '：'汽车'，'身份证'：7，名称：“火车”}
    # {'supercategory '：'汽车'，'身份证'：8，名称：“卡车”}
    # {'supercategory '：'户外'，'身份证'：10，名称：“红绿灯”}

    now_count=0


    for date2 in load_dict['annotations']:
        out_file_name = ''
        out_txt_into = ''


        ###保存人数据
        if date2['category_id']==1:
            out_file_name =str(date2['image_id'])
            # print(len(str(date2['image_id'])))
            if len(str(date2['image_id']))<12:
                for i in range(12-len(str(date2['image_id']))):
                    out_file_name='0'+out_file_name
            # print('行人', out_file_name)
            img=cv.imread('E:/xcc_download/train2017/'+out_file_name+'.jpg')
            box=date2['bbox']
            box = [int(i) for i in box]
            img_X = float((box[0] + box[2] / 2) / img.shape[1] )
            img_Y = float((box[1] + box[3] / 2) / img.shape[0])
            img_W = float((box[2] / 2) / img.shape[1])
            img_H = float((box[3] / 2) / img.shape[0])

            out_txt_into+= '0 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'

            # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])), (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (255, 0, 5), 2)
            # img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            # segmentation=date2['segmentation']
            # # segmentation = [int(i) for i in segmentation]
            # for xian in segmentation:
            #     xian = [int(i) for i in xian]
            #     xian = np.array(xian).reshape(int(len(xian) / 2), 1, 2)
            #     pts = np.array(xian, np.int32)
            #     img = cv.polylines(img, [pts], True, (0, 255, 255), 2)
            #
            # print('是人', date2['image_id'], date2['bbox'])
            # # print('date2',date2)
            # cv.imshow('img',img)
            # cv.waitKey()
        ###自行车
        if date2['category_id'] == 2 :
            out_file_name = str(date2['image_id'])
            # print(len(str(date2['image_id'])))
            if len(str(date2['image_id'])) < 12:
                for i in range(12 - len(str(date2['image_id']))):
                    out_file_name = '0' + out_file_name
            # print('自行车', out_file_name)
            img = cv.imread('E:/xcc_download/train2017/' + out_file_name + '.jpg')
            box = date2['bbox']
            box = [int(i) for i in box]
            img_X = float((box[0] + box[2] / 2) / img.shape[1])
            img_Y = float((box[1] + box[3] / 2) / img.shape[0])
            img_W = float((box[2] / 2) / img.shape[1])
            img_H = float((box[3] / 2) / img.shape[0])
            out_txt_into += '2 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'




            # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])), (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (255, 0, 5), 2)
            # img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            # segmentation=date2['segmentation']
            # # segmentation = [int(i) for i in segmentation]
            # for xian in segmentation:
            #     xian = [int(i) for i in xian]
            #     xian = np.array(xian).reshape(int(len(xian) / 2), 1, 2)
            #     pts = np.array(xian, np.int32)
            #     img = cv.polylines(img, [pts], True, (0, 255, 255), 2)
            # print('自行车', date2['image_id'], date2['bbox'])
            # cv.imshow('img',img)
            # cv.waitKey()

        ###摩托车
        if date2['category_id'] == 4:
            out_file_name = str(date2['image_id'])
            # print(len(str(date2['image_id'])))
            if len(str(date2['image_id'])) < 12:
                for i in range(12 - len(str(date2['image_id']))):
                    out_file_name = '0' + out_file_name
            # print('摩托车', out_file_name)
            img = cv.imread('E:/xcc_download/train2017/' + out_file_name + '.jpg')
            box = date2['bbox']
            box = [int(i) for i in box]
            img_X = float((box[0] + box[2] / 2) / img.shape[1])
            img_Y = float((box[1] + box[3] / 2) / img.shape[0])
            img_W = float((box[2] / 2) / img.shape[1])
            img_H = float((box[3] / 2) / img.shape[0])
            out_txt_into += '3 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'




            # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
            #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (255, 0, 5), 2)
            # img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            # segmentation = date2['segmentation']
            # # segmentation = [int(i) for i in segmentation]
            # for xian in segmentation:
            #     xian = [int(i) for i in xian]
            #     xian = np.array(xian).reshape(int(len(xian) / 2), 1, 2)
            #     pts = np.array(xian, np.int32)
            #     img = cv.polylines(img, [pts], True, (0, 255, 255), 2)
            # print('摩托车', date2['image_id'], date2['bbox'])
            # cv.imshow('img', img)
            # cv.waitKey()

        ##汽车车
        if date2['category_id'] == 3 or date2['category_id'] == 6 or date2['category_id'] == 8 :
            out_file_name = str(date2['image_id'])
            # print(len(str(date2['image_id'])))
            if len(str(date2['image_id'])) < 12:
                for i in range(12 - len(str(date2['image_id']))):
                    out_file_name = '0' + out_file_name
            # print('汽车', out_file_name)
            img = cv.imread('E:/xcc_download/train2017/' + out_file_name + '.jpg')
            box = date2['bbox']
            box = [int(i) for i in box]
            img_X = float((box[0] + box[2] / 2) / img.shape[1])
            img_Y = float((box[1] + box[3] / 2) / img.shape[0])
            img_W = float((box[2] / 2) / img.shape[1])
            img_H = float((box[3] / 2) / img.shape[0])
            out_txt_into += '1 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'


            #
            # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
            #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (255, 0, 5), 2)
            # img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            # segmentation = date2['segmentation']
            # # segmentation = [int(i) for i in segmentation]
            # for xian in segmentation:
            #     xian = [int(i) for i in xian]
            #     xian = np.array(xian).reshape(int(len(xian) / 2), 1, 2)
            #     pts = np.array(xian, np.int32)
            #     img = cv.polylines(img, [pts], True, (0, 255, 255), 2)
            # print('汽车', date2['image_id'], date2['bbox'])
            # cv.imshow('img', img)
            # cv.waitKey()


        ##红绿灯
        if date2['category_id'] == 10:
            out_file_name = str(date2['image_id'])
            # print(len(str(date2['image_id'])))
            if len(str(date2['image_id'])) < 12:
                for i in range(12 - len(str(date2['image_id']))):
                    out_file_name = '0' + out_file_name

            img = cv.imread('E:/xcc_download/train2017/' + out_file_name + '.jpg')
            # print('红绿灯',out_file_name)
            box = date2['bbox']
            box = [int(i) for i in box]
            img_X = float((box[0] + box[2] / 2) / img.shape[1])
            img_Y = float((box[1] + box[3] / 2) / img.shape[0])
            img_W = float((box[2] / 2) / img.shape[1])
            img_H = float((box[3] / 2) / img.shape[0])
            out_txt_into += '4 ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H) + '\n'



            # img = cv.ellipse(img, (int(img_X * img.shape[1]), int(img_Y * img.shape[0])),
            #                  (int(img_W * img.shape[1]), int(img_H * img.shape[0])), 0, 0, 360, (255, 0, 5), 2)
            # img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            # segmentation = date2['segmentation']
            # # segmentation = [int(i) for i in segmentation]
            # for xian in segmentation:
            #     xian = [int(i) for i in xian]
            #     xian = np.array(xian).reshape(int(len(xian) / 2), 1, 2)
            #     pts = np.array(xian, np.int32)
            #     img = cv.polylines(img, [pts], True, (0, 255, 255), 2)
            # print('红绿灯', date2['image_id'], date2['bbox'])
            # cv.imshow('img', img)
            # cv.waitKey()

        if len(out_txt_into)>3:


            file_path = 'E:/COCO/train/'+out_file_name
            image = cv.resize(img, (416, 416), interpolation=cv.INTER_CUBIC)
            cv.imwrite(file_path + '.jpg', image)




            new_txt =out_txt_into
            txt_name =file_path+'.txt'
            # print(file_path + '.jpg')
            # print(txt_name)
            if os.path.exists(txt_name):
                print('第{0}张图片：{1}\t\t文本已存在'.format(now_count, file_path))
                exist_txt = open(txt_name)
                old_data = exist_txt.read()
                new_txt = old_data + new_txt
                exist_txt.close()
                write_txt = open(txt_name, 'w')
                write_txt.write(new_txt)
                write_txt.close()
            else:
                print('第{0}张图片：{1}\t\t文本已创建'.format(now_count, file_path))
                write_txt = open(txt_name, 'w')
                write_txt.write(new_txt)
                write_txt.close()


            now_count += 1

            # cv.imshow('img', img)
            # cv.imshow('image', image)
            # cv.waitKey()

        # now_name = str(now_count)
        # if len(now_name) < 6:
        #     for i in range(6 - len(now_name)):
        #         now_name = '0' + now_name
        # now_count += 1
    # print('共有{0}个人，自行车{1}辆，摩托车{2}辆，汽车{3}辆，红绿灯{4}个'.format(COUNT1,COUNT2,COUNT3,COUNT4,COUNT5))






    # print(load_dict)
    # result=str(load_dict).split('segmentation')
    #
    # print('',len(result))
    # for i in range(len(result)):
    #     print(i)
    #     print(result[i])
    # # print(load_dict)





