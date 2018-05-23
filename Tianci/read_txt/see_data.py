# coding:UTF-8
import os
import numpy as np
import  cv2 as cv


txt_path='E:/xcc_download/ICPR_text_train_part2_20180313/txt_9000'
img_path='E:/xcc_download/ICPR_text_train_part2_20180313/image_9000'
count_img=0
out_path='E:/BOT_Txt/train/'
for file in os.listdir(txt_path):
    print(txt_path+'/'+file)
    read_txt=open(txt_path+'/'+file,encoding = 'utf-8')
    txt_into=read_txt.read()
    txt_line=txt_into.split('\n')
    img=cv.imread(img_path+'/'+file[:-4]+'.jpg')
    out_data = ''
    try:
        for box in txt_line:
            sbox=box.split(',')

            if len(sbox)>5:
                xbox= [int(float(i)) for i in sbox[:-1]]

                # spots = np.array([[[xbox[0], xbox[1]], [xbox[2], xbox[3]], [xbox[4], xbox[5]], [xbox[6], xbox[7]]]],
                #                  dtype=np.int32)
                # cv.polylines(img, spots, 2, (0,255,0))
                # cv.polylines(img,[np.int(spots)],True,255,10,cv.LINE_AA)



                if xbox[1]>xbox[3]:
                    # cv.circle(img, (xbox[0], xbox[1]), 2, (0, 0, 255), -1)
                    # cv.circle(img, (xbox[6], xbox[7]), 2, (0, 0, 255), -1)

                    if xbox[0]>xbox[6]:
                        left_x = xbox[6]
                        left_y = xbox[7]
                        right_y = xbox[1]
                        right_x = xbox[0]
                    else:
                        left_x = xbox[0]
                        left_y = xbox[1]
                        right_y = xbox[7]
                        right_x = xbox[6]
                else:
                    # cv.circle(img, (xbox[2], xbox[3]), 2, (0, 0, 255), -1)
                    # cv.circle(img, (xbox[4], xbox[5]), 2, (0, 0, 255), -1)

                    if xbox[2] > xbox[4]:
                        left_x = xbox[4]
                        left_y = xbox[5]
                        right_y = xbox[3]
                        right_x = xbox[2]
                    else:
                        left_x = xbox[2]
                        left_y = xbox[3]
                        right_y = xbox[5]
                        right_x = xbox[4]

                if abs(xbox[6]-xbox[0])>=abs(xbox[4]-xbox[2]):
                    x1 = xbox[6]
                    x2 = xbox[0]
                    y1 = xbox[7]
                    y2 = xbox[1]
                else:
                    x1 = xbox[4]
                    x2 = xbox[2]
                    y1 = xbox[5]
                    y2 = xbox[3]
                if abs(xbox[3]-xbox[1])>=abs(xbox[7]-xbox[5]):
                    x3 = xbox[2]
                    x4 = xbox[0]
                    y3 = xbox[3]
                    y4 = xbox[1]
                else:
                    x3 = xbox[6]
                    x4 = xbox[4]
                    y3 = xbox[7]
                    y4 = xbox[5]

                if x1 == x3 and y1 == y3:
                    mx = float(abs(x2 + x4)) / 2
                    my = float(abs(y2 + y4)) / 2
                elif x1 == x4 and y1 == y4:
                    mx = float(abs(x2 + x3)) / 2
                    my = float(abs(y2 + y3)) / 2
                elif x2 == x4 and y2 == y4:
                    mx = float(abs(x1 + x3)) / 2
                    my = float(abs(y1 + y3)) / 2
                elif x2 == x3 and y2 == y3:
                    mx = float(abs(x1 + x4)) / 2
                    my = float(abs(y1 + y4)) / 2
                #
                # cv.circle(img, (int(x1), int(y1)), 4, (255, 0, 0), -1)
                # cv.circle(img, (int(x2), int(y2)), 4, (255, 0, 0), -1)
                # cv.circle(img, (int(x3), int(y3)), 4, (255, 0, 0), -1)
                # cv.circle(img, (int(x4), int(y4)), 4, (255, 0, 0), -1)


                out_data += '0 ' + str(mx / img.shape[1]) + ' ' + str(my / img.shape[0]) + ' ' + str(
                    left_x / img.shape[1]) + ' ' + str(left_y/ img.shape[0]) + '\n'
                out_data += '1 ' + str(mx / img.shape[1]) + ' ' + str(my / img.shape[0]) + ' ' + str(
                    right_x / img.shape[1]) + ' ' + str(right_y / img.shape[0]) + '\n'

                # cv.circle(img, (int(mx), int(my)), 4, (0, 255, 255), -1)
                #
                # cv.circle(img, (int(left_x), int(left_y)), 4, (255, 255, 0), -1)
                # cv.circle(img, (int(right_x), int(right_y)), 4, (255, 0, 255), -1)


                # if xbox[0]>xbox[2]:
                #     cv.circle(img, (xbox[0], xbox[1]), 2, (0, 0, 255), -1)
                #     cv.circle(img, (xbox[2], xbox[3]), 2, (0, 0, 255), -1)
                # else:
                # cv.circle(img, (xbox[4], xbox[5]), 2, (0, 0, 255), -1)
                # cv.circle(img, (xbox[6], xbox[7]), 2, (0, 0, 255), -1)
        new_name = str(count_img)
        for i in range(5 - len(new_name)):
            new_name = '0' + new_name
        print(new_name)
        out_txt = open(out_path + new_name + '.txt', 'w')
        out_txt.write(out_data)
        out_txt.close()
        count_img += 1
        cv.imwrite(out_path + new_name + '.jpg', img)
        # cv.imshow('img',img)
        # cv.waitKey()
    except:
        print('error')