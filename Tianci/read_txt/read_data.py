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
                xbox= [float(i) for i in sbox[:-1]]


                max_x = max(xbox[0],xbox[2],xbox[4],xbox[6])
                max_x = max(xbox[0], xbox[2], xbox[4], xbox[6])


                out_data +='0 '+str(xbox[0]/img.shape[1])+' '+str(xbox[1]/img.shape[0])+' '+str(xbox[4]/img.shape[1])+' '+str(xbox[5]/img.shape[0])+'\n'
                out_data += '1 ' + str(xbox[2] / img.shape[1]) + ' ' + str(xbox[3] / img.shape[0]) + ' ' + str(
                    xbox[6] / img.shape[1]) + ' ' + str(xbox[7] / img.shape[0]) + '\n'


                # spots =np.array([[[xbox[0], xbox[1]],[xbox[2], xbox[3]],[xbox[4], xbox[5]],[xbox[6], xbox[7]]]], dtype=np.int32)
                #
                #
                # # spots=np.float32([[xbox[0], xbox[1]],[xbox[2], xbox[3]],[xbox[4], xbox[5]],[xbox[6], xbox[7]]]).reshape(-1,1,2)
                # cv.polylines(img, spots, 2, (0,255,0))
                # # cv.polylines(img,[np.int(spots)],True,255,10,cv.LINE_AA)
                # cv.circle(img, (xbox[0], xbox[1]), 2, (0, 0, 255), -1)
                # cv.circle(img, (xbox[2], xbox[3]), 2, (0, 0, 255), -1)
                # cv.circle(img, (xbox[4], xbox[5]), 2, (0, 0, 255), -1)
                # cv.circle(img, (xbox[6], xbox[7]), 2, (0, 0, 255), -1)

        new_name=str(count_img)
        for i in range(5-len(new_name)):
            new_name='0'+new_name
        print(new_name)
        out_txt = open(out_path+new_name+'.txt', 'w')
        out_txt.write(out_data)
        out_txt.close()
        count_img += 1

        cv.imwrite(out_path+new_name+'.jpg',img)
    except:
        print('error')

    # cv.imshow('img',img)
    # cv.waitKey()

