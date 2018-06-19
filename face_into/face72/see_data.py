import os
import cv2 as cv



path_files = 'E:/dectect/dectect/face68'

for file in os.listdir(path_files):
    if (file[-4:]=='.txt'):
        print(file)
        img = cv.imread(path_files+'/' + file[:-4]+'.jpg')
        txt_open = open(path_files+'/' +  file)
        txt_read = txt_open.read()
        txt_lines =txt_read.split(' ')
        txt_float = [float(i) for  i in txt_lines]
        biaoq= 'xiao'
        if txt_float[0]==0:
            biaoq='buxiao'
        elif txt_float[0]==2:
            biaoq='daxiao'
        biaoq +=  str(txt_float[1])
        img = cv.putText(img, biaoq, (0, 25), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
        for x in range(int(len(txt_float)/2)-1):
            img=cv.circle(img,(int(txt_float[2 + x * 2]*img.shape[1]),int(txt_float[2 + x * 2 + 1]*img.shape[0])),1,(0,255,0),-1)
        cv.imshow('img', img)
        txt_open.close()
        k = cv.waitKey(0) & 0xFF
        if k == ord('d'):
            os.remove(path_files + '/' + file)
            os.remove(path_files + '/' + file[:-4] + '.jpg')
            print('删除成功', path_files + '/' + file)

        elif k == ord('e'):
            os.remove(last_img)
            os.remove(last_img[:-4] + '.jpg')
            print('删除前一张', last_img)
        else:
            last_img = path_files + '/' + file



