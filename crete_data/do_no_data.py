import os
import cv2 as cv



file_path='E:/2018-07-27/noperson'

atxt= open('E:/BOT_Person/notrain.txt','w')
xxxx= 99500
for file in os.listdir(file_path):
    if file[-6:] =='a.jpeg':
        img = cv.imread(file_path+'/'+file[:-6]+'o.jpeg')

        # cv.imshow(file,img)
        imgs = cv.resize(img,(480,480),interpolation=cv.INTER_CUBIC)
        image_name = str(xxxx)+'.jpg'
        cv.imwrite('E:/BOT_Person/noperson/'+image_name,imgs)
        mtxt = open('E:/BOT_Person/noperson/'+str(xxxx)+'.txt', 'w')
        mtxt.close()
        atxt.write('E:/BOT_Person/trainb480/'+image_name+'\n')
        xxxx+=1
        # cv.waitKey()
        # cv.destroyAllWindows()
atxt.close()
