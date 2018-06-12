import cv2 as cv
import os


paths='E:/xbot/crete_data/train'
# paths='E:/xbot/crete_data/train'
paths ='E:/BOT_Car/train'
last_img=''
for file in os.listdir(paths):
    if file[-4:]=='.jpg' : ##
        img=cv.imread(paths + '/' +file)
        new_box = ''
        new_txt = open(paths + '/' + file[:-4]+'.txt')
        old_data = new_txt.read()
        for bbox in old_data.split('\n'):
            box = bbox.split(' ')
            if len(box) == 5:
                box = [float(i) for i in box]
                if box[0]==0:
                    img = cv.rectangle(img, (int((box[1] - box[3] / 2) * img.shape[1]), int((box[2] - box[4] / 2) * img.shape[0])), (
                                       int((box[1] + box[3] / 2) * img.shape[1]),
                                       int((box[2] + box[4] / 2) * img.shape[0])), (255, 0, 0), 2)
                else:
                    img = cv.rectangle(img, (
                    int((box[1] - box[3] / 2) * img.shape[1]), int((box[2] - box[4] / 2) * img.shape[0])), (
                                           int((box[1] + box[3] / 2) * img.shape[1]),
                                           int((box[2] + box[4] / 2) * img.shape[0])), (0, 0, 255), 2)

        cv.imshow(file, img)
        new_txt.close()
        k = cv.waitKey(0) & 0xFF
        if k == ord('d'):
            os.remove(paths + '/' + file)
            os.remove(paths + '/' + file[:-4]+'.txt')
            print('删除成功',paths + '/' + file)

        elif  k == ord('e'):
            os.remove(last_img)
            os.remove(last_img[:-4] + '.txt')
            print('删除前一张',last_img)
        else:
            last_img=paths + '/' + file
        cv.destroyAllWindows()
