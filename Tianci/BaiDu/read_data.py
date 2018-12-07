import os
import cv2 as cv
# file_path = 'E:/xcc_download/ColorImage_road04/ColorImage'#
#
# file_path_bin = 'E:/xcc_download/Labels_road04/Label'
file_path = 'E:/xcc_download/Road02/ColorImage_road02/ColorImage'# "E:/xcc_download/baidu/ColorImage_road03/ColorImage"
file_path_bin = 'E:/xcc_download/Road02/Labels_road02/Label'#"E:/xcc_download/baidu/Labels_road03/Label"
out_txt = ""
for filej in os.listdir(file_path):
    for afile in os.listdir(file_path+'/'+filej):
        for aimg in os.listdir(file_path+'/'+filej+'/'+afile):
            # img = cv.imread(file_path+'/'+filej+'/'+afile+'/'+aimg)
            # print(file_path+'/'+filej+'/'+afile+'/'+aimg)
            # print(file_path_bin + '/' + filej + '/' + afile + '/' + aimg[:-4]+'_bin'+'.png')
            #
            # img_bin = cv.imread(file_path_bin + '/' + filej + '/' + afile + '/' + aimg[:-4]+'_bin'+'.png')
            #
            # print(img_bin.shape)
            # cv.imshow(file_path_bin + '/' + filej + '/' + afile + '/' + aimg[:-4]+'_bin'+'.png', img_bin)
            # cv.imshow(file_path+'/'+filej+'/'+afile+'/'+aimg,img)
            # cv.waitKey()
            # cv.destroyAllWindows()

            out_txt += file_path + '/' + filej + '/' + afile + '/' + aimg + '---' + file_path_bin + '/' + filej + '/' + afile + '/' + aimg[
                                                                                                                                    :-4] + '_bin' + '.png' + '\n'
file_txt = open('train3.txt', 'w')
file_txt.write(out_txt)
file_txt.close()

# E:/xcc_download/baidu/Labels_road03/Label/Record001/Camera 5/171206_025742296_Camera_5_bin.jpg
# E:\xcc_download\baidu\Labels_road03\Label\Record001\Camera 5