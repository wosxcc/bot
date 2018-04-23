import cv2 as cv
import  numpy  as np
# 0.417831 0.365896 0.109276 0.181045
# 0.61405 0.477247 0.0923704 0.170564

# 0 0.533777 0.34452 0.0286396 0.0508707
# 0 0.199226 0.399607 0.0388255 0.0763999
# 0 0.724397 0.312903 0.0653683 0.134447
# 0 0.336084 0.230029 0.070514 0.141721
# 0 0.173281 0.316982 0.0320518 0.0586911
# 0 0.102856 0.336895 0.0213764 0.0403173


box= [0.4964601769911504 ,0.51015625, 0.47345132743362833, 0.47578125]

box= [float(i) for i in box]
img =cv.imread('E:\COCO/train/000000000073.jpg')
print(img.shape[0]*0.323323)

print( int(box[0]* img.shape[1]), int(box[1]* img.shape[0]))
print( int(box[2]* img.shape[1]), int(box[3]* img.shape[0]))

img=cv.ellipse(img, (int(box[0]* img.shape[1]), int(box[1]* img.shape[0])), (int(box[2]* img.shape[1]), int(box[3]* img.shape[0])), 0, 0, 360, (0, 255, 255), 2)
cv.imshow('img', img)
cv.waitKey()
cv.destroyAllWindows()