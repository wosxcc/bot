
import  cv2 as cv


img =cv.imread('E:/XTF/Ccao/ConsoleApplication1/imahe/1.jpg')
# img =cv.cvtColor(img,cv.COLOR_BGR2)


cv.resize(img,(160,160),cv.INTER_CUBIC)
print(img.shape,img)
cv.imshow('img',img)
cv.waitKey()