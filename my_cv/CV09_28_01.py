import cv2 as cv


cap = cv.VideoCapture('rtsp://admin:admin123@192.168.0.3/doc/page/preview.asp')

ret,frame = cap.read()
while ret:
    ret,frame = cap.read()
    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
cap.release()

