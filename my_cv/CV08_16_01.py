import cv2

camera = cv2.VideoCapture('E:/BOT_Car/bot_car/test2.mp4') # 参数0表示第一个摄像头
mog = cv2.createBackgroundSubtractorMOG2()

while (1):
    grabbed, frame_lwpCV = camera.read()
    fgmask = mog.apply(frame_lwpCV)
    cv2.imshow('frame', fgmask)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()