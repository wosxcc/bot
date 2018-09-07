import  os
import  sys


sys.path.append('E:/BOT_Person/bot_face')


os.system('E:/BOT_Person/bot_face/darknet.exe detector demo E:/BOT_Person/bot_face/voc.data E:/BOT_Person/bot_face/zxcc_yolo_test.cfg E:/BOT_Person/zxcc_yolofjian_16000.weights -i 0 -thresh 0.4 E:/BOT_Person/bot_face/222.mp4 pause')