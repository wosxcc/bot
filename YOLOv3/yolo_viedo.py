from YOLOv3.yolo import YOLO
from YOLOv3.yolo import detect_video
detect_video(YOLO())


# import cv2 as cv
# from PIL import Image, ImageFont, ImageDraw
# import numpy as np
# from timeit import default_timer as timer


#
# yolo=YOLO()
# video_path=""
# output_path=""
#
# vid = cv.VideoCapture(0)
# if not vid.isOpened():
#     raise IOError("Couldn't open webcam or video")
# video_FourCC = int(vid.get(cv.CAP_PROP_FOURCC))
# video_fps = vid.get(cv.CAP_PROP_FPS)
# video_size = (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
#               int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
# isOutput = True if output_path != "" else False
# if isOutput:
#     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
#     out = cv.VideoWriter(output_path, video_FourCC, video_fps, video_size)
# accum_time = 0
# curr_fps = 0
# fps = "FPS: ??"
# prev_time = timer()
#
# return_value, frame = vid.read()
# # imgsize=(frame.shape[1],frame.shape[0])
# while True:
#     return_value, frame = vid.read()
#     image = Image.fromarray(frame)
#     image = yolo.detect_image(image)
#     result = np.asarray(image)
#     curr_time = timer()
#     exec_time = curr_time - prev_time
#     prev_time = curr_time
#     accum_time = accum_time + exec_time
#     curr_fps = curr_fps + 1
#     if accum_time > 1:
#         accum_time = accum_time - 1
#         fps = "FPS: " + str(curr_fps)
#         curr_fps = 0
#     cv.putText(result, text=fps, org=(3, 15), fontFace=cv.FONT_HERSHEY_SIMPLEX,
#                 fontScale=0.50, color=(255, 0, 0), thickness=2)
#     cv.namedWindow("result", cv.WINDOW_NORMAL)
#     # cv.resize(result,imgsize,interpolation=cv.INTER_CUBIC)
#     cv.imshow("result", result)
#     if isOutput:
#         out.write(result)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
# yolo.close_session()