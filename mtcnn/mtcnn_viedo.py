from mtcnn.mtcnn import MTCNN
import cv2 as cv
import  datetime
# cap = cv.VideoCapture(0)####"movie.mpg"
# while True:
#     ret, frame = cap.read()

cap =cv.VideoCapture(0)
detector =MTCNN()
while(1):
    ret, frame = cap.read()
    # frame=cv.imread('../image/PP01.jpg')

    # [{'box': [163, 119, 53, 67], 'confidence': 0.99999451637268066, 'keypoints': {'left_eye': (180, 146), 'right_eye': (205, 146), 'nose': (193, 159), 'mouth_left': (181, 170), 'mouth_right': (203, 171)}}, {'box': [600, 116, 58, 71], 'confidence': 0.99974900484085083, 'keypoints': {'left_eye': (614, 148), 'right_eye': (638, 139), 'nose': (628, 155), 'mouth_left': (622, 171), 'mouth_right': (646, 163)}}, {'box': [894, 119, 47, 68], 'confidence': 0.99947327375411987, 'keypoints': {'left_eye': (905, 145), 'right_eye': (928, 149), 'nose': (911, 159), 'mouth_left': (902, 167), 'mouth_right': (925, 171)}}, {'box': [1155, 108, 48, 65], 'confidence': 0.99944323301315308, 'keypoints': {'left_eye': (1164, 133), 'right_eye': (1186, 133), 'nose': (1169, 145), 'mouth_left': (1163, 156), 'mouth_right': (1185, 156)}}, {'box': [363, 111, 53, 72], 'confidence': 0.99910056591033936, 'keypoints': {'left_eye': (383, 141), 'right_eye': (407, 138), 'nose': (400, 152), 'mouth_left': (385, 165), 'mouth_right': (410, 162)}}]
    # cv.imshow('img', img)
    # cv.waitKey()
    print(datetime.datetime.now())
    face_into=detector.detect_faces(frame)
    print(datetime.datetime.now())
    for i in range(len(face_into)):
        # face_into[i]['box']
        # face_into[i]['confidence']
        frame=cv.rectangle(frame,(face_into[i]['box'][0],face_into[i]['box'][1]),(face_into[i]['box'][0]+face_into[i]['box'][2],face_into[i]['box'][1]+face_into[i]['box'][3]),(0,0,255),2)
        frame = cv.circle(frame, face_into[i]['keypoints']['left_eye'], 2, (255, 255, 0), -1)
        frame = cv.circle(frame, face_into[i]['keypoints']['right_eye'], 2, (255, 255, 0), -1)
        frame = cv.circle(frame, face_into[i]['keypoints']['nose'], 2, (255, 255, 0), -1)
        frame = cv.circle(frame, face_into[i]['keypoints']['mouth_left'], 2, (255, 255, 0), -1)
        frame = cv.circle(frame, face_into[i]['keypoints']['mouth_right'], 2, (255, 255, 0), -1)

        # face_into[i]['keypoints']['right_eye']
        # face_into[i]['keypoints']['nose']
        # face_into[i]['keypoints']['mouth_left']
        # face_into[i]['keypoints']['mouth_right']
        print(face_into[i])

    cv.imshow('img',frame)
    cv.waitKey(30)



