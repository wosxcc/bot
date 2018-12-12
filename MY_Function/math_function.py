import math
import numpy as np
import cv2 as cv
pi = 3.1415926

# 向量夹角计算    （向量1，向量2）
def vector_angle(vector1,vector2):  # 返回夹角角度
    my_vector = (vector1[0]*vector2[0] +vector1[1]*vector2[1])/(math.sqrt(vector1[0]*vector1[0]+vector1[1]*vector1[1])* math.sqrt(vector2[0]*vector2[0]+vector2[1]*vector2[1]))
    my_angle =math.acos(my_vector) * 360 / 2 / pi
    return my_angle

# baxsa =[
# [4.0 ,0.315, 0.34739583333333335, 0.046, 0.12604166666666666],
# [5.0 ,0.84966666666666668, 0.5833333333333334, 0.126, 0.13541666666666666],
# [6.0 ,0.32666666666666668, 0.66, 0.1, 0.21]
# ]
# baxsb =[
# [4, 0.8496666666666667, 0.45, 0.046, 0.10833333333333334],
# [5, 0.7896666666666666, 0.6703125, 0.13266666666666665, 0.14270833333333333],
# [6, 0.3266666666666667, 0.66, 0.1, 0.24]
# ]
def re_iou(boxA,boxB):
    iou = -1
    Ax1= boxA[1]-(boxA[3]/2)
    Ay1= boxA[2]-(boxA[4]/2)
    Ax2 = boxA[1] + (boxA[3] / 2)
    Ay2 = boxA[2] + (boxA[4] / 2)

    Bx1 = boxB[1] - (boxB[3] / 2)
    By1 = boxB[2] - (boxB[4] / 2)
    Bx2 = boxB[1] + (boxB[3] / 2)
    By2 = boxB[2] + (boxB[4] / 2)

    xover = boxA[3]+boxB[3]- max(abs(Ax1-Bx2),abs(Bx1-Ax2))
    yover = boxA[4] + boxB[4] - max(abs(Ay1 - By2), abs(By1 - Ay2))
    if (xover<=0 or yover<=0):
        return iou
    else:
        iou= (xover*yover)/(boxA[3]*boxA[4]+boxB[3]*boxB[4]-xover*yover)
        return iou


def re_max_iou(abox,boxs):
    max_v = -1
    max_b = -1
    for i in range(len(boxs)):
        nvalue = re_iou(abox,boxs[i])
        if nvalue>max_v:
            max_v=nvalue
            max_b=i

    return max_v,max_b
def re_score(Forecast_boxs,True_Bboxs):
    recall_rate = 0.0     # 查全率
    Precision_rate = 0.0  # 查准率
    # error_rate = 0.0      # 误查率
    iou_score = 0.0       # iou得分
    # leak_rate = 0.0
    for tbox in True_Bboxs:
        max_v, max_b =re_max_iou(tbox, Forecast_boxs)
        if max_v!= -1 and max_v>0.3:
            recall_rate+=1
            iou_score += max_v
            if tbox[0] == True_Bboxs[max_b][0]:
                Precision_rate+=1
            # else:
            #     error_rate+=1
        # else:
        #     error_rate += 1
    re_iou = iou_score/recall_rate
    re_recall = recall_rate / len(True_Bboxs)
    re_prec = Precision_rate/ len(True_Bboxs)
    re_error = (len(Forecast_boxs)-Precision_rate)/len(True_Bboxs)
    return re_prec, re_recall, re_error, re_iou   # 查准率 查全率 误查率,iou得分

# img = np.zeros([800,800,3],np.uint8)
# colors=(255,0,0)
# re_prec ,re_recall,re_error,re_ious =re_score(baxsa,baxsb)
# print(re_prec ,re_recall,re_error,re_ious)
#
# for i in  range(len(baxsa)):
#     box = baxsa[i]
#     boxb = baxsb[i]
#     if box[0]==4:
#         colors = (255,255,0)
#     if box[0]==5:
#         colors = (255,0,255)
#     if box[0]==6:
#         colors = (0,255,255)
#     print(re_iou(box, boxb))
#     cv.rectangle(img,(int((box[1]-(box[3]/2))*img.shape[1]),int((box[2]-(box[4]/2))*img.shape[1])),(int((box[1]+(box[3]/2))*img.shape[1]),int((box[2]+(box[4]/2))*img.shape[1])),colors,-1)
#     cv.rectangle(img, (int((boxb[1] - (boxb[3] / 2)) * img.shape[1]), int((boxb[2] - (boxb[4] / 2)) * img.shape[1])),
#                  (int((boxb[1] + (boxb[3] / 2)) * img.shape[1]), int((boxb[2] + (boxb[4] / 2)) * img.shape[1])), colors, -1)
#     cv.putText(img,str(re_iou(box, boxb)),(int((box[1]-(box[3]/2))*img.shape[1]),int((box[2]-(box[4]/2))*img.shape[1])),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# # for box in  baxsb:
# #     if box[0]==4:
# #         colors = (255,255,0)
# #     if box[0]==5:
# #         colors = (255,0,255)
# #     if box[0]==6:
# #         colors = (0,255,255)
# #     cv.rectangle(img,(int((box[1]-(box[3]/2))*img.shape[1]),int((box[2]-(box[4]/2))*img.shape[1])),(int((box[1]+(box[3]/2))*img.shape[1]),int((box[2]+(box[4]/2))*img.shape[1])),colors,-1)
#
# cv.imshow('img',img)
# cv.waitKey()