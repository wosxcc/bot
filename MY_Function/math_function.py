import math
import numpy

pi = 3.1415926

# 向量夹角计算    （向量1，向量2）
def vector_angle(vector1,vector2):  # 返回夹角角度
    my_vector = (vector1[0]*vector2[0] +vector1[1]*vector2[1])/(math.sqrt(vector1[0]*vector1[0]+vector1[1]*vector1[1])* math.sqrt(vector2[0]*vector2[0]+vector2[1]*vector2[1]))
    my_angle =math.acos(my_vector) * 360 / 2 / pi
    return my_angle


