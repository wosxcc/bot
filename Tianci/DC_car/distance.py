from math import *
import tensorflow as tf
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001
# GaussianNoise

print(getDistance(31.777582000000002,117.19014299999999,31.784909999999996,117.203741))

latA = tf.constant([31.777582000000002/180.0], shape=[1], dtype=tf.float64, name="latA")
lonA = tf.constant([117.19014299999999/180.0], shape=[1], dtype=tf.float64, name="lonA")
latB = tf.constant([31.784909999999996/180.0], shape=[1], dtype=tf.float64, name="latB")
lonB = tf.constant([117.203741/180.0], shape=[1], dtype=tf.float64, name="lonB")


ra = tf.constant(6378140, dtype=tf.float64, name="ra")
rb = tf.constant(6356755, dtype=tf.float64, name="rb")
pi = tf.constant(0.017453292519943295769236907*180.00, dtype=tf.float64, name="mpi")

radLatA = latA * pi
radLonA = lonA * pi
radLatB = latB * pi
radLonB = lonB * pi

pA = tf.atan(rb / ra * tf.tan(radLatA))
pB = tf.atan(rb / ra * tf.tan(radLatB))
x = tf.acos(tf.sin(pA) * tf.sin(pB) + tf.cos(pA) * tf.cos(pB) * tf.cos(radLonA - radLonB))
c1 = (tf.sin(x) - x) * (tf.sin(pA) + tf.sin(pB)) ** 2 / tf.cos(x / 2) ** 2
c2 = (tf.sin(x) + x) * (tf.sin(pA) - tf.sin(pB)) ** 2 / tf.sin(x / 2) ** 2
dr = ((ra - rb) / ra) / 8 * (c1 - c2)
# distance = ra * (x + dr)
loss = ra * (x + dr)

with tf.Session()as sess:
    print(sess.run(loss))

