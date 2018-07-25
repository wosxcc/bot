import tensorflow as tf
x2=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
y2=tf.constant(2.0)
#注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
z2=tf.multiply(x2,y2)
sess =tf.Session()

print(sess.run(z2))
sess.close()