import  tensorflow as tf
import  numpy as np

weight_into=[0.1, 0.5, 1.2, 2.4, 0.75, 0.8, 0.33, 0.89, 1.2, 2.1]

weight_count=len(weight_into)

# x_data = np.random.random((3, 1000))
x_data=np.float32(np.random.rand(weight_count,1000))
# 系数矩阵的shape必须是（3，1）。如果是（3，）会导致收敛效果差，猜测可能是y-y_label处形状不匹配
y_data = np.dot(weight_into, x_data) + 1.  ##矩阵相乘

b=tf.Variable(tf.zeros(1))
w=tf.Variable(tf.random_uniform([1,weight_count],-1.0,1.0))
y=tf.matmul(w,x_data)+b
loss=tf.reduce_mean(tf.square(y-y_data))      ##计算损失值平方差
optimizer=tf.train.GradientDescentOptimizer(0.001) ##优化函数
train=optimizer.minimize(loss)
init =tf.initialize_all_variables() ##初始化变量
sess= tf.Session()
sess.run(init)
for step in range(0,200001):
    sess.run(train)
    if step%2000==0:
        print(step,sess.run(w),sess.run(b))