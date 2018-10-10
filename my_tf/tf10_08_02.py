import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

x = tf.Variable(0.0)
# 返回一个op，表示给变量x加1的操作
x_plus_1 = tf.assign_add(x, 1)

# control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前
# 先执行control_dependencies中的内容（在这里就是 x_plus_1）
with tf.control_dependencies([x_plus_1]):
    y = x+1.0                          # +0.0就可以实现累加，不然不变
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in range(5):
        print(session.run(x))
        print(session.run(y))