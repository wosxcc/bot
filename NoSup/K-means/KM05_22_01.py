import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

K = 4 # 类别数目
MAX_ITERS = 20000 # 最大迭代次数
N = 200 # 样本点数目

centers = [[-7, -7], [-8, 7.5], [9.5, -6], [9, 8.5]] # 簇中心

# 生成人工数据集
#data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
data, features = make_blobs(n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=False, random_state=42)
print(data)
print(features)

# 计算类内平均值函数
def clusterMean(data, id, num):
    total = tf.unsorted_segment_sum(data, id, num) # 第一个参数是tensor，第二个参数是簇标签，第三个是簇数目
    count = tf.unsorted_segment_sum(tf.ones_like(data), id, num)
    return total/count

# 构建graph
points = tf.Variable(data)
cluster = tf.Variable(tf.zeros([N], dtype=tf.int64))
centers = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))# 将原始数据前k个点当做初始中心
repCenters = tf.reshape(tf.tile(centers, [N, 1]), [N, K, 2]) # 复制操作，便于矩阵批量计算距离
repPoints = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
sumSqure = tf.reduce_sum(tf.square(repCenters-repPoints), reduction_indices=2) # 计算距离
bestCenter = tf.argmin(sumSqure, axis=1)  # 寻找最近的簇中心
change = tf.reduce_any(tf.not_equal(bestCenter, cluster)) # 检测簇中心是否还在变化
means = clusterMean(points, bestCenter, K)  # 计算簇内均值
# 将粗内均值变成新的簇中心，同时分类结果也要更新
with tf.control_dependencies([change]):
    update = tf.group(centers.assign(means),cluster.assign(bestCenter)) # 复制函数

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    changed = True
    iterNum = 0
    while iterNum < MAX_ITERS: ## and changed == True
        iterNum += 1
        # 运行graph
        [changed, _] = sess.run([change, update])
        [centersArr, clusterArr] = sess.run([centers, cluster])
        # print(clusterArr)
        # print(centersArr)
        print(iterNum)

        # 显示图像
    print(clusterArr)
    print(centersArr)
    fig, ax = plt.subplots()
    ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=clusterArr)
    plt.plot()
    plt.show()