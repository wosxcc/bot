import matplotlib.pyplot as plt
import numpy as np
from  scipy.cluster.vq import *
from sklearn.datasets.samples_generator import make_blobs

centers = [[-7, -7], [-8, 7.5], [9.5, -6], [9, 8.5]] # 簇中心
N = 300
# 生成人工数据集
#data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
data, features = make_blobs(n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=False, random_state=42)

print(data.shape)

centroids,variance = kmeans(data,4)
code,distance=vq(data,centroids)

# print('code',code)
#
# print('distance',distance)
# print('variance',variance)

# print(data.transpose()[0])
fig, ax = plt.subplots()
for  i in range(len(code)):
    if code[i]==1:
        ax.scatter(data[i].transpose()[0], data[i].transpose()[1], marker='v', s=30, c='y')
    elif code[i] == 2:
        ax.scatter(data[i].transpose()[0], data[i].transpose()[1], marker='o', s=30, c='r')
    elif code[i] == 3:
        ax.scatter(data[i].transpose()[0], data[i].transpose()[1], marker='s', s=30, c='g')
    else:
        ax.scatter(data[i].transpose()[0], data[i].transpose()[1], marker='*', s=30, c='b')

# ax.scatter(data.transpose()[0], data.transpose()[1], marker='v', s=10 ,c='r')
plt.plot()
plt.show()


