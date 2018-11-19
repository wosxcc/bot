import numpy as np

# N 是一个batch的样本数量; D_in是输入维度;
# H 是隐藏层向量的维度; D_out是输出维度.
N, D_in, H, D_out = 20, 100, 100, 10

# 创建随机的输入输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

print("真实值",y)
# 随机初始化权重参数
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(50000):
    # 前向计算, 算出y的预测值
    h = x.dot(w1)
    h_relu =h # np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算并打印误差值
    loss = np.square(y_pred - y).sum()


    # 在反向传播中, 计算出误差关于w1和w2的导数
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
print(t, loss,"预测值",y_pred)


# -0.87390075  0.56570065 -0.86921654 -0.612732    0.73536282  1.27897324
#-0.87390075  0.56570065 -0.86921654 -0.612732    0.73536282  1.27897324



# -1.03071983 -1.03101988  0.67950525  1.05058903  0.20094151 -0.75738235
# -1.03071983 -1.03101988  0.67950525  1.05058903  0.20094151 -0.75738235