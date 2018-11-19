import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_dao(x):
    return x*(1-x)
bias = [0.35, 0.60]
weight = [0,0,0,0,0,0,0,0] #[0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55]
output_layer_weights = [0.4, 0.45, 0.5, 0.55]
i1 = 0.05
i2 = 0.10
target1 = 0.33
target2 = 0.66
learning_rate = 0.5 #学习速率
numIter = 50000#迭代次数
for i in range(numIter):
    #正向传播
    neth1 = i1*weight[1-1] + i2*weight[2-1] + bias[0]
    neth2 = i1*weight[3-1] + i2*weight[4-1] + bias[0]
    outh1 = sigmoid(neth1)
    outh2 = sigmoid(neth2)
    neto1 = outh1*weight[5-1] + outh2*weight[6-1] + bias[1]
    neto2 = outh1*weight[7-1] + outh2*weight[8-1] + bias[1]
    outo1 = sigmoid(neto1)
    outo2 = sigmoid(neto2)

    target1_outo1 = outo1 - target1
    outo1_neto1 = outo1 * (1 - outo1)
    outh1_neth1 = outh1 * (1 - outh1)

    neth1_weight1 = i1
    neth1_weight2 = i2

    target2_outo2 = outo2 - target2
    outo2_neto2 = outo2*(1-outo2)
    outh2_neth2 = outh2*(1-outh2)

    neth2_weight3 = i1
    neth2_weight4 = i2

    weight[5 - 1] -= learning_rate*target1_outo1 * outo1_neto1 * outh1
    weight[6 - 1] -= learning_rate*target1_outo1 * outo1_neto1 * outh2
    weight[7 - 1] -= learning_rate*target2_outo2 * outo2_neto2 * outh1
    weight[8 - 1] -= learning_rate*target2_outo2 * outo2_neto2 * outh2

    weight[1 - 1] -= learning_rate*(target1_outo1 * outo1_neto1 * weight[5 - 1]+target2_outo2 * outo2_neto2 * weight[7 - 1]) * outh1_neth1 * neth1_weight1
    weight[2 - 1] -= learning_rate*(target1_outo1 * outo1_neto1 * weight[6 - 1]+target2_outo2 * outo2_neto2 * weight[8 - 1]) * outh1_neth1 * neth1_weight2
    weight[3 - 1] -= learning_rate*(target2_outo2 * outo2_neto2 * weight[5 - 1]+target1_outo1 * outo1_neto1 * weight[7 - 1]) * outh2_neth2 * neth2_weight3
    weight[4 - 1] -= learning_rate*(target2_outo2 * outo2_neto2 * weight[7 - 1]+target1_outo1 * outo1_neto1 * weight[8 - 1]) * outh2_neth2 * neth2_weight4

    if i==0:
        print("权重变化后",weight[1 - 1],weight[2 - 1],weight[3 - 1],weight[4 - 1],weight[5- 1],weight[6- 1],weight[7 - 1],weight[8 - 1])

print("运行结果",outo1,outo2)
print("权重变化后",weight[1 - 1],weight[2 - 1],weight[3 - 1],weight[4 - 1],weight[5- 1],weight[6- 1],weight[7 - 1],weight[8 - 1])