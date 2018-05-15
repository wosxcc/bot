# Titanic题目实战

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

data = pd.read_csv('train.csv')

print(data.info())  # 查看数据概况

# 取部分特征字段用于分类，并将所有缺失的字段填充为0
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

# 两种分类分别为幸存和死亡，‘Survived’字段是其中一种分类的标签
# 新增‘Deceased’表示第二种分类的标签，取值为‘Survived’字段取非
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, \
                                                    test_size=0.2, random_state=42)
# 构建计算图
# 声明输入数据占位符
# shape参数的第一个元素为None,表示可以同时放入任意条记录
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])
# 声明变量
W = tf.Variable(tf.random_normal([6, 2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')
# 创建容器vars。它收集了tensor变量W和b。之后，tensorflow将这一容器保存
tf.add_to_collection('vars', W)
tf.add_to_collection('vars', b)

# 逻辑回归的公式
y_pred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
# 声明代价函数：使用交叉熵作为代价函数
cross_entroy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10))
cost = tf.reduce_mean(cross_entroy)

# 加入优化算法:其中0.001是learning rate
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# 定义saver
saver = tf.train.Saver()

# 构建训练迭代过程
with tf.Session() as sess:
    # 初始化所有变量，必须最先执行
    #    sess.run(tf.global_variables_initializer())
    tf.global_variables_initializer().run()
    # 以下为训练迭代，迭代10轮
    for epoch in range(10):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], y: [y_train[i]]}
            # 通过session.run接口触发执行
            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print('Training complete!')

    pred = sess.run(y_pred, feed_dict={X: X_train})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_train, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print('Accuracy on validation set: %.9f' % accuracy)
    # 存储变量
#    saver.save(sess,'./modelVar/model.ckpt')

#    上面的代码运行结束后，当前目录下出现四个文件：
#    my-model.ckpt.meta
#    my-model.ckpt.data-*
#    my-model.ckpt.index
#    checkpoint
#    利用这四个文件就能恢复出 w1和w2这两个变量。

W = tf.Variable(tf.random_normal([6, 2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')
X = tf.placeholder(tf.float32, shape=[None, 6])
yHat = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
Weights = []
with tf.Session() as sess1:
    tf.global_variables_initializer().run()
    model_file = tf.train.latest_checkpoint('./xxx')
    saver.restore(sess1, model_file)
    all_vars = tf.get_collection('vars')
    for i, v in enumerate(all_vars):
        #        print('v',v)
        #        print('vname',v.name)
        v_ = v.eval()  # sess.run(v)
        #        print(i,v_)
        Weights.append(v_)
    Weights9 = Weights[14]
    bias9 = Weights[15]
    y_Hat = sess1.run(yHat, feed_dict={X: X_test, W: Weights9, b: bias9})

# 预测测试数据结果
testdata = pd.read_csv('test.csv')
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
XTest = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

# 开启session进行预测
with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    predictions = np.argmax(sess2.run(yHat, feed_dict={X: XTest, W: Weights9, b: bias9}), 1)

# 构建提交结果的数据结构，并将结果存储为csv文件
submission = pd.DataFrame({'PassengerId': testdata['PassengerId'], \
                           'Survived': predictions})
submission.to_csv('mySubmission201712.csv', index=False)