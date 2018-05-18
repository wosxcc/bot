#### 0.8164794

from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import random
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from numpy import array
from sklearn.model_selection import train_test_split

def set_missing_ages(data):#使用RandomForestClassifier填补缺失的年龄属性
    age_df = data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y --age
    y = known_age[:, 0]
    # X --feature
    X = known_age[:, 1:]
    # fit to RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1::])
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges
    return data

def attribute_to_number(data):
    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
    data = pd.concat([data, dummies_Pclass,dummies_Embarked, dummies_Sex], axis=1)
    data.drop(['Pclass','Sex', 'Embarked'], axis=1, inplace=True)
    return data


def Scales(data):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(data['Age'].reshape(-1, 1))
    data['Age_scaled'] = scaler.fit_transform(data['Age'].reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(data['Fare'].reshape(-1, 1))
    data['Fare_scaled'] = scaler.fit_transform(data['Fare'].reshape(-1, 1), fare_scale_param)
    SibSp_scale_param = scaler.fit(data['SibSp'].reshape(-1, 1))
    data['SibSp_scaled'] = scaler.fit_transform(data['SibSp'].reshape(-1, 1), SibSp_scale_param)
    Parch_scale_param = scaler.fit(data['Parch'].reshape(-1, 1))
    data['Parch_scaled'] = scaler.fit_transform(data['Parch'].reshape(-1, 1), Parch_scale_param)
    data.drop(['Parch', 'SibSp', 'Fare', 'Age'], axis=1, inplace=True)
    return data


def DataPreProcess(in_data): #数据预处理
    in_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
    data_ages_fitted = set_missing_ages(in_data) #填补缺失的年龄属性
    data = attribute_to_number(data_ages_fitted) #类目属性转化为数值型特征
    data_scaled = Scales(data) #数值归一化

    #划分特征X,和label Y
    data_copy = data_scaled.copy(deep=True)
    data_copy.drop(
        ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_female', 'Sex_male',
         'Age_scaled', 'Fare_scaled', 'SibSp_scaled', 'Parch_scaled'], axis=1, inplace=True)
    data_y = np.array(data_copy)
    data_scaled.drop(['Survived'], axis=1, inplace=True)
    data_X = np.array(data_scaled)

    return data_X,data_y


def LR(data_X,data_y):#tensorflow 实现 Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(data_X,data_y, test_size=0.4, random_state=0)

    y_train = tf.concat([1 - y_train, y_train], 1)
    y_test = tf.concat([1 - y_test, y_test], 1)

    learning_rate = 0.001
    training_epochs = 200
    batch_size = 50
    display_step = 10

    n_samples = X_train.shape[0] #sample_num
    n_features = X_train.shape[1] #feature_num
    n_class = 2

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])

    W = tf.Variable(tf.zeros([n_features, n_class]),name="weight")
    b = tf.Variable(tf.zeros([n_class]),name="bias")

    # predict label
    pred = tf.matmul(x, W) + b

    # accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # cross entropy
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    # train
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: X_train[i * batch_size: (i + 1) * batch_size],
                                           y: y_train[i * batch_size: (i + 1) * batch_size, :].eval()})
                avg_cost = c / total_batch
            plt.plot(epoch + 1, avg_cost, 'co')

            if (epoch + 1) % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "cost=", avg_cost)

        print("Optimization Finished!")
        print("Testing Accuracy:", accuracy.eval({x: X_train, y: y_train.eval()}))

        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    data_X,data_y = DataPreProcess(data)
    LR(data_X,data_y)