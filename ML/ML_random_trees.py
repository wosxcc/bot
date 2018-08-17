

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.datasets import load_iris
iris=load_iris()

print(iris['data'],iris['target'])

rf = RandomForestRegressor()

rf.fit(iris.data[:150],iris.target[:150])

print('预计值:',rf.predict(iris.data[[100]]),'实际值',iris.target[[100]])



