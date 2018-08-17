from  sklearn import tree
import pydotplus
x = [[0,0],[1,1]]
y = [0,1]


clf= tree.DecisionTreeClassifier()

clf = clf.fit(x,y)
print(clf.predict([[0.6,0.6]]))