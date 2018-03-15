from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# load iris dataset
iris = load_iris()
iris_x = iris.data
iris_y = iris.target

# split train & test
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=.3)

# create classification - DT
clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_x, train_y)

# predict with test_x
predicted_y = iris_clf.predict(test_x)

# get the accuracy
dc_accuracy = accuracy_score(test_y, predicted_y)
print(dc_accuracy)
