from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()

# get two column of data
iris_x = iris.data[:, 2:4]
iris_y = iris.target

# split train & test
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=.27)

# create classification - KNN
clf = naive_bayes.GaussianNB()
iris_clf = clf.fit(train_x, train_y)

# predict with test_x
predicted_y = iris_clf.predict(test_x)

print(test_y==predicted_y)