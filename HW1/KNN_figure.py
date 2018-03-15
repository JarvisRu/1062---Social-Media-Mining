from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()

# get two column of data
iris_x = iris.data[:, :2]
iris_y = iris.target

# split train & test
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=.27)

# create classification - KNN
clf = neighbors.KNeighborsClassifier(8)
iris_clf = clf.fit(train_x, train_y)

# predict with test_x
predicted_y = iris_clf.predict(test_x)

# get x[0] and x[1] limit
x0_min, x0_max = iris_x[:, 0].min()-1, iris_x[:,0].max()+1 
x1_min, x1_max = iris_x[:, 1].min()-1, iris_x[:,1].max()+1


# generate meshgrid and cmap for figure
xx, yy  = np.meshgrid(np.arange(x0_min, x0_max, .02),np.arange(x1_min, x1_max, .02))
cmad_background = ListedColormap(['red','green','blue'])


# # get the accuracy
# knn_accuracy = accuracy_score(test_y, predicted_y)
# print(knn_accuracy)