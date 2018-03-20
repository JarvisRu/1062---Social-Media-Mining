from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# load iris dataset
iris = load_iris()

# get two column of data
iris_x = iris.data[:, :2]   # using sepal length & width
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

# generate meshgrid
xx, yy  = np.meshgrid(np.arange(x0_min, x0_max, .02),np.arange(x1_min, x1_max, .02))

# predict class & reshape it for colormesh
predicted_z = iris_clf.predict(np.c_[xx.ravel(), yy.ravel()])
predicted_z = predicted_z.reshape(xx.shape)

# prepare cmap
cmap_background = ListedColormap(['#97cbff', '#adfedc', '#ffad86'])
cmap_scatter = ListedColormap(['#005ab5', '#019858', '#a23400'])

# prepare patches
blue_patch = mpatches.Patch(color="#97cbff", label="iris_Setosa")
green_patch = mpatches.Patch(color="#adfedc", label="iris_Versicolour")
red_patch = mpatches.Patch(color="#ffad86", label="iris_Virginica")

# figure part
plt.figure()
plt.pcolormesh(xx, yy, predicted_z, cmap=cmap_background)                             # predict 
plt.scatter(iris_x[:,0], iris_x[:,1], c=iris_y, cmap=cmap_scatter, marker='o', s=20)  # real data
plt.legend(handles= [blue_patch, green_patch, red_patch])

plt.xlim(xx.min()+.5, xx.max()-.5)
plt.ylim(yy.min()+.5, yy.max()-.5)
plt.xlabel("sepal length")
plt.ylabel("sepal width")

# get the accuracy
knn_accuracy = accuracy_score(test_y, predicted_y)
plt.title("KNN : k=8 / Accuracy={}".format(knn_accuracy))

plt.show()