from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# load iris gerneally
iris = load_iris()
iris_x = iris.data
iris_y = iris.target

# data with target = ?
# iris_x[iris_y==0]
# iris_x[iris_y==1]
# iris_x[iris_y==2]

# draw a scatter
plt.figure()
plt.scatter(iris_x[iris_y==0][:,0], iris_x[iris_y==0][:,1], color='r', label='iris_Setosa')
plt.scatter(iris_x[iris_y==1][:,0], iris_x[iris_y==1][:,1], color='g', label='iris_Versicolour')
plt.scatter(iris_x[iris_y==2][:,0], iris_x[iris_y==2][:,1], color='b', label='iris_Virginica')
plt.legend()
plt.show()

