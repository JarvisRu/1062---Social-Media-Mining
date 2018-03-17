from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()

# get two column of data
iris_x = iris.data[:, :2]
iris_y = iris.target

# split train & test
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=.27)

# # find the best k
k_range = np.arange(1, round(train_x.shape[0] * 0.2))
accuarcies = []
for k in k_range:
    clf = neighbors.KNeighborsClassifier(k)
    iris_clf = clf.fit(train_x, train_y)
    predicted_y = iris_clf.predict(test_x)
    knn_accuracy = accuracy_score(test_y, predicted_y)
    accuarcies.append(knn_accuracy)
    print(" K = ",k, "->", knn_accuracy)

# print the figure
print(max(accuarcies))
plt.plot(k_range, accuarcies, '--o')
plt.xlabel("Value of K for KNN")
plt.ylabel("Accuracy")
plt.title("The Highest Accuracy is {}".format(max(accuarcies)))
plt.show()



# basic predict
# # create classification - KNN
# clf = neighbors.KNeighborsClassifier(8)
# iris_clf = clf.fit(train_x, train_y)

# # predict with test_x
# predicted_y = iris_clf.predict(test_x)

# # get the accuracy
# knn_accuracy = accuracy_score(test_y, predicted_y)
# print(knn_accuracy)