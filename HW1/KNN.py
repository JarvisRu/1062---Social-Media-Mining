from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()
iris_x = iris.data
iris_y = iris.target

# split train & test
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=.27)

# find the best k
k_range = np.arange(1, round(train_x.shape[0] * 0.2))
accuarcies = []
for k in k_range:
    clf = neighbors.KNeighborsClassifier(k)
    iris_clf = clf.fit(train_x, train_y)
    predicted_y = iris_clf.predict(test_x)
    knn_accuracy = accuracy_score(test_y, predicted_y)
    accuarcies.append(knn_accuracy)
    print(k, ":", knn_accuracy)

# print the scatter of k
print(max(accuarcies))
plt.scatter(k_range, accuarcies)
plt.show()


# # create classification - KNN
# clf = neighbors.KNeighborsClassifier()
# iris_clf = clf.fit(train_x, train_y)

# # predict with test_x
# predicted_y = iris_clf.predict(test_x)

# # get the accuracy
# knn_accuracy = accuracy_score(test_y, predicted_y)
# print(knn_accuracy)