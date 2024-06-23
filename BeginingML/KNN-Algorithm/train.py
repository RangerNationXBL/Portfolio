import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
from main import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target #type: ignore

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c = y, cmap = cmap, edgecolor = 'k', s = 20)
plt.show()

clf = KNN(k=5)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(predictions)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)