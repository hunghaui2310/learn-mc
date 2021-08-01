from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # chia dataset
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)

model = DecisionTreeClassifier()
myModel = model.fit(X_train, y_train)
X_New = np.array([[6.0, 3.23, 4.5, 2.0]])
# kiem tra xem voi X_new nay thi thuoc nhom hoa nao
print(myModel.predict(X_New))
# danh gia do chinh xac
print(myModel.score(X_test, y_test))

# print(iris_dataset.data)
# print(iris_dataset.target)