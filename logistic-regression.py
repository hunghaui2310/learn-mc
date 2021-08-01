import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv', header=None)
# print(data)

true_x = []
true_y = []
false_x = []
false_y = []

for i in data.values:
    if i[2] == 1.:
        true_x.append(i[0])
        true_y.append(i[1])
    else:
        false_x.append(i[0])
        false_y.append(i[1])

plt.scatter(true_x, true_y, marker='o', c='b')
plt.scatter(false_x, false_y, marker='s', c='r')
plt.show()


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def phan_chia(p):
    if p >= 0.5:
        return 1
    else:
        return 0

# ham du doan
def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

def cost_function(features, labels, weights):
    """
    :param features: (100 * 3)
    :param labels:  (100 * 1) co | khong 1 - 0
    :param weights: (3 * 1)
    :return: chi phi cost
    """
    n = len(labels)
    predictions = predict(features, weights)
    cost_class1 = -labels * np.log(predictions)
    cost_class2 = -(1 - labels) * np.log(predictions)
    cost = cost_class1 + cost_class2
    return cost.sum() / n

def update_weight(features, labels, weights, learning_rate):
    """

    :param features: (100 * 3)
    :param labels: (100 * 1)
    :param weights: (3 * 1)
    :param learning_rate: float
    :return: new weight: float
    """
    n = len(labels)
    predictions = predict(features, weights)
    gradient = np.dot(features.T, (predictions - labels))
    gradient = gradient / n
    gradient = gradient * learning_rate
    weights = weights - gradient
    return weights

def train(features, labels, weights, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weights = update_weight(features, labels, weights, learning_rate)
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)
    return weights, cost_history

