import pandas as pd
import matplotlib.pyplot as plt

dataFrame = pd.read_csv('Advertising.csv', header=0)
x = dataFrame.values[:, 2]
y = dataFrame.values[:, 4]
plt.scatter(x, y, marker='o')
plt.show()

# print(x)

# ham du doan
def predict(new_radio, weight, bias):
    return weight*new_radio + bias

# ham tinh chi phi
def cost_function(x, y, weight, bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight * x[i] + bias))**2
    return sum_error / n

# cap nhat trong so
def update_weight(x, y, weight, bias, learning_rate):
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * x[i] * (y[i] - (weight * x[i] + bias))
        bias_temp += -2 * (y[i] - (weight * x[i] + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias / n) * learning_rate

    return weight, bias

# training
def training(x, y, weight, bias, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight, bias = update_weight(x, y, weight, bias, learning_rate)
        cost = cost_function(x, y, weight, bias)
        cost_history.append(cost)

    return weight, bias, cost_history

weight = 0.03
bias = 0.0014
learning_rate = 0.001 # toc do hoc
iter = 10 # so lan lap
w, b, cost_his = training(x, y, weight, bias, learning_rate, iter)

print('Ket Qua = ', w, b, cost_his)

print('Gia tri du doan = ')
print(predict(19, w, b))

so_lan_lap = [i for i in range(iter)]
plt.plot(so_lan_lap, cost_his)
plt.show()