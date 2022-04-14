from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from numpy import genfromtxt
from sklearn.model_selection import RepeatedKFold
import math
from sklearn.metrics import mean_squared_error

my_data = genfromtxt('data/task_1A/train.csv', delimiter=',')
my_data = my_data[1:, :]
complete_y = my_data[:, 0]
complete_X = my_data[:, 1:]

kf = RepeatedKFold(n_splits=10, random_state=0, n_repeats=70)

rmse_list = []
result = []
alphas = [0.1, 1, 10, 100, 200]
count = 0
for alpha in alphas:
    count = 0
    for train_index, test_index in kf.split(complete_X):
        count += 1
        X = my_data[train_index, 1:]
        y = my_data[train_index, 0]
        y_test = my_data[test_index, 0]
        X_test = my_data[test_index, 1:]
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        pred = ridge.predict(X_test)
        rmse_list.append(mean_squared_error(y_test, pred, squared=False))
    print("Average for alpha {} --> {} ".format(alpha, np.average(rmse_list)))
    result.append(np.average(rmse_list))
    rmse_list.clear()
print(count)