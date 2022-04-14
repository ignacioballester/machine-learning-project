import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor


def sol():
    train_data = pd.read_csv('data/task_1B/train.csv', float_precision='round_trip')
    y = train_data.values[:, 1]
    X = np.matrix(train_data.values[:, 2:])


    square_x = np.square(X)
    exp_x = np.exp(X)
    cos_x = np.cos(X)
    ones_x = np.ones((X.shape[0], 1))

    complete_X = np.hstack((X, square_x, exp_x, cos_x, ones_x))
    complete_X = np.asarray(complete_X)

    X_train, X_test, y_train, y_test = train_test_split(complete_X, y, test_size=0.001, random_state=42, shuffle=False)

    reg = SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2', fit_intercept=False, random_state=0)
    reg.fit(X_train, y_train)
    print(reg.coef_)
    return reg.coef_

def my():
    pass

def save(reg):
    with open('sub_task1B.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in reg:
            writer.writerow([i])
    csv_file.close()


save(my())
