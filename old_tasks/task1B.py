import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.metrics import mean_squared_error

my_data = np.genfromtxt('data/task_1B/train.csv', delimiter=',')
my_data = my_data[1:, 1:]

X_quadratic = np.square(my_data[:, 1:])
X_exp = np.exp(my_data[:, 1:])
X_cos = np.cos(my_data[:, 1:])
X_const = np.full((X_quadratic.shape[0], 1), 1)

complete_X = np.concatenate((my_data[:, 1:], X_quadratic, X_exp, X_cos, X_const), axis=1)
X, X_test, y, y_test = train_test_split(complete_X, my_data[:, 0], test_size=10, random_state=0, shuffle=False)

# model = Lasso(alpha=0.1)
model = RANSACRegressor(
    SGDRegressor(loss='squared_error', penalty='l2', fit_intercept=False, random_state=0),
    max_trials=10000,  # Number of Iterations
    min_samples=10,  # Minimum size of the sample
    loss='squared_error',  # Metrics for loss
    residual_threshold=10,  # Threshold
    random_state=70
)
model.fit(X, y)
pred = model.predict(X_test)
print(mean_squared_error(y_test, pred, squared=False))
print(model.estimator_.coef_)
