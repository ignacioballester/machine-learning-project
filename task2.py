import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures

from sklearn.neural_network import MLPClassifier

np.set_printoptions(suppress=True)


def model():

    task1 = ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2']
    task2 = ['LABEL_Sepsis']
    task3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    X = pd.read_csv('feature_X.csv', delimiter=',')
    feature_X_test = pd.read_csv('test_feature_X.csv', delimiter=',')

    timeseries_labels = pd.read_csv('data/task2/train_labels.csv', delimiter=',')
    X, X_test, y, y_test = train_test_split(X, timeseries_labels, test_size=10, random_state=0, shuffle=False)
    timeseries_labels = timeseries_labels.drop('pid', axis=1)
    probs = dict()
    scores = dict()
    num_labels = 2

    # TASK 1 #

    for label in task3:
        current_model = MLPClassifier(alpha=3, n_iter_no_change=30, max_iter=500, verbose=False)
        current_model.fit(X, y[label])
        probs[label] = current_model.predict_proba(feature_X_test)[:, 1]

        # from sklearn.metrics import roc_auc_score
        # roc = roc_auc_score(y_test, current_model.predict_proba(X_test)[:, 1])

        from sklearn.linear_model import ElasticNet
        regr = ElasticNet(random_state=0, alpha = 0.5)
        regr.fit(X, y[label])
        regr.predict(feature_X_test)

        from sklearn.model_selection import cross_val_score
        score = cross_val_score(current_model, X, y[label], cv=3, scoring='roc_auc')
        print(score)
        if num_labels == 0:
            break
        num_labels -= 1

    for label in task1:
        current_model = MLPClassifier(alpha=3, n_iter_no_change=30, max_iter=500, verbose=False)
        current_model.fit(X, y[label])
        probs[label] = current_model.predict_proba(feature_X_test)[:, 1]

        # from sklearn.metrics import roc_auc_score
        #roc = roc_auc_score(y_test, current_model.predict_proba(X_test)[:, 1])


        from sklearn.metrics import accuracy_score
        predictions_train = current_model.predict(X)
        predictions_test = current_model.predict(X_test)
        train_score = accuracy_score(predictions_train, y[label])
        print("score on train data: ", train_score)
        test_score = accuracy_score(predictions_test, y_test[label])
        print("score on test data: ", test_score)
        scores[label] = test_score
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(current_model, X, y[label], cv=3, scoring='roc_auc')
        print(score)
        if num_labels ==0:
            break
        num_labels -= 1



    submission_df = pd.DataFrame(probs).to_numpy()
    np.savetxt("submission.csv", submission_df, delimiter=",", fmt='%f')
    print(scores)


def extract_features():

    timeseries = pd.read_csv('data/task2/test_features.csv', delimiter=',')
    timeseries_labels = pd.read_csv('data/task2/train_labels.csv', delimiter=',')
    ids = timeseries['pid'].unique()
    def start_end_diff(col):
        values = col.dropna(how='any',inplace = False).to_numpy()
        if len(values) <= 1:
            return 0, 0, 0
        else:
            return values[0], values[-1], values[0] - values[-1]


    test_ids_num = -1
    X = np.array(range(0, 205))
    counter = len(ids)
    for id in ids:
        if counter %500 == 0:
            print(counter)
        data_rows = timeseries[timeseries['pid'] == id]
        age = data_rows.iloc[0]['Age']
        df_feat_extr= data_rows.drop(['Time', 'pid', 'Age'], axis=1, inplace=False)
        features_num = len(df_feat_extr.columns) * 6
        features =  [0.0] * features_num
        feature_idx = 0
        for col_name in df_feat_extr:
            col =df_feat_extr[col_name]
            if col.isnull().values.all():
                feature_idx+= 6
                continue
            features[feature_idx] = col.mean()
            features[feature_idx+1] = col.max()
            features[feature_idx + 2] = col.min()
            features[feature_idx + 3] , features[feature_idx + 4], features[feature_idx + 5] = start_end_diff(col)
            feature_idx +=6

        features.append(age)
        X = np.vstack((X, features))
        if test_ids_num ==0:
            break
        test_ids_num -= 1


        counter -= 1
    np.savetxt("test_feature_X.csv", X, delimiter=",", fmt='%f')


model()