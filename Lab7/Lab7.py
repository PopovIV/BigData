import numpy as np
import pandas as pd
from scipy import stats
import math as m
from dateutil import parser, rrule
from datetime import datetime, time, date

from sklearn import linear_model
from mlxtend import feature_selection
from sklearn import preprocessing

N = 20
VAR = 3

def task1():
    func = lambda x1, x2, x3, e: 3 * x1 - 2 * x2 + x3 + e
    nois = stats.norm.rvs(size=N)

    X = np.array([[stats.uniform.rvs(scale=20) for _ in range(VAR)] for _ in range(N)])
    y = func(*(X[:, i] for i in range(VAR)), nois)

    lr_model = linear_model.LinearRegression()
    lr_model.fit(X=X, y=y)

    print(lr_model.coef_)

    # rss, rse, nu

    err = np.array(y - lr_model.predict(X=X))
    n = len(err)

    # Мера рассогласования
    rss = err.dot(err)
    # Стандартная ошибка остатков
    # измеряет количество изменчивости
    rse = np.sqrt(rss / (n - 2))
    # измеряет полную дисперсию
    tss = np.var(y) * n
    # Корреляционное отношение
    # Чем ближе к 1, тем лучше модель объясняет большую часть дисперсии отклика
    nu = (tss - rss) / tss

    print(f"RSS {rss}")
    print(f"TSS {tss}")
    print(f"RSE {rse}")
    print(f"NU {nu}")

def task2():
    data_raw = pd.read_csv('LONDON.csv')
    # Give the variables some friendlier names and convert types as necessary.
    data_raw['mean_temp'] = data_raw['mean_temp'].astype(float)

    data_raw['date'] = [datetime.strptime(str(d), '%Y%m%d') for d in data_raw['date']]
    
    # Extract out only the data we need.
    data = data_raw.loc[:, ['date', 'mean_temp']]
    print(data)

    y_data = data["mean_temp"]
    X_data = data["date"]
    print(X_data)

    y = np.array(y_data)
    X = np.array(X_data)

    degree = 6
    poly_reg = preprocessing.PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    lr_model = linear_model.LinearRegression()
    feature_selector = feature_selection.SequentialFeatureSelector(lr_model, k_features=32, forward=True)
    features = feature_selector.fit(X=X_poly, y=y)
    print(features.k_feature_idx_)

    lr_model = linear_model.LinearRegression()
    mX = X_poly[:, features.k_feature_idx_]
    lr_model.fit(X=mX, y=y)
    predict_y = lr_model.predict(X=mX)

    err = np.array(y - predict_y)
    n = len(err)

    rss = err.dot(err)

    rse = np.sqrt(rss / (n - 2))

    tss = np.var(y) * n

    nu = (tss - rss) / tss

    print(f"N {n}")
    print(f"RSS {rss}")
    print(f"TSS {tss}")
    print(f"RSE {rse}")
    print(f"NU {nu}")

print("Task 1:\n")
task1()
print("Task 2:\n")
task2()
