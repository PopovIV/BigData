import numpy as np
import pandas as pd
from scipy import stats
import math as m
import matplotlib.pyplot as plt

from sklearn import linear_model
from mlxtend import feature_selection
from sklearn import preprocessing

def init_linear_regression(X, y):
    mlr = linear_model.LinearRegression()
    mlr.fit(X=X, y=y)
    return mlr

def output_linear_regrestion_errors(new_y : np.array, origin_y : np.array):
    err = np.array(origin_y - new_y)
    n = len(err)
    
    rss = err.dot(err)
    rse = np.sqrt(rss / (n - 2))
    tss = np.var(origin_y) * n
    nu = (tss - rss) / tss

    print("RSS = {:.6f}".format(rss))
    print("TSS = {:.6f}".format(tss))
    print("RSE = {:.6f}".format(rse))
    print("NU = {:.6f}".format(nu))

def task_1(series_size : int, variables_nums : int, xs_distribution, y_distribution, noise_distribution):
    X = [xs_distribution(series_size) for _ in range(variables_nums)]
    noise = noise_distribution(series_size)
    y = y_distribution(*(X[i] for i in range(variables_nums)), noise)
    X = np.array(X).transpose()
    mlr = init_linear_regression(X, y)
    
    # task 1 - check multi-linear-regresstion
    print(mlr.coef_)
    
    # task 2 - check multi-linear-model
    output_linear_regrestion_errors(mlr.predict(X=X), y)



def line_criterius(datatime: str):
    data, time = datatime.split(" ")
    day, mon, year = data.split(".")
    hour, minut = time.split(":")
    dt_dict = {
        "year": int(year),
        "mon": int(mon),
        "day": int(day),
        "hour": int(hour),
        "min": int(minut)  
    }
    return dt_dict

def init_data_from_table(csv_path : str):
    # open csv file
    data = pd.read_csv(csv_path, sep=";", index_col=False, encoding="utf-8", comment="#")
    data = pd.DataFrame({"datetime": data["Местное время в Санкт-Петербурге"], "T": data["T"]}).dropna()
    print(data)
    
    new_features = [
        "year",
        "mon",
        "day",
        "hour",
        "min"
    ]
    
    for new_feature in new_features:
        data[new_feature] = data["datetime"].apply(lambda line: line_criterius(line)[new_feature])

    data = data.loc[data["min"] == 0]
    data = data.loc[data["hour"] == 0]
    data = data.drop(["datetime", "min", "hour"], axis=1)
    print(data)

    y_data = data["T"]
    X_data = data.drop("T", axis=1)

    y = np.array(y_data)
    X = np.array(X_data)
    
    return X, y

def init_polynome_regrestion_model(poly_degree : int, X : np.array, y : np.array):
    poly_reg = preprocessing.PolynomialFeatures(degree=poly_degree)
    X_poly = poly_reg.fit_transform(X)
    lr_model = linear_model.LinearRegression()
    
    feature_selector = feature_selection.SequentialFeatureSelector(lr_model, 
        k_features=32,
        forward=True
    )
    
    features = feature_selector.fit(X=X_poly, y=y)
    mX = X_poly[:, features.k_feature_idx_]
    lr_model.fit(X=mX, y=y)
    return mX, lr_model

def task_2(csv_path : str, poly_degree : int):
    X, y = init_data_from_table(csv_path)
    print(X)
    print(y)
    X, mlr = init_polynome_regrestion_model(poly_degree, X, y)
    output_linear_regrestion_errors(mlr.predict(X=X), y)

if __name__ == '__main__':
    # CONSTANTS
    SERIES_SIZE = 20
    VARIABLES_NUM = 3
    NOISE_DISTR = lambda size : stats.norm.rvs(size = size)
    ORIGIN_X_DISTRIBUTION = lambda size : stats.uniform.rvs(size=size, scale = 30)
    ORIGIN_Y_DISTRIBUTION = lambda x1, x2, x3, noise = None : 1 + 3 * x1 - 2 * x2 + x3 + noise
    
    DATA_SRC = "./weather.csv"
    POLY_DEGREE = 6

    # run main func
    #task_1(SERIES_SIZE, VARIABLES_NUM, ORIGIN_X_DISTRIBUTION, ORIGIN_Y_DISTRIBUTION, NOISE_DISTR)
    task_2(DATA_SRC, POLY_DEGREE)
