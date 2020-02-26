# -*- coding:UTF-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from sklearn.svm import SVR
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


# https://blog.csdn.net/llx1026/article/details/77942838
def get_svr_forecast_workflow(time_series, divide_length, lag_num=5):
    """
    linear_svr = SVR(kernel='linear')
    linear_svr.fit(x_train, y_train.ravel())
    linear_svr_predict = linear_svr.predict(x_test)
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(x_train, y_train.ravel())
    poly_svr_predict = poly_svr.predict(x_test)
    rbf_svr = SVR(kernel='rbf')
    rbf_svr.fit(x_train, y_train.ravel())
    rbf_svr_predict = rbf_svr.predict(x_test)
    """
    # 训练数据的因变量
    train_y_series = time_series.shift(-1)[:(divide_length - 1)]
    # 训练数据的自变量
    train_x_df = pd.DataFrame()
    for i in range(lag_num):
        train_x_df[i] = time_series.shift(i)[:(divide_length - 1)].fillna(0)
    # 预测数据的自变量
    predict_x_df = pd.DataFrame()
    for j in range(lag_num):
        predict_x_df[j] = time_series.shift(j)[(divide_length - 1):-1].fillna(0)

    linear_svr = SVR(kernel='linear')
    linear_svr.fit(train_x_df.values, train_y_series.values)
    predict_y_list = linear_svr.predict(predict_x_df.values)

    return predict_y_list


# https://www.2cto.com/net/201606/518910.html
# https://blog.csdn.net/qq_37353105/article/details/80612561
def get_bayes_forecast_workflow(time_series, divide_length, lag_num=5):
    # 训练数据的因变量
    train_y_series = time_series.shift(-1)[:(divide_length - 1)]
    # 训练数据的自变量
    train_x_df = pd.DataFrame()
    for i in range(lag_num):
        train_x_df[i] = time_series.shift(i)[:(divide_length - 1)].fillna(0)
    # 预测数据的自变量
    predict_x_df = pd.DataFrame()
    for j in range(lag_num):
        predict_x_df[j] = time_series.shift(j)[(divide_length - 1):-1].fillna(0)

    linear_bayes = linear_model.BayesianRidge()
    # BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,normalize=False, tol=0.001, verbose=False)
    linear_bayes.fit(train_x_df.values, train_y_series.values)
    predict_y_list = linear_bayes.predict(predict_x_df.values)

    return predict_y_list


# 基于ARIMA模型预测
def get_arima_forecast(time_series, arma_p, arma_q, arma_diff=1):
    arima_model = ARIMA(time_series.values, (arma_p, arma_diff, arma_q)).fit()
    return arima_model.forecast()[0][0]


# 基于ARIMA模型进行预测
#https://baike.baidu.com/item/ARIMA%E6%A8%A1%E5%9E%8B/10611682?fr=aladdin
#https://www.jianshu.com/p/4130bac8ebec
#https://blog.csdn.net/meng_shangjy/article/details/79714747
def get_arima_forecast_workflow(time_series, divide_length, max_ar=4, max_ma=2):
    # 根据BIC标准，确定ARIMA模型的阶数
    arma_ic_result = sm.tsa.arma_order_select_ic(time_series.values, max_ar=max_ar, max_ma=max_ma,
                                                 ic='aic')
    arma_p = arma_ic_result['aic_min_order'][0]
    arma_q = arma_ic_result['aic_min_order'][1]
    if arma_p==0 and arma_q==0:
        arma_p=3
        arma_q=2
    
    # 基于ARIMA模型预测
    predict_y_list = []
    for data_i in range(divide_length, len(time_series)):
        train_x = time_series[:data_i]
        try:
            predict_value = get_arima_forecast(train_x, arma_p, arma_q)
        except Exception as e:
            try:
                predict_value = get_arima_forecast(train_x, arma_p - 1, arma_q)
            except Exception as e:
                try:
                    predict_value = get_arima_forecast(train_x, arma_p + 1, arma_q)
                except Exception as e:
                    try:
                        predict_value = get_arima_forecast(train_x, arma_p, arma_q - 1)
                    except Exception as e:
                        predict_value = time_series.loc[data_i-1]
        predict_y_list.append(predict_value)
    return predict_y_list
