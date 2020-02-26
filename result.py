#utf-8
from process_data import generate_train_test_data,save_into_csv,get_files_list
import numpy as np 
from new_model import get_bayes_model,get_svr_model,get_arima_forecast,get_lstm_model,train_lstm,lstm_predict
import new_model
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import timeit
from collections import defaultdict
import statsmodels.api as sm
from sklearn.externals import joblib
import os

def get_svr_result():
    model = get_svr_model()
    path = 'test_data'
    data = generate_train_test_data(0.8,path,9)
    for key, val in data.items():
        x_train, y_train = val[0][0],val[0][1]
        x_test, y_test = val[1][0],val[1][1]
        x_train = x_train.values
        x_test = x_test.values
        y_train = np.array(y_train.values).ravel()
        y_test = y_test.values.ravel()
        model.fit(x_train, y_train)
        y_predict = []
        time_spend = []
        time_dict = defaultdict(list)
        pre_dict = defaultdict(list)
        for x,y_t,id_name in zip(x_test,y_test,val[2]):
            start = timeit.default_timer()
            y = model.predict([x])
            spend = timeit.default_timer() - start
            time_spend.append(spend)
            acc = 1- abs((y[0] - y_t)/y_t)
            pre_dict[id_name].append(acc)
            time_dict[id_name].append(spend)
            y_predict.append(y[0])
        df = pd.DataFrame(pre_dict)
        tm = pd.DataFrame(time_dict)
        tm = tm.astype('str')
        save_into_csv(key + "_svr", "new_result", df)
        save_into_csv(key + "_svr_time", "new_result", tm)


def get_bayes_result(path, lag):
    model = get_bayes_model()
    path1 = path + '.csv'
    data = generate_train_test_data(0.8, path1, lag)
    for key, val in data.items():
        x_train,y_train = val[0][0],val[0][1]
        x_test,y_test = val[1][0],val[1][1]
        x_train = x_train.values
        x_test = x_test.values
        y_train = np.array(y_train.values).ravel()
        y_test = y_test.values.ravel()
        if os.path.exists("model/" + path + "_" + str(lag) + ".model"):
            joblib.load("model/" + path + "_" + str(lag) + ".model", model)
        else:
            print('train')
            model.fit(x_train,y_train)
        y_predict = []
        time_spend = []
        time_dict = defaultdict(list)
        pre_dict = defaultdict(list)
        for x,y_t,id_name in zip(x_test,y_test,val[2]):
            start = timeit.default_timer()
            y = model.predict([x])
            spend = timeit.default_timer() - start
            time_spend.append(spend)
            acc = 1- abs((y[0] - y_t)/ 1 )
            #acc = 1- abs((y[0] - y_t)/(y_t + 0.001))
            pre_dict[id_name].append(acc)
            spend = format(spend, 'f')
            time_dict[id_name].append(str(spend))
            y_predict.append(y[0])

        df = pd.DataFrame()
        for key1, value in pre_dict.items():
            df = pd.concat([df, pd.DataFrame({key1: value})], axis=1)
        tm = pd.DataFrame()
        for key1, value in time_dict.items():
            tm = pd.concat([tm, pd.DataFrame({key1: value})], axis=1)


        print ("<<<<<<<<<<<<<<<save result:" + key + "_bayes")
        save_into_csv(path + "_bayes", "new_result",df)
        print ("<<<<<<<<<<<<<<<save result:" + key + "_bayes_time")
        save_into_csv(path + "_bayes_time","new_result",tm)

def bayes_point_predict(path, window_size, train_set_size, point, length):
    model = get_bayes_model()
    path1 = 'processed_data/' + path + '.csv'
    df = pd.read_csv(path1)
    time_train = []
    for len in window_size:
        path1 = path + '.csv'
        data = generate_train_test_data(0.8, path1, len)
        for key, val in data.items():
            x_train, y_train = val[0][0], val[0][1]
            x_test, y_test = val[1][0], val[1][1]
            x_train = x_train.values
            x_test = x_test.values
            y_train = np.array(y_train.values).ravel()
            y_test = y_test.values.ravel()
            if os.path.exists("model/" + path + "_" + str(train_set_size) + ".model"):
                joblib.load("model/" + path + "_" + str(train_set_size) + ".model", model)
            else:
                print('train')
                start = timeit.default_timer()
                model.fit(x_train[:train_set_size], y_train[:train_set_size])
                spend = timeit.default_timer() - start
                time_train.append(spend)
        y_predict = []
        time_spend = []

        train_dict = defaultdict(list)
        pre_dict = defaultdict(list)
        time_dict = defaultdict(list)
        for column in df.columns:
            # model = get_bayes_model()
            if df[column].size < point + length:
                print('no such point')
                continue
            df[column] = df[column].fillna(0.00001)
            # x_train, y_train = df[column][point - len - 10: point - 10], df[column][point]
            for i in range(length):
                x_test, y_test = df[column][point + i - len - 10: point + i - 10], df[column][point + i]
                # x_train = x_train.values
                # print(x_train)
                x_test = x_test.values
                # y_train = np.array(y_train).ravel()
                y_test = y_test.ravel()
                start = timeit.default_timer()
                y = model.predict([x_test])
                spend = timeit.default_timer() - start
                time_spend.append(spend)
                #acc = 1 - abs((y[0] - y_test) / (y_test + 0.001))
                acc = 1 - abs((y[0] - y_test) / 1)
                pre_dict[column].append(acc)
                spend = format(spend, 'f')
                time_dict[column].append(str(spend))
                train_dict[column].append(str(time_train[0]))
                y_predict.append(y[0])
        df1 = pd.DataFrame(pre_dict)
        tm = pd.DataFrame(time_dict)
        tt = pd.DataFrame(train_dict)
        print("<<<<<<<<<<<<<<<save result:" + path + "_bayes")
        save_into_csv(path + str(len) + "_bayes", "new_predict", df1)
        print("<<<<<<<<<<<<<<<save result:" + path + "_bayes_time")
        save_into_csv(path + str(len) + "_bayes_time", "new_predict", tm)
        print("<<<<<<<<<<<<<<<save result:" + path + "_bayes_train")
        save_into_csv(path + str(len) + "_bayes_train", "new_predict", tt)

def get_arima_result(processed_file_folder, lag):
    files_list = [processed_file_folder]

    for file_info in files_list:
        file_name = file_info
        file_path = 'processed_data/' + file_info + '.csv'
        print("<<<<<<<<<<<<<<<get the data from the file: " + file_name)
        data_df = pd.read_csv(file_path)
        # lag = 9
        x = []
        for data_column in data_df.columns:
            temp = pd.DataFrame()
            temp[data_column] = data_df[data_column].dropna()
            divide_length = int(len(temp[data_column]) * 0.8)
            true_x_list = temp[data_column][:divide_length].values.tolist()
            x.extend(true_x_list)
        print('train arima model')
        arma_ic_result = sm.tsa.arma_order_select_ic(x, max_ar=5, max_ma=3,
                                                     ic='aic')
        arma_p = arma_ic_result['aic_min_order'][0]
        arma_q = arma_ic_result['aic_min_order'][1]

        result = {}
        time_df = {}
        for data_column in data_df.columns:
            temp = pd.DataFrame()
            temp[data_column] = data_df[data_column].dropna()
            print("<<<<<<<<<<<arima model predict the task_id:" + data_column)
            divide_length = int(len(temp[data_column]) * 0.8)
            true_y_list = temp[data_column][divide_length - lag:]
            y_true = temp[data_column][divide_length:-1]
            train_x_df = pd.DataFrame()
            for j in range(lag):
                train_x_df[j] = true_y_list.shift(j)
            train_x_df = train_x_df.dropna()
            accuracy = []
            spend_time = []
            for data, y_true in zip(train_x_df.values, y_true.values):
                start = timeit.default_timer()
                try:
                    y_pre = get_arima_forecast(data, arma_p, arma_q)
                except Exception as e:
                    try:
                        y_pre = get_arima_forecast(data, arma_p - 1, arma_q)
                    except Exception as e:
                        try:
                            y_pre = get_arima_forecast(data, arma_p + 1, arma_q)
                        except Exception as e:
                            try:
                                y_pre = get_arima_forecast(data, arma_p, arma_q - 1)
                            except Exception as e:
                                y_pre = data[-1]
                spend = timeit.default_timer() - start
                acc = 1 - abs((y_pre - y_true)/1)
                #acc = 1 - abs((y_pre - y_true)/y_true +0.001)
                accuracy.append(acc)
                spend_time.append(str(spend))
            result[data_column] = accuracy
            time_df[data_column] = spend_time
        df = pd.DataFrame()
        for key1, value in result.items():
            df = pd.concat([df, pd.DataFrame({key1: value})], axis=1)
        tm = pd.DataFrame()
        for key1, value in time_df.items():
            tm = pd.concat([tm, pd.DataFrame({key1: value})], axis=1)
        print ("<<<<<<<<<<<<<<<save result:" + file_name + "_arima")
        save_into_csv(file_name+'_arima', "new_result", df)
        print ("<<<<<<<<<<<<<<<save result:" + file_name + "_arima_time")
        save_into_csv(file_name+'_arima_time', "new_result", tm)

def arima_point_predict(path, window_size, train_set_size, point, leng):
    file_name = path
    file_path = 'processed_data/' + path + '.csv'
    print("<<<<<<<<<<<<<<<get the data from the file: " + file_name)
    data_df = pd.read_csv(file_path)
    # lag = 9
    x = []
    for data_column in data_df.columns[:train_set_size]:
        temp = pd.DataFrame()
        temp[data_column] = data_df[data_column].dropna()
        divide_length = int(len(temp[data_column]) * 0.8)
        true_x_list = temp[data_column][:divide_length].values.tolist()
        x.extend(true_x_list)
    print('train arima model')
    start = timeit.default_timer()
    arma_ic_result = sm.tsa.arma_order_select_ic(x, max_ar=5, max_ma=3,
                                                 ic='aic')
    arma_p = arma_ic_result['aic_min_order'][0]
    arma_q = arma_ic_result['aic_min_order'][1]

    for length in window_size:
        result = {}
        time_df = {}
        train_df = {}

        for column in data_df.columns:
            x = data_df[column].fillna(0.0001).tolist()
            if len(x) < point + leng + 1:
                print('no such point')
                continue
            start = timeit.default_timer()
            arma_ic_result = sm.tsa.arma_order_select_ic(x, max_ar=5, max_ma=3,
                                                         ic='aic')
            spend = timeit.default_timer() - start
            arma_p = arma_ic_result['aic_min_order'][0]
            arma_q = arma_ic_result['aic_min_order'][1]
            accuracy = []
            spend_time = []
            train_df[column] = [str(spend)]
            for i in range(leng):
                x_train, y_train = data_df[column][point + i - length - 10: point + i - 10], data_df[column][point + i]
                x_test, y_test = data_df[column][point + i - length - 10: point + i - 10], data_df[column][point + i]
                start = timeit.default_timer()
                try:
                    y_pre = get_arima_forecast(x_train, arma_p, arma_q)
                except Exception as e:
                    try:
                        y_pre = get_arima_forecast(x_train, arma_p - 1, arma_q)
                    except Exception as e:
                        try:
                            y_pre = get_arima_forecast(x_train, arma_p + 1, arma_q)
                        except Exception as e:
                            try:
                                y_pre = get_arima_forecast(x_train, arma_p, arma_q - 1)
                            except Exception as e:
                                y_pre = y_train
                spend = timeit.default_timer() - start
                acc = 1 - abs((y_pre - y_train) / y_train + 0.001)
                accuracy.append(acc)
                spend_time.append(str(spend))
                result[column] = accuracy
                time_df[column] = spend_time
        df1 = pd.DataFrame()
        for key1, value in result.items():
            df1 = pd.concat([df1, pd.DataFrame({key1: value})], axis=1)
        tm = pd.DataFrame()
        for key1, value in time_df.items():
            tm = pd.concat([tm, pd.DataFrame({key1: value})], axis=1)
        tt = pd.DataFrame()
        for key1, value in train_df.items():
            tt = pd.concat([tt, pd.DataFrame({key1: value})], axis=1)
        print("<<<<<<<<<<<<<<<save result:" + path + "_arima")
        save_into_csv(path + str(length) + '_arima', "new_predict", df1)
        print("<<<<<<<<<<<<<<<save result:" + path + "_arima_time")
        save_into_csv(path + str(length) + '_arima_time', "new_predict", tm)
        print("<<<<<<<<<<<<<<<save result:" + path + "_arima_train")
        save_into_csv(path + str(length) + '_arima_train', "new_predict", tt)

def get_lstm_result(path, lag):
    # lag = 9
    #model = get_lstm_model()
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1), units=100, return_sequences=False))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    path1 = path + '.csv'
    data = generate_train_test_data(0.8,path1,lag)
    for key, val in data.items():
        x_train,y_train = val[0][0],val[0][1]
        x_test,y_test = val[1][0],val[1][1]
        x_train = x_train.values
        x_test = x_test.values
        y_train = np.array(y_train.values).ravel()
        y_test = y_test.values.ravel()
        print('x', x_train)
        model = train_lstm(model,x_train,y_train,key, lag, 32, 0.1)
        y_predict = []
        time_spend = []
        time_dict = defaultdict(list)
        pre_dict = defaultdict(list)
        for x, y_t, id_name in zip(x_test, y_test, val[2]):
            start = timeit.default_timer()
            x = x.reshape(1,lag,1)
            y = lstm_predict(x,model)
            spend = timeit.default_timer() - start
            time_spend.append(spend)
            acc = 1- abs((y[0] - y_t)/1)
            #acc = 1- abs((y[0] - y_t)/(y_t + 0.001))
            pre_dict[id_name].append(acc)
            time_dict[id_name].append(spend)
            y_predict.append(y[0])
        df = pd.DataFrame()
        for key1, value in pre_dict.items():
            df = pd.concat([df, pd.DataFrame({key1: value})], axis=1)
        tm = pd.DataFrame()
        for key1, value in time_dict.items():
            tm = pd.concat([tm, pd.DataFrame({key1: value})], axis=1)

        print("<<<<<<<<<<<<<<<save result:" + key + "_lstm")
        save_into_csv(path + "_lstm", "new_result", df)
        print("<<<<<<<<<<<<<<<save result:" + key + "_lstm_time")
        save_into_csv(path + "_lstm_time", "new_result", tm)

def lstm_point_predict(path, window_size, train_set_size, point, length):
    model = get_lstm_model()
    path1 = './processed_data/' + path + '.csv'
    df = pd.read_csv(path1)

    for len in window_size:
        path1 = path + '.csv'
        data = generate_train_test_data(0.8, path1, len)
        time_dict = defaultdict(list)
        pre_dict = defaultdict(list)
        train_dict = defaultdict(list)
        for key, val in data.items():
            x_train, y_train = val[0][0], val[0][1]
            x_test, y_test = val[1][0], val[1][1]
            x_train = x_train.values
            x_test = x_test.values
            y_train = np.array(y_train.values).ravel()
            y_test = y_test.values.ravel()
            start = timeit.default_timer()
            model = train_lstm(model, x_train[:train_set_size], y_train[:train_set_size], key, len, 1, 0.1)
            spend = timeit.default_timer() - start

        for column in df.columns:
            train_dict[column] = [spend]
            df[column] = df[column].fillna(0.0001)
            for i in range(length):
                x_train, y_train = df[column][point + i - len - 10: point + i - 10], df[column][point + i]
                x_test, y_test = df[column][point + i - len - 10: point + i - 10], df[column][point + i]
                x_train = x_train.values
                x_test = x_test.values
                y_train = np.array(y_train).ravel()
                y_test = y_test.ravel()
                y_predict = []
                time_spend = []
                time_train = defaultdict(list)
                start = timeit.default_timer()
                x = np.array([x_test]).reshape(1, len, 1)
                y = lstm_predict(x, model)
                spend = timeit.default_timer() - start
                time_spend.append(spend)
                acc = 1 - abs((y[0] - y_train) / y_train + 0.001)
                pre_dict[column].append(acc)
                time_dict[column].append(spend)
                y_predict.append(y[0])
        df1 = pd.DataFrame(pre_dict)
        tm = pd.DataFrame(time_dict)
        tt = pd.DataFrame(train_dict)
        print("<<<<<<<<<<<<<<<save result:" + path + "_lstm")
        save_into_csv(path + str(len) + "_lstm", "new_predict", df1)
        print("<<<<<<<<<<<<<<<save result:" + path + "_lstm_time")
        save_into_csv(path + str(len) + "_lstm_time", "new_predict", tm)
        print("<<<<<<<<<<<<<<<save result:" + path + "_lstm_train")
        save_into_csv(path + str(len) + "_lstm_time", "new_predict", tt)

if __name__ == "__main__":
    # run bayes model, 参数为文件名：short_term_cpu_usage_rate， window size - 1：10-1=9
    get_bayes_result('short_term_memory',11)
    get_bayes_result('long_term_memory', 11)
    get_bayes_result('short_term_cpu',11)
    get_bayes_result('long_term_cpu',11)
    #预测2，输入文件名， windows size , train_set_size, 预测点; 四个参数
    # window size 为 list 即[10, 15, 20]， 第二个参数
    #train_set_size 为第三个参数
    #第四个参数 为预测起点
    # 第五个参数为预测长度
    #bayes_point_predict('long_term_maximum_cpu_usage',[10, 15],100, 30, 30)
    # run arima model,参数为文件名：short_term_cpu_usage_rate， window size - 1：10-1=9
    # 预测2，输入文件名、jobid， windows size 、预测点
    get_arima_result('short_term_memory',11)
    get_arima_result('long_term_memory',11)
    get_arima_result('short_term_cpu', 11)
    get_arima_result('long_term_cpu',11)
    #arima_point_predict('short_term_cpu_usage_rate', [10, 15, 20], 10, 60, 5)
    # run lstm model
    # 预测2，输入文件名、jobid， windows size 、预测点
    get_lstm_result('short_term_memory', 11)
    get_lstm_result('long_term_memory',11)
    get_lstm_result('short_term_cpu',11)
    get_lstm_result('long_term_cpu',11)
    #lstm_point_predict('short_term_cpu_usage_rate', [10, 15, 20],100, 60, 5)


