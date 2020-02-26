# -*- coding:UTF-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from sklearn.svm import SVR
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.models import load_model

def get_svr_model(kernel='linear'):
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
    linear_svr = SVR(kernel=kernel)

    return linear_svr

def get_bayes_model():
    """ 
    linear_bayes.fit(x_train, y_train)
    y_test = linear_bayes.predict(x_test)

    """
    linear_bayes = linear_model.BayesianRidge()

    return linear_bayes


def get_arima_forecast(time_series, arma_p, arma_q, arma_diff=1):
    arima_model = ARIMA(time_series, (arma_p, arma_diff, arma_q)).fit()
    return arima_model.forecast()[0][0]


def get_lstm_model():
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1), units=100, return_sequences=False))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def train_lstm(lstm_model,x_train,y_train,model_name, time_span, batch_size, validation_split=0):
    model = lstm_model
    # time_span = 9
    x_train = x_train.reshape(x_train.shape[0],time_span,1)
    model.fit(x_train,y_train,batch_size =batch_size, epochs=3,validation_split = validation_split)
    model.save(model_name + '_lstm_model.h5')
    return model

def lstm_predict(x_test,model):
    #model = load_model('lstm_model.h5')
    predict = model.predict(x_test)
    return predict