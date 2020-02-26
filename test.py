from process_data import generate_train_test_data,save_into_csv,get_files_list
import numpy as np 
#from new_model import get_bayes_model
#from new_model import get_svr_model
#from new_model import get_arima_forecast
from new_model import get_lstm_model
#from new_model import train_lstm 
#from new_model import lstm_predict
import pandas as pd 
import timeit
from collections import defaultdict
import statsmodels.api as sm
from sklearn.externals import joblib
import os
import result

if __name__ == "__main__":
	path = "long_term_maximum_CPU_usage.csv"
	lag = 8
	data = generate_train_test_data(0.8, path, lag)
	result.get_lstm_result(path, lag)
	print('a')