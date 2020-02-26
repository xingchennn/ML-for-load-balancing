import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import model_function as mf


# 获取指定文件夹下特定格式的所有文件
def get_files_list(file_folder, abs_dir_flag=False, file_type='.xlsx'):
    if abs_dir_flag:
        path = file_folder + '\\'
    else:
        path = os.getcwd() + '\\' + file_folder + '\\'
    files_list = []
    if not os.path.exists(path):
        return files_list
    for file_name in os.listdir(path):
        if os.path.splitext(file_name)[1] == file_type:
            index = file_name.rfind(".")
            files_list.append([file_name[:index], path + file_name])
    return files_list


# 保存数据成CSV文件
def save_into_csv(file_name, file_folder, data_df):
    path = os.getcwd() + '\\' + file_folder + '\\'
    if not os.path.exists(path):
        os.makedirs(path)
    data_df.to_csv(path + file_name + '.csv', encoding="utf-8-sig", index=0)


def draw_trend_plot(time_series, image_folder, image_file):
    path = os.getcwd() + '\\' + image_folder + '\\'
    if not os.path.exists(path):
        os.makedirs(path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    time_series.plot()
    plt.grid(True)
    plt.savefig(path + "\\" + image_file + '.png', format='png')


def get_processed_data():
    # 读取数据
    data_df = pd.DataFrame()
    original_file_folder = "original_data"
    files_list = get_files_list(original_file_folder, abs_dir_flag=False, file_type='.csv')
    for file_i in range(len(files_list)):
        file_info = files_list[file_i]
        file_name = file_info[0]
        file_path = file_info[1]
        print("<<<<<<<<<<<<<<<<<<get data from the file: " + file_name)
        temp_df = pd.read_csv(file_path, header=None)
        data_df = data_df.append(temp_df)

    # 生成新的ID TaskId=JobId+MachinedId
    data_df = data_df.reset_index(drop=True)
    data_df[20] = data_df[2].astype(np.str) + "-" + data_df[4].astype(np.str)

    # 保存中间数据长度 
    data_dict = {}
    max_length = 300
    mean_cpu_usage_rate_df = pd.DataFrame()
    maximum_memory_usage_df = pd.DataFrame()
    maximum_cpu_usage_df = pd.DataFrame()
    short_term_memory = pd.DataFrame()
    long_term_memory = pd.DataFrame()
    short_term_cpu= pd.DataFrame()
    long_term_cpu = pd.DataFrame()

    for task_id in data_df[20].unique()[:1000]:
        print("<<<<<<<<<<<<<<<<<< process the task id(=Jobid+Machinedid) :" + task_id)
        temp_df = data_df[data_df[20] == task_id]
        if len(temp_df) < 150:

            # maximum memory usage
            short_term_memory = pd.concat([short_term_memory, pd.DataFrame({task_id:temp_df[10].tolist()})], axis=1)
            short_term_cpu = pd.concat([short_term_cpu, pd.DataFrame({task_id:temp_df[13].tolist()})], axis=1)
            continue

        data_dict[task_id] = temp_df

        # maximum memory usage
        long_term_memory = pd.concat([long_term_memory, pd.DataFrame({task_id:temp_df[10].tolist()})], axis=1)
        # maximu.tm CPU usage
        long_term_cpu = pd.concat([long_term_cpu, pd.DataFrame({task_id:temp_df[13].tolist()})], axis=1)

    # # 保存数据
    # path = os.getcwd() + '\\temp_data'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # pickle.dump(data_dict, open(path + '\\data_dict.pkl', 'wb'))
    # 保存数据
    processed_file_folder = "processed_data"
    save_into_csv("short_term_cpu_usage_rate", processed_file_folder, short_term_memory)
    save_into_csv("long_term_cpu_usage_rate", processed_file_folder, long_term_memory)
    save_into_csv("short_term_maximum_memory_usage", processed_file_folder, short_term_cpu)
    save_into_csv("long_term_maximum_cpu_usage", processed_file_folder, long_term_cpu)


# 基于SVR进行预测
def get_svr_prediction_result(processed_file_folder="processed_data"):
    files_list = get_files_list(processed_file_folder, abs_dir_flag=False, file_type='.csv')
    for file_info in files_list:
        file_name = file_info[0]
        file_path = file_info[1]
        print("<<<<<<<<<<<<<<<get the data from the file: " + file_name)
        statics_df = pd.DataFrame()
        predict_df = pd.DataFrame()
        data_df = pd.read_csv(file_path).dropna()
        if len(data_df) < 30:
            continue
        for data_column in data_df.columns:
            print("<<<<<<<<<<<svr model predict the task_id: " + data_column)
            divide_length = int(len(data_df[data_column]) * 0.8)
            true_y_list = data_df[data_column][divide_length:].values.tolist()
            predict_y_list = mf.get_svr_forecast_workflow(data_df[data_column], divide_length)
            predict_df[data_column + "_true"] = true_y_list
            predict_df[data_column + "_predict"] = predict_y_list
            statics_df[data_column] = predict_df[data_column + "_predict"] / predict_df[data_column + "_true"]
        draw_trend_plot(statics_df.T.median(), "chart_data", file_name + "_svr_median_plot")
        save_into_csv(file_name + "_svr_prediction", "record_data", predict_df)
        save_into_csv(file_name + "_svr_prediction", "result_data", statics_df)


# 基于bayes进行预测
def get_bayes_prediction_result(processed_file_folder="processed_data"):
    files_list = get_files_list(processed_file_folder, abs_dir_flag=False, file_type='.csv')
    for file_info in files_list:
        file_name = file_info[0]
        file_path = file_info[1]
        print("<<<<<<<<<<<<<<<get the data from the file: " + file_name)
        statics_df = pd.DataFrame()
        predict_df = pd.DataFrame()
        data_df = pd.read_csv(file_path).dropna()
        if len(data_df) < 30:
            continue
        for data_column in data_df.columns:
            print("<<<<<<<<<<<bayes model predict the task_id: " + data_column)
            divide_length = int(len(data_df[data_column]) * 0.8)
            true_y_list = data_df[data_column][divide_length:].values.tolist()
            predict_y_list = mf.get_bayes_forecast_workflow(data_df[data_column], divide_length)
            predict_df[data_column + "_true"] = true_y_list
            predict_df[data_column + "_predict"] = predict_y_list
            statics_df[data_column] = predict_df[data_column + "_predict"] / predict_df[data_column + "_true"]
        draw_trend_plot(statics_df.T.median(), "chart_data", file_name + "_bayes_median_plot")
        save_into_csv(file_name + "_bayes_prediction", "record_data", predict_df)
        save_into_csv(file_name + "_bayes_prediction", "result_data", statics_df)


# 基于ARIMA进行预测
def get_arima_prediction_result(processed_file_folder="processed_data"):
    files_list = get_files_list(processed_file_folder, abs_dir_flag=False, file_type='.csv')
    for file_info in files_list:
        file_name = file_info[0]
        file_path = file_info[1]
        print("<<<<<<<<<<<<<<<get the data from the file: " + file_name)
        statics_df = pd.DataFrame()
        predict_df = pd.DataFrame()
        data_df = pd.read_csv(file_path).dropna()
        
        if len(data_df) < 30:
            continue
        for data_column in data_df.columns:
            print("<<<<<<<<<<<arima model predict the task_id: " + data_column)
            divide_length = int(len(data_df[data_column]) * 0.8)
            true_y_list = data_df[data_column][divide_length:].values.tolist()
            predict_y_list = mf.get_arima_forecast_workflow(data_df[data_column], divide_length, max_ar=5, max_ma=3)
            predict_df[data_column + "_true"] = true_y_list
            predict_df[data_column + "_predict"] = predict_y_list
            statics_df[data_column] = predict_df[data_column + "_predict"] / predict_df[data_column + "_true"]
        draw_trend_plot(statics_df.T.median(), "chart_data", file_name + "_arima_median_plot")
        save_into_csv(file_name + "_arima_prediction", "record_data", predict_df)
        save_into_csv(file_name + "_arima_prediction", "result_data", statics_df)


# 绘制正常准确率的图
def get_normal_accuracy_plot():
    result_file_folder = "result_data"
    files_list = get_files_list(result_file_folder, abs_dir_flag=False, file_type='.csv')
    for file_info in files_list:
        file_name = file_info[0]
        file_path = file_info[1]
        print("<<<<<<<<<<<<<<<draw normal accuracy plot for the data file: " + file_name)
        data_df = pd.read_csv(file_path)
        accuracy_df = 1 - np.abs(data_df - 1)
        draw_trend_plot(accuracy_df.T.median(), "accuracy_chart", file_name + "_accuracy_plot")
        save_into_csv(file_name + "_accuracy", "accuracy_data", accuracy_df)

if __name__ == '__main__':
    get_processed_data()