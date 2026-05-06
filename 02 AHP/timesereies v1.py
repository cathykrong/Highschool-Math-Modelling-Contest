# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:28:01 2024

@author: Personal
"""
#%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import datetime
# from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# def parser(x):
#  	return datetime.strptime('%Y-%m')
# Replace 'your_file.csv' with the actual path to your CSV file

file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/prediction/popularity/multiTimeline.csv'


# Assuming 'data.csv' is your dataset file
data = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True) # , squeeze=True, date_parser=parser)
# data['Month'] = pd.to_datetime(data['Month'])
data.index = data.index.to_period('M')
data.index = data.index.to_timestamp()
print(data.info())

plt.figure(figsize=(16, 8), dpi=150) 
data.plot() 
# plt.plot(data)
# plt.title('Time Series Data')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.show()



result = adfuller(data['Frisbee'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# for column in data:
#     plot_acf(data[column].diff().dropna(), title=column+'.diff acf')
#     plot_pacf(data[column].diff().dropna(), title=column+'.diff pacf')
#     plt.show()

# ARIMA model parameters (p, d, q)
d = 1  # Differencing order
p = 1  # AutoRegressive (AR) order
q = 2  # Moving Average (MA) order
forecast_steps = 96
# pd.plotting.register_matplotlib_converters()
# pd.plotting.deregister_matplotlib_converters()

for column in data:
    print(column)
    model = ARIMA(data[column], order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    # model_fit.plot_predict(dynamic=False)
    # plt.show()
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()

    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast.row_labels)
    # fcpara = 
    print(forecast)
    plt.figure(figsize=(14,7))
    plt.plot(data[column], label='Training Data')
    plt.plot(forecast_series, label='Forecasted Data', color='green')
    plt.fill_between(forecast.row_labels, 
                     forecast.conf_int().iloc[:, 0], 
                     forecast.conf_int().iloc[:, 1], 
                     color='k', alpha=.15)
    plt.title('ARIMA Model Evaluation - ' + column)
    plt.xlabel('Date')
    plt.ylabel('Number of Births')
    plt.legend()
    plt.show()



print(forecast_series)


# Assuming 'data.csv' is your dataset file
# data = pd.read_csv('data.csv', parse_dates=['date_column'], index_col='date_column')