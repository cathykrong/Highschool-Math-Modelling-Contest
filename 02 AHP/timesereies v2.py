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


##########liniear model TEST

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Step 1: Load your data
# Replace this with your own dataset
data = {
    'Year': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],  # Square feet
    'Sport': [31.91666667,28.66666667,24.08333333,20,17.66666667,17.33333333,17,17,17.08333333,16.41666667,16,15.41666667,13.33333333,13.16666667,11.91666667,11.91666667,11.75,11.16666667,11.75,12.91666667,11.25]  # Price in dollars
}
df = pd.DataFrame(data)

# Features (X) and target (y)
#X = df[['Size']]
#y = df['Price']

X=data['Year']


# Reshape X_train to be 2D
X = np.array(X).reshape(-1, 1)


y=data['Sport']


# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
predictions = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Predict a new value
new_size = [[22]]  # Predict for a 2028 2032 value
predicted_price = model.predict(new_size)
print(f"Predicted value: {predicted_price[0]}")


# Predict a new value
new_size = [[24]]  # Predict for a 2028 2032 value
predicted_price = model.predict(new_size)
print(f"Predicted value: {predicted_price[0]}")







