# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:08:03 2024

@author: Personal
"""

import os
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd



##
# Step 1: Load your data

file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/prediction/popularity trends.csv'


# Assuming 'data.csv' is your dataset file
data = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True) 

# Example DataFrame with multiple variables
df = pd.DataFrame(data)

# Create group labels for every 12 rows
df['Group'] = np.arange(len(df)) // 48

# Calculate the average for each group
result = df.groupby('Group').mean()


# Add a Year column: Year = Group + 1
result['Year'] = result.index + 1
print(result)


X=result['Year']


# Reshape X_train to be 2D
X = np.array(X).reshape(-1, 1)



# Initialize a dictionary to store regression results
regression_results = {}
predictions = {}
# Perform linear regression for each variable except 'Year'
for column in result.columns:
    if column != 'Year':
        X = result[['Year']].values  # Independent variable
        y = result[column].values   # Dependent variable
        
        # Create and fit the Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Store the results
        regression_results[column] = {
            'Intercept': model.intercept_,
            'Slope': model.coef_[0],
            'R^2': model.score(X, y)
        }
        # Predict for Year = 22 and Year = 23
        pred_years = np.array([[6], [7]])
        predictions[column] = model.predict(pred_years)

# Display regression results
for var, stats in regression_results.items():
    print(f"Regression Results for {var}:")
    print(f"  Intercept: {stats['Intercept']}")
    print(f"  Slope: {stats['Slope']}")
    print(f"  R^2: {stats['R^2']}")
    print()

# Display predictions
for var, pred in predictions.items():
    print(f"Predictions for {var}:")
    print(f"  Year = 22: {pred[0]}")
    print(f"  Year = 23: {pred[1]}")
    print()




# Create matrices
popularity_22 = {sport: pred[0] for sport, pred in predictions.items()}  # Popularity for Year = 22
popularity_23 = {sport: pred[1] for sport, pred in predictions.items()}  # Popularity for Year = 23

# Convert dictionaries to DataFrames
popularity_22_df = pd.DataFrame.from_dict(popularity_22, orient='index', columns=['Popularity_22'])
popularity_23_df = pd.DataFrame.from_dict(popularity_23, orient='index', columns=['Popularity_23'])

# Display the matrices
print("Popularity for Year = 22:")
print(popularity_22_df)

print("\nPopularity for Year = 23:")
print(popularity_23_df)


#save to be csv

folder_path1= 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/prediction/'

output_file2028 = os.path.join(folder_path1, "10linearpop_2028predictiom.csv")
popularity_22_df.to_csv(output_file2028, index=True)


output_file2032 = os.path.join(folder_path1, "11Linearpop_2032predictiom.csv")
popularity_23_df.to_csv(output_file2032, index=True)



