# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:08:03 2024

@author: Personal
"""

import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


# Folder path containing the CSV files
folder_path = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/prediction/"


# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Ensure there's at least one file
if len(csv_files) < 1:
    raise ValueError("No CSV files found in the folder!")

# Read the first CSV file as the base DataFrame
base_file = os.path.join(folder_path, csv_files[0])
base_df = pd.read_csv(base_file)

# Remove the first two rows
base_df = base_df.iloc[2:]

# Clean column names: remove ": (Worldwide)" and strip whitespace
base_df.columns = [col.replace(": (Worldwide)", "").strip() for col in base_df.columns]

# Replace '<1' with 0.5 in the base file
base_df.replace('<1', 0.5, inplace=True)

# Loop through the remaining files and left join them to the base DataFrame
for file in csv_files[1:]:
    file_path = os.path.join(folder_path, file)
    new_df = pd.read_csv(file_path)
    
    # Remove the first two rows
    new_df = new_df.iloc[2:]
    
    # Clean column names
    new_df.columns = [col.replace(": (Worldwide)", "").strip() for col in new_df.columns]
    
    # Replace '<1' with 0.5
    new_df.replace('<1', 0.5, inplace=True)
    
   


# Save the combined DataFrame to a new file if needed
output_file = os.path.join(folder_path, "combined_cleaned.csv")
base_df.to_csv(output_file, index=False)

# Display the final combined DataFrame
print("Combined and Cleaned DataFrame:")
print(base_df)
