# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:35:11 2024

@author: Personal
"""
import pandas as pd
import numpy as np
# Folder path containing the CSV files
folder_path = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights"

# File paths
matrix_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Normalized decision matrix.csv'
vector_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/final weights1.csv'

# Step 1: Load the matrix and vector
matrix_df = pd.read_csv(matrix_csv_path, header=0)  # Load as DataFrame to retain headers
vector = pd.read_csv(vector_csv_path, header=0).iloc[:, 1].values  # Extract second column as vector
vector = vector.astype(float).reshape(-1, 1)  # Ensure numeric and reshape to (n x 1)

# Step 2: Separate labels (first column) and the numeric part of the matrix
labels = matrix_df.iloc[0:, 0].values  # Retain labels (first column)
rest_matrix = matrix_df.iloc[0:, 1:].values.astype(float)  # Numeric part of the matrix (m x n)

print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {rest_matrix.shape}")
print(f"Vector shape: {labels.shape}")
# Step 3: Perform matrix multiplication
result_matrix = np.dot(rest_matrix, vector)  # (m x n) x (n x 1) = (m x 1)

# Step 4: Combine labels and results
final_result = pd.DataFrame({
    'Label': labels,
    'Result': result_matrix.flatten()  # Convert to 1D for DataFrame
})

# Step 5: Print or Save Results
print("Final Result:")
print(final_result)
# Save to a CSV file if needed
final_result.to_csv('E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/final_result.csv', index=False)

# Check dimensions
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {rest_matrix.shape}")
print(f"Matrix shape: {final_result.shape}")




# prepare matrix for unsupervised learning


# Assuming `rest_matrix` is a DataFrame for this operation
# Convert rest_matrix to a DataFrame if it is currently a NumPy array
rest_matrix_df = pd.DataFrame(rest_matrix, columns=[f'Feature{i}' for i in range(1, rest_matrix.shape[1] + 1)])

# Add the labels to rest_matrix_df
rest_matrix_df['Label'] = labels  # Ensure `labels` corresponds to rest_matrix rows

# Assuming final_result is already a DataFrame with 'Label' and 'Result'
# Perform a left join on 'Label'
merged_df = pd.merge(rest_matrix_df, final_result, on='Label', how='left')

# Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# Save to CSV if needed
merged_df.to_csv('E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Res_normDeciMatr.csv', index=False)




