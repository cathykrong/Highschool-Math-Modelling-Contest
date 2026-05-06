# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:57:29 2024

@author: Personal
"""
#Final weights times normalized decision matrix
import pandas as pd
import numpy as np

# Step 1: Read the CSV files
# Example CSV file for vector (n*1)



# Folder path containing the CSV files
folder_path = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/"

vector_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/final weights1.csv'
vector = pd.read_csv(vector_csv_path, header=None).values  # n*1 matrix

# Example CSV file for data matrix (m*n)
matrix_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Normalized decision matrix.csv'
matrix = pd.read_csv(matrix_csv_path, header=None).values  # m*n matrix

# Step 2: Check dimensions for compatibility
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")


# File paths
vector_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/final weights1.csv'
matrix_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Normalized decision matrix.csv'

# Read and convert to numeric
# Load the vector and keep only the second column


# Load the CSV with header
vector_df = pd.read_csv(vector_csv_path)

# Keep only the second column by name or index

# Option 2: By position (if you don't know the name)
vector = vector_df.iloc[:, 1].values  # Second column (0-based index)

print(vector)

matrix = pd.read_csv(matrix_csv_path, header=True).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Reshape the vector if necessary
vector = vector.reshape(-1, 1)  # Ensure it's an n*1 matrix

# Check dimensions
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")

# Perform matrix multiplication if dimensions align
if vector.shape[0] == matrix.shape[1]:
    result = np.dot(matrix, vector)  # m*1 matrix
    print("Resultant Matrix:")
    print(result)
else:
    print("Error: Dimensions do not align for matrix multiplication.")



###################


# Step 1: Load the matrix and vector
matrix = pd.read_csv(matrix_csv_path, header=0).values  # (m+1) x n matrix
vector = pd.read_csv(vector_csv_path, header=0).iloc[:, 1].values  # Extract second column as vector
vector = vector.reshape(-1, 1)  # Reshape to (n x 1)

# Step 2: Separate the first column and the rest of the matrix
first_column = matrix[1:, 0]  # First column, excluding the first row
rest_matrix = matrix[1:, 1:]  # Remaining matrix, excluding the first column


print(first_column)

print(rest_matrix)

# Step 3: Perform matrix multiplication
result_matrix = np.dot(rest_matrix, vector)  # (m x n) x (n x 1) = (m x 1)
#######

# Step 2: Separate labels (first column) and the numeric part of the matrix
labels = matrix.iloc[1:, 0].values  # Retain labels (first column)
rest_matrix = matrix.iloc[1:, 1:].values.astype(float)  # Numeric part of the matrix (m x n)

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
final_result.to_csv('final_result.csv', index=False)

# Step 4: Final result: Combine the first column with the result
final_result = result_matrix.flatten() + first_column

# Print the result
print("Final Result:")
print(final_result)



















# Ensure the inner dimensions match for multiplication
if vector.shape[0] == matrix.shape[1]:
    # Step 3: Perform matrix multiplication
    result = np.dot(matrix, vector)  # m*1 matrix
    print("Resultant Matrix (m*1):")
    print(result)
else:
    print("Error: Dimensions do not align for matrix multiplication.")
