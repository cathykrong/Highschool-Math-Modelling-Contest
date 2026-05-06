# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:35:11 2024

@author: Personal
"""
import pandas as pd
import numpy as np

%matplotlib inline
# Folder path containing the CSV files
folder_path = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights"

folder_path_ML = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/05 ML/"


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

#######################################
## K mean

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your data (78 sports x 16 features matrix)



# Step 1: Separate `merged_df` into labels and data
label = merged_df[['Label']]  # Extract the labels (78 x 1)
data = merged_df.drop(columns=['Label'])  # Extract the data (78 x 16)

print(f"Vector shape: {data.shape}")
# Step 2: Normalize the data (important for K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Run K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust `n_clusters` as needed
clusters = kmeans.fit_predict(data_scaled)

# Step 4: Attach labels back with their clusters
merged_result = pd.DataFrame({
    'Label': label['Label'],
    'Cluster': clusters
})

# Step 5: Save or display the results
print("Clustered Labels:")
print(merged_result.head())

# Optional: Save to CSV
# Step 6: Save the results to the defined folder path
output_file = folder_path + "clustered_labels.csv"
merged_result.to_csv(output_file, index=False)

print(f"Results have been saved to: {output_file}")

#####K-mean results are not great


### ########################################################################Supervised learning

#Read in olympic indicator data

# File paths
matrix_withID = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Normalized decision matrix withID.csv'

# Step 1: Load the matrix and vector
matrix_withID_df = pd.read_csv(matrix_withID, header=0)  

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# Define the folder path
folder_path = "E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/05 ML/"

olympic_csv_path = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/05 ML/Olympic Indicator2024.csv'

# Step 1: Load the data and prepare features and target
olympic_df = pd.read_csv(olympic_csv_path, header=0)  # Load as DataFrame to retain headers
print(olympic_df)
print(matrix_df)
print(matrix_withID_df)

#merge vith merged_result

# Merge the datasets on the 'ID' column
merged_data1 = pd.merge(matrix_withID_df, olympic_df, on='ID', how='inner')  # Change 'how' as needed

print(merged_data1)
label = olympic_df[['Label']]  # Extract labels for reference
ID = olympic_df[['ID']]  # Extract ID for reference

data = merged_data1.drop(columns=['Label', 'Unnamed: 0','ID','Olympic_outcome'])  # Features--Cluster
target = merged_data1['Olympic_outcome']  # Target variable

print(data)
print(target)

#In feature table data, we have both categorical and numerical variables. Cluster is a categorical variable and need to be hotkeyed.

# Step 1: Exclude a specific categorical variable
exclude_variable = 'Cluster'

# Get the names of all other variables dynamically
remaining_features = [col for col in data.columns if col != exclude_variable]

# Step 2: Further divide into numerical and categorical
numerical_features = data[remaining_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_features = data['Cluster']

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)



# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # Set sparse=False to return a dense array


# Convert Series to NumPy array and reshape
categorical_features = categorical_features.to_numpy().reshape(-1, 1)

# Fit and transform the data
category_encoded = encoder.fit_transform(categorical_features)


# Create a DataFrame with the encoded features
encoded_columns = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Cluster']))

# Combine the encoded columns with the original DataFrame (excluding the original column)
data_encoded = pd.concat([data.drop(columns=['Cluster']), encoded_columns], axis=1)


# Separate features and target
X = data_encoded
y = target


print(X)
print(y)

## Try logistic regression


# Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
    ]
)

# Step 2: Build the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 3: Predict on the test set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for the positive class

# Step 4: Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.2f}")

# Step 5: Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_pred_prob)))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Combine IDs, true labels, and predictions into a DataFrame
results_test = pd.DataFrame({
    'True': y_test,
    'Predicted Label': y_pred,
    'Probability Positive Class': y_pred_prob
})

# Export to CSV
output_file = folder_path_ML + "06logistic_predictions test2024.csv"
results_test.to_csv(output_file, index=False)
print(f"Predictions exported to {output_path}")


#############Apply to full dataset
# Predict on the entire dataset
y_pred = model.predict(X)
y_pred_prob = model.predict_proba(X)[:, 1]  # Probability of the positive class


# Check dimensions
print(f"Vector shape: {y_pred.shape}")

print(f"Vector shape: {y_pred_prob.shape}")

print(f"Vector shape: {label.shape}")


# Combine IDs, true labels, predictions, and probabilities into a DataFrame
results_full = pd.DataFrame({
    'Label':merged_data1['Label'],
    'True Label': y,
    'Predicted Label': y_pred,
    'Probability Positive Class': y_pred_prob
})


print(results_full)
# Export to CSV
output_file_log_full = folder_path_ML + "08logistic_full_predictions2024.csv"
results_full.to_csv(output_file_log_full, index=False)
print(f"Predictions exported to {output_file_log_full}")







########################################Lasso ############




import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Train the Lasso Logistic Regression model
model_lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
model_lasso.fit(X_train, y_train)

# Step 3: Predict on the test set
y_predmodel_lasso = model_lasso.predict(X_test)
y_pred_probmodel_lasso = model_lasso.predict_proba(X_test)[:, 1]  # Probability scores for the positive class


# Step 5: Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probmodel_lasso)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_pred_probmodel_lasso)))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

#############Apply to full dataset
# Predict on the entire dataset
y_pred_lasso = model_lasso.predict(X)
y_pred_prob_lasso = model_lasso.predict_proba(X)[:, 1]  # Probability of the positive class

# Combine IDs, true labels, predictions, and probabilities into a DataFrame
results_full_lasso = pd.DataFrame({
    'Label':merged_data1['Label'],
    'True Label': y,
    'Predicted Label': y_pred_lasso,
    'Probability Positive Class': y_pred_prob_lasso
})


print(results_full_lasso)
# Export to CSV
output_file_log_full_lasso = folder_path_ML + "07logistic__lassofull_predictions.csv"
results_full.to_csv(output_file_log_full_lasso, index=False)
print(f"Predictions exported to {output_file_log_full_lasso}")


*** Lasso results are as same as logistic regression













