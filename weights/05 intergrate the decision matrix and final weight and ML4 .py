 # -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:35:11 2024
#this program will test on the data that for 3 recent added or removed= breaking, climbing, baseball
#Test on three continuously in olympic: Basketball, swimming, badminton
# changed the normalized decison matrix for baseball due to male dominated 
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
final_result.to_csv('E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/final_score result2.csv', index=False)

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

# Step 6: Calculate SSE for different numbers of clusters
sse = []
k_values = range(1, 11)  # Test for 1 to 10 clusters
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Step 7: Plot SSE vs. Number of Clusters
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.title('SSE vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(k_values)
plt.grid()
plt.show()


#####K-mean results are not great


### ########################################################################Supervised learning

#Read in olympic indicator data

# File paths
matrix_withID = 'E:/02 Cathy/09 math contest/10 math modelling/2024 contest high school/weights/Normalized decision matrix withID_baseball adjusted.csv'

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

from scipy.stats import norm


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
# Select rows where ID is baseball

test_rows = merged_data1[merged_data1['ID'].isin([10])]

merged_data1_rest = merged_data1[~merged_data1['ID'].isin([10])]



print(merged_data1)
label = olympic_df[['Label']]  # Extract labels for reference
ID = olympic_df[['ID']]  # Extract ID for reference

data = merged_data1_rest.drop(columns=['Label', 'Unnamed: 0','ID','Olympic_outcome'])  # Features--Cluster
target = merged_data1_rest['Olympic_outcome']  # Target variable

print(data)
print(target)

print(f"Vector shape: {data.shape}")
#In feature table data, we have both categorical and numerical variables. Cluster is a categorical variable and need to be hotkeyed.

# Step 1: Exclude a specific categorical variable
exclude_variable = 'Cluster'

# Get the names of all other variables dynamically
remaining_features = [col for col in data.columns if col != exclude_variable]

# Step 2: Further divide into numerical and categorical
numerical_features = data[remaining_features].select_dtypes(include=['int64', 'float64']).columns.tolist()



print("Numerical Features:", numerical_features)


# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # Set sparse=False to return a dense array

categorical_features = data['Cluster']
# Convert Series to NumPy array and reshape
categorical_features = categorical_features.to_numpy().reshape(-1, 1)

print(f"Vector shape: {categorical_features.shape}")
# Fit and transform the data
category_encoded = encoder.fit_transform(categorical_features)


# Create a DataFrame with the encoded features
encoded_columns = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Cluster']))
print(f"Vector shape: {encoded_columns.shape}")

# Combine the encoded columns with the original DataFrame (excluding the original column)
print(f"Vector shape: {data.shape}")


data_reset = data.drop(columns=['Cluster']).reset_index(drop=True)
encoded_columns_reset = encoded_columns.reset_index(drop=True)

# Concatenate the DataFrames
data_encoded = pd.concat([data_reset, encoded_columns_reset], axis=1)

# Check the shape of the concatenated DataFrame
print(f"Vector shape after concatenation: {data_encoded.shape}")

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
output_file = folder_path_ML + "09logistic_predictions test2024_baseadj.csv"
results_test.to_csv(output_file, index=False)
print(f"Predictions exported to {output_file}")

##################################################
# View coefficients
coefficients = model.coef_[0]  # Coefficients for each feature
intercept = model.intercept_[0]  # Intercept

# Combine with feature names
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

print("Model Coefficients:")
print(coef_df)
print(f"Intercept: {intercept}")

# Approximate standard errors
# The inverse of the Hessian matrix gives the covariance matrix
X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
cov_matrix = np.linalg.inv(np.dot(X_train_with_intercept.T, X_train_with_intercept))
standard_errors = np.sqrt(np.diag(cov_matrix))

# Step 5: Calculate 95% confidence intervals
z = norm.ppf(0.975)  # 1.96 for 95% confidence
confidence_intervals = [
    (coef - z * se, coef + z * se)
    for coef, se in zip(coefficients, standard_errors[1:])  # Skip intercept
]

# Combine results into a DataFrame
summary_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Standard Error': standard_errors[1:],  # Skip intercept
    'Confidence Interval Lower': [ci[0] for ci in confidence_intervals],
    'Confidence Interval Upper': [ci[1] for ci in confidence_intervals]
})

print("\nSummary of Logistic Regression:")
print(summary_df)


#outpot to csv

output_file_coef= folder_path_ML + "10logistic_2024_baseadj_modelCoef.csv"
summary_df.to_csv(output_file_coef, index=False)
print(f"Coefficients exported to {output_file_log_full}")


#############Apply to full dataset 78 sports
# Predict on the entire dataset


data1 = merged_data1.drop(columns=['Label', 'Unnamed: 0','ID','Olympic_outcome'])  # Features--Cluster
target1 = merged_data1['Olympic_outcome']  # Target variable


print(f"Vector shape: {data1.shape}")

# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # Set sparse=False to return a dense array

categorical_features1 = data1['Cluster']
# Convert Series to NumPy array and reshape
categorical_features1 = categorical_features1.to_numpy().reshape(-1, 1)

print(f"Vector shape: {categorical_features1.shape}")
# Fit and transform the data
category_encoded1 = encoder.fit_transform(categorical_features1)

# Create a DataFrame with the encoded features
encoded_columns1 = pd.DataFrame(category_encoded1, columns=encoder.get_feature_names_out(['Cluster']))
print(f"Vector shape: {encoded_columns1.shape}")

# Combine the encoded columns with the original DataFrame (excluding the original column)
print(f"Vector shape: {data1.shape}")


data_reset1 = data1.drop(columns=['Cluster']).reset_index(drop=True)
encoded_columns_reset1 = encoded_columns1.reset_index(drop=True)

# Concatenate the DataFrames
data_encoded1 = pd.concat([data_reset1, encoded_columns_reset1], axis=1)
print(f"Vector shape: {data_encoded1.shape}")
X1=data_encoded1

y_pred1 = model.predict(X1)
y_pred_prob1 = model.predict_proba(X1)[:, 1]  # Probability of the positive class


# Check dimensions
print(f"Vector shape: {y_pred1.shape}")

print(f"Vector shape: {y_pred_prob1.shape}")

print(f"Vector shape: {label.shape}")


# Combine IDs, true labels, predictions, and probabilities into a DataFrame
results_full1 = pd.DataFrame({
    'Label':merged_data1['Label'],
    'True Label': y,
    'Predicted Label': y_pred1,
    'Probability Positive Class': y_pred_prob1
})


print(results_full1)
# Export to CSV
output_file_log_full = folder_path_ML + "08logistic_full_predictions2024_baseadj2.csv"
results_full1.to_csv(output_file_log_full, index=False)
print(f"Predictions exported to {output_file_log_full}")



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results

print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2023 high school/01 problem A/02 Data/D04 test data1.csv'

# Read the CSV file into a DataFrame
df_test = pd.read_csv(file_path)
# Print all column names
print(df_test.columns.tolist())


# Separate features (X) and target variable (y)
X_test1 = df_test.drop(['id', 'Invasive Species'], axis=1)
y_test1 = df_test['Invasive Species']

y_pred1 = model.predict(X_test1)



##############################################
##############################################
###Test on a small sample 16 sports
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score


 #we add the Baseball to test data: y_test 
X1=data_encoded1
y1=target1


labels = merged_data1['Label']  # Labels for the samples

# Step 3: Split the data

# Split data while retaining labels
X_train2, X_test2, y_train2, y_test2, labels_train2, labels_test2 = train_test_split(
    X1, y1,  labels, test_size=0.2, random_state=42
)

#X_train2, X_test2, y_train2, y_test2 = train_test_split(X1, y1, test_size=0.2, random_state=42)

y_pred2 = model.predict(X_test2)  # X_test1 should match the dataset for y_test1

y_pred_prob2 = model.predict_proba(X_test2)[:, 1]  # Probability of the positive class

# Evaluate the model
print(f"Length of y_test1: {len(y_test2)}")
print(f"Length of y_pred1: {len(y_pred2)}")

accuracy = accuracy_score(y_test2, y_pred2)
CM = confusion_matrix(y_test2, y_pred2)
class_report = classification_report(y_test2, y_pred2)

# Print the results
print(f'Sensitivity: {sensitivity}')

print(f'Accuracy: {accuracy}')


print(f'Confusion Matrix:\n{CM}')
print(f'Classification Report:\n{class_report}')

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

# multiple evaluation metrics
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
recall = sensitivity
precision = TP / (TP + FP)
f1_score = 2 * TP / (2 * TP + FP + FN)

# add negative and positive predictive value
pos_pred_value = TP / (TP + FP)
neg_pred_value = TN / (TN + FN)

print("balanced_accuracy", balanced_accuracy)
print("sensitivity", sensitivity)
print("specificity", specificity)

# Combine IDs, true labels, predictions, and probabilities into a DataFrame
results_test16 = pd.DataFrame({
    'Label':labels_test2.values,
    'True Y':  y_test2.values,
    'Predicted Y': y_pred2,
    'Probability Positive Class': y_pred_prob2
})

# Display results
print(results_test16)



#outpot to csv

file_results_test16= folder_path_ML + "11logistic_2024_baseadj_test16.csv"
results_test16.to_csv(file_results_test16, index=False)
print(f"Coefficients exported to {file_results_test16}")



#########Special test only on baseball ID=10

####################################################################
####################################################################
##### Sensitivity anlaysis
#Change the values of individual features (e.g., increase/decrease by 10%) while keeping others constant and observe how predictions change.

# Example: If X is a NumPy array, assign column names:
X_test2feature= pd.DataFrame(X_test2, columns=['1 Fatality Rate', '2 CO2 footprint',
       '3 Number of countries playing it', '4 Ratio of Players of each gender',
       '5 Number of Players', '6 Cost of Equipment for Players',
       '7 Presence in University Sports', '8 Injury Rate',
       '9 Social Media Presence', '10 Gender Pay Gap',
       '11 Ratio of each gender in coaching positions', '12 Doping Statistics',
       '13 Ease of developing new strategies',
       '14 Accessibility across different streaming platforms',
       '15 Movies/documentaries', 'Score', 'Cluster_0', 'Cluster_1',
       'Cluster_2', 'Cluster_3']) 


sample = X_test2.iloc[0].copy()  # Choose a test sample
results_sensitivity_changeProb = []
results_sensitivity=[]
original_probability = y_pred_prob2 # Probability for class 1


for feature in X_test2.columns:
    modified_sample = sample.copy()
    modified_sample[feature] *= 1.1  # Increase the feature value by 10%
    modified_probability = model.predict_proba([modified_sample])[0][1]
    
    # Calculate Change in Probability
    change_in_probability = modified_probability - original_probability
    
    # Store results
    results_sensitivity_changeProb.append({
        'Feature': feature,
        'Original Probability': original_probability,
        'Modified Probability': modified_probability,
        'Change in Probability': change_in_probability
    })


results_sensitivity_changeProb1 = pd.DataFrame(results_sensitivity_changeProb)
print(results_sensitivity_changeProb)






for feature in X_test2.columns:
    modified_sample = sample.copy()
    modified_sample[feature] *= 1.1  # Increase by 10%
    probability = model.predict_proba([modified_sample])[0][1]  # Probability for class 1
    results_sensitivity.append({'Feature': feature, 'Modified Probability': probability})

sensitivity_results = pd.DataFrame(results_sensitivity)
print(sensitivity_results)


# Plot results
plt.figure(figsize=(8, 6))
plt.bar(sensitivity_results['Feature'], sensitivity_results['Modified Probability'], color='skyblue')
plt.title('Sensitivity Analysis')
plt.xlabel('Feature')
plt.ylabel('Modified Probability')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#save to csv


file_sensitivity_12= folder_path_ML + "12_sensitivity.csv"
sensitivity_results.to_csv(file_sensitivity_12, index=False)
print(f"file_sensitivity_12 with Noise exported to {file_sensitivity_12}")


#############################################################################################
#############################################################################################
## Evaluate Model Robustness
##Check the impact of adding noise to inputs:
##Add random noise to features 
#Predict outcomes and calculate changes in probabilities or metrics like accuracy.

# Example of adding noise
noise = np.random.normal(0, 0.1, X_test2.shape)  # Add noise with mean 0 and std 0.1
X_test2_noisy = X_test2 + noise
y_pred2_noisy = model.predict(X_test2_noisy)

accuracy_noisy = accuracy_score(y_test2, y_pred2_noisy)

# Calculate baseline accuracy for comparison
y_pred2_original = model.predict(X_test2)
accuracy2_original = accuracy_score(y_test2, y_pred2_original)

# Store results
results_noise = pd.DataFrame({
    'Metric': ['Accuracy'],
    'Original': [accuracy2_original],
    'Noisy': [accuracy_noisy],
    'Change': [accuracy_noisy - accuracy2_original]
})
print(results_noise)

#Add Multiple Noise Levels for Sensitivity Analysis
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]  # Different noise standard deviations
accuracies_noisy = []

for noise_std in noise_levels:
    X_test2_noisy = X_test2 + np.random.normal(0, noise_std, X_test.shape)
    y_pred2_noisy = model.predict(X_test2_noisy)
    accuracy2_noisy = accuracy_score(y_test2, y_pred2_noisy)
    accuracies_noisy.append(accuracy2_noisy)

# Create a results DataFrame
results2_noise2 = pd.DataFrame({
    'Noise Level (σ)': noise_levels,
    'Accuracy with Noise': accuracies_noisy
})
print(results2_noise2)

#outpot to csv

file_robust_13= folder_path_ML + "13robustn_noise_sensitivity.csv"
results2_noise2.to_csv(file_robust_13, index=False)
print(f"Accuracy with Noise exported to {file_robust_13}")