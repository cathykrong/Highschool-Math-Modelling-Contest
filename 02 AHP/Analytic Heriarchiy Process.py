# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:31:04 2024

@author: Personal
"""

import pandas as pd
import numpy as np

import ahpy as ahpy
## swim
# Creating the matrix as a DataFrame
swim = {				
('Injury', 'Cost'): 4, ('Injury', 'Gender'): 3.5, ('Injury', 'Pop'): 3, ('Injury', 'Player'): 3.5,				
('Cost', 'Gender'): 1/2.5, ('Cost', 'Pop'): 1/3,				
('Cost', 'Player'): 1/2.8,				
('Gender', 'Pop'): 2.8, ('Gender', 'Player'): 2.5,				
('Pop', 'Player'): 3				
}				

swim_ahp = ahpy.Compare(name='Swimming', comparisons=swim, precision=3, random_index='saaty')

print(swim_ahp.target_weights)

print(swim_ahp.consistency_ratio)


#########baseball
# Baseball - significantly simplified comparisons																									
baseball = {																									
('Injury', 'Cost'): 3, ('Injury', 'Gender'): 1.5, ('Injury', 'Pop'): 2.5, ('Injury', 'Player'): 2,																									
('Cost', 'Gender'): 1/2, ('Cost', 'Pop'): 1/2.5, ('Cost', 'Player'): 1/2,																									
('Gender', 'Pop'): 2, ('Gender', 'Player'): 1,																									
('Pop', 'Player'): 2																									
}																									

baseball_ahp = ahpy.Compare(name='baseball', comparisons=baseball, precision=3, random_index='saaty')

print(baseball_ahp.target_weights)

print(baseball_ahp.consistency_ratio)

#### Breaking

# Breaking																									
breaking = {																									
('Injury', 'Cost'): 4, ('Injury', 'Gender'): 3, ('Injury', 'Pop'): 3.5, ('Injury', 'Player'): 3,																									
('Cost', 'Gender'): 1/2.5, ('Cost', 'Pop'): 1/4, ('Cost', 'Player'): 1/3,																									
('Gender', 'Pop'): 2.5, ('Gender', 'Player'): 2.3,																									
('Pop', 'Player'): 3																									
}																									

breaking_ahp = ahpy.Compare(name='breaking', comparisons=breaking, precision=3, random_index='saaty')

print(breaking_ahp.target_weights)

print(breaking_ahp.consistency_ratio)

pickleball = {				
('Injury', 'Cost'): 4, ('Injury', 'Gender'): 3, ('Injury', 'Pop'): 3.5, ('Injury', 'Player'): 3,				
('Cost', 'Gender'): 1/2.5, ('Cost', 'Pop'): 1/4, ('Cost', 'Player'): 1/3,				
('Gender', 'Pop'): 2.5, ('Gender', 'Player'): 2.3,				
('Pop', 'Player'): 3				
}				
pickleball_ahp = ahpy.Compare(name='Pickleball', comparisons=pickleball, precision=3, random_index='saaty')				
print("Pickleball Weights:", pickleball_ahp.target_weights)				
print("Pickleball Consistency Ratio:", pickleball_ahp.consistency_ratio)				
				

### output;

# Retrieve target weights as dictionaries for each sport
breaking_weights = breaking_ahp.target_weights
swim_weights = swim_ahp.target_weights
baseball_weights = baseball_ahp.target_weights

# Create DataFrames for each set of weights
breaking_df = pd.DataFrame(list(breaking_weights.items()), columns=["Criteria", "Breaking Weight"])
swim_df = pd.DataFrame(list(swim_weights.items()), columns=["Criteria", "Swim Weight"])
baseball_df = pd.DataFrame(list(baseball_weights.items()), columns=["Criteria", "Baseball Weight"])

# Merge all three DataFrames on the "Criteria" column
combined_df = pd.merge(breaking_df, swim_df, on="Criteria")
combined_df = pd.merge(combined_df, baseball_df, on="Criteria")

# Display the combined dataset
print(combined_df)



# Retrieve consistency ratios from each comparison object
breaking_cr = breaking_ahp.consistency_ratio
swim_cr = swim_ahp.consistency_ratio
baseball_cr = baseball_ahp.consistency_ratio

# Create a DataFrame to hold the consistency ratios
consistency_ratios_df = pd.DataFrame({
    "Sport": ["Breaking", "Swim", "Baseball"],
    "Consistency Ratio": [breaking_cr, swim_cr, baseball_cr]
})

# Display the DataFrame
print(consistency_ratios_df)


####Entropy method
import pandas as pd
import numpy as np

# Step 1: Define the decision matrix
data = {
    "Sport": ["Baseball", "Breaking Dance", "Swimming"],
    "Risk of Injury": [5, 7, 4],
    "Avg Cost of Equipment": [300, 150, 200],
    "Gender Equity": [60, 80, 70],
    "Popularity on Social Media": [70, 85, 90],
    "Player Counts": [150e6, 60e6, 400e6]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df.set_index("Sport", inplace=True)

# Step 2: Normalize the matrix
normalized_df = df / df.sum(axis=0)

# Step 3: Calculate the entropy for each criterion
# Constant for entropy calculation
k = 1 / np.log(len(df))

# Calculate entropy for each criterion
entropy = -k * (normalized_df * np.log(normalized_df + 1e-9)).sum(axis=0)  # Adding small value to avoid log(0)

# Step 4: Calculate the degree of diversification (1 - entropy)
diversification = 1 - entropy

# Step 5: Calculate the weights for each criterion
weights = diversification / diversification.sum()

# Display the results
print("Normalized Decision Matrix:\n", normalized_df)
print("\nEntropy for each criterion:\n", entropy)
print("\nDiversification (1 - Entropy) for each criterion:\n", diversification)
print("\nWeights for each criterion:\n", weights)



### add pickleball

import pandas as pd
import numpy as np

# Define the decision matrix with the added sport, Pickleball, and 5 criteria


data = {
    "Sport": ["swimming", "breaking", "baseball", "pickleball"],
    "Risk of Injury": [0.80040, 0.00200, 0.20160, 1.00000],
    "Avg Cost of Equipment": [0.33467, 1.00000, 0.00200, 0.66733],
    "Gender Equity": [0.66733, 0.22378, 0.00200, 1.00000],
    "Popularity on Social Media": [0.91683, 0.00200, 1.00000, 0.33467],
    "Player Counts": [0.46140, 0.00200, 1.00000, 0.55644]
}
# Convert to DataFrame
decision_matrix = pd.DataFrame(data)
decision_matrix.set_index("Sport", inplace=True)

# Step 2: Normalize the matrix
normalized_df = decision_matrix / decision_matrix.sum(axis=0)

# Step 3: Calculate the entropy for each criterion
# Constant for entropy calculation
k = 1 / np.log(len(decision_matrix))

# Calculate entropy for each criterion
entropy = -k * (normalized_df * np.log(normalized_df + 1e-9)).sum(axis=0)  # Adding small value to avoid log(0)

# Step 4: Calculate the degree of diversification (1 - entropy)
diversification = 1 - entropy

# Step 5: Calculate the weights for each criterion
weights = diversification / diversification.sum()

# Display the results
results = pd.DataFrame({
#    "Entropy": entropy,
#    "Diversification (1 - Entropy)": diversification,
    "Weights": weights
})

#import ace_tools as tools; tools.display_dataframe_to_user(name="Entropy Method Analysis Results and Weights", dataframe=results)

print(weights)

