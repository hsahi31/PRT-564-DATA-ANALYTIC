import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import seaborn as sns

# Load the preprocessed dataset
data = pd.read_csv('retractions_cleaned.csv', index_col=False, dtype='unicode')
# Calculate Z-scores
data['RetractionLag'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days
data = data[data['RetractionLag'].notnull()]  # Filter out null values

data['Z_Score'] = (data['RetractionLag'] - data['RetractionLag'].mean()) / data['RetractionLag'].std()

# Identify outliers
outliers_z = data[(data['Z_Score'] > 3) | (data['Z_Score'] < -3)]
print("Outliers detected using Z-score method:")
print(outliers_z[['Record ID', 'RetractionLag', 'Z_Score']])

# Calculate IQR
Q1 = data['RetractionLag'].quantile(0.25)
Q3 = data['RetractionLag'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers_iqr = data[(data['RetractionLag'] < (Q1 - 1.5 * IQR)) | (data['RetractionLag'] > (Q3 + 1.5 * IQR))]
print("Outliers detected using IQR method:")
print(outliers_iqr[['Record ID', 'RetractionLag']])

# Boxplot for RetractionLag
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['RetractionLag'])
plt.title('Boxplot of RetractionLag')
plt.xlabel('RetractionLag (days)')
plt.show()