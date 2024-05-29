import pandas as pd

# Load the preprocessed dataset
data = pd.read_csv('retractions_cleaned.csv', index_col=False, dtype='unicode')

# Display descriptive statistics for numerical columns
numerical_desc = data.describe()

# Display descriptive statistics for categorical columns
categorical_desc = data.describe(include=['O'])

# Print numerical and categorical statistics
print("Numerical Descriptive Statistics:")
print(numerical_desc)
print("\nCategorical Descriptive Statistics:")
print(categorical_desc)

