import pandas as pd

# Load the dataset
data = pd.read_csv('retractions35215(3).csv')

# Display the first few rows of the dataset
print(data.head())

# --- Data Preprocessing ---# 
# Check for missing values
missing_values = data.isnull().sum()

# Print missing values
print(missing_values)

# Example: Fill missing values in 'RetractionDate' and 'OriginalPaperDate' with a placeholder
data['RetractionDate'].fillna('1900-03-10', inplace=True)
data['OriginalPaperDate'].fillna('1900-11-06', inplace=True)

# Example: Drop rows with missing 'Reason' values as they are crucial for analysis
data.dropna(subset=['Reason'], inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Convert date columns to datetime format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'],dayfirst=True,errors='coerce')
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'],dayfirst=True,errors='coerce')
                                           
#Data Transformation

# Split semi-colon delineated lists into separate rows
data = data.assign(Authors=data['Author'].str.split(';')).explode('Authors')
data = data.assign(Institutions=data['Institution'].str.split(';')).explode('Institutions')
data = data.assign(Countries=data['Country'].str.split(';')).explode('Countries')

# Define a function to categorize retraction reasons
def categorize_reasons(reason):
    if 'error' in reason.lower():
        return 'Error'
    elif 'misconduct' in reason.lower() or 'fraud' in reason.lower():
        return 'Misconduct'
    elif 'plagiarism' in reason.lower():
        return 'Plagiarism'
    else:
        return 'Other'

# Apply the function to the 'Reason' column
data['ReasonCategory'] = data['Reason'].apply(categorize_reasons)

# ---Featured Engineering --- #
# Calculate the time between original publication and retraction
data['RetractionLag'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days

# Count retractions per institution or country
institution_retraction_counts = data['Institutions'].value_counts().reset_index()
institution_retraction_counts.columns = ['Institution', 'RetractionCount']
country_retraction_counts = data['Countries'].value_counts().reset_index()
country_retraction_counts.columns = ['Country', 'RetractionCount']

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Subject', 'ArticleType', 'ReasonCategory'], drop_first=True)

# --- Data Exploration & Visualization Prep --- #

# Save the cleaned and preprocessed data to a new CSV file
data.to_csv('retractions_cleaned.csv', index=False)

# Display the first few rows of the preprocessed dataset
print(data.head())

# --- Descriptive Statistic ---#
# Load the preprocessed dataset
data = pd.read_csv('retractions_preprocessed.csv')

# Display descriptive statistics for numerical columns
numerical_desc = data.describe()

# Display descriptive statistics for categorical columns
categorical_desc = data.describe(include=['O'])

# Additional descriptive statistics for numerical columns
additional_numerical_stats = {
    'Skewness': data.skew(),
    'Kurtosis': data.kurtosis()
}

# Concatenate additional statistics to numerical description
numerical_desc = pd.concat([numerical_desc, pd.DataFrame(additional_numerical_stats)])

# Prepare summary for unique counts and top frequencies for categorical data
categorical_summary = {
    'Unique': data.select_dtypes(include=['object']).nunique(),
    'Top': data.select_dtypes(include=['object']).mode().iloc[0],
    'Freq': data.select_dtypes(include=['object']).apply(lambda x: x.value_counts().iloc[0])
}

categorical_summary_df = pd.DataFrame(categorical_summary)

import ace as tools; tools.display_dataframe_to_user(name="Categorical Descriptive Statistics", dataframe=categorical_summary_df)

# Display the descriptive statistics
numerical_desc, categorical_summary_df

