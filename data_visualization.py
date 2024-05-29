import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
data = pd.read_csv('retractions_cleaned.csv', index_col=False, dtype='unicode')


# # Convert RetractionDate to datetime
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# # Filter out invalid dates
data = data[data['RetractionDate'].notnull()]

# # Time Series Analysis: Number of retractions over time
time_series_data = data['RetractionDate'].dt.year.value_counts().sort_index()
plt.figure(figsize=(10, 6))
time_series_data.plot(kind='line', marker='o')
plt.title('Number of Retractions Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.grid(True)
plt.show()

# # Bar Charts and Histograms: Distribution by Subject, Institution, Country, and Reasons for Retraction
# # Top 10 Authors
plt.figure(figsize=(10, 6))
data['Authors'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Authors by Retraction Count')
plt.xlabel('Authors')
plt.ylabel('Number of Retractions')
plt.show()

# # Top 10 Institutions
plt.figure(figsize=(10, 6))
data['Institutions'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Institutions by Retraction Count')
plt.xlabel('Institution')
plt.ylabel('Number of Retractions')
plt.show()

# # Top 10 Countries
plt.figure(figsize=(10, 6))
data['Countries'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Countries by Retraction Count')
plt.xlabel('Country')
plt.ylabel('Number of Retractions')
plt.show()

# # Reasons for Retraction
plt.figure(figsize=(10, 6))
data['Reason'].value_counts().plot(kind='bar')
plt.title('Reasons for Retraction')
plt.xlabel('Reason Category')
plt.ylabel('Number of Retractions')
plt.show()

