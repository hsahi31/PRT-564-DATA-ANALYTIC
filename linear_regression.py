import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the preprocessed dataset
data = pd.read_csv('retractions_cleaned.csv')

# Convert RetractionDate to datetime
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# Filter out invalid dates
data = data[data['RetractionDate'].notnull()]

# Extract the year from the RetractionDate
data['Year'] = data['RetractionDate'].dt.year

# Group by year and count the number of retractions per year
annual_retractions = data.groupby('Year').size().reset_index(name='Retractions')

# Ensure data is sorted by year
annual_retractions = annual_retractions.sort_values(by='Year')
print(annual_retractions.head())

# Define features and target variable
X = annual_retractions[['Year']]
y = annual_retractions['Retractions']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Predict the retractions for the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')
# Plot the testing data
plt.scatter(X_test, y_test, color='green', label='Testing data')
# Plot the regression line
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.title('Linear Regression to Predict Future Retractions')
plt.legend()
plt.show()