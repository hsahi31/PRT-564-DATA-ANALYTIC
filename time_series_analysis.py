import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt


# # Load the preprocessed dataset
# data = pd.read_csv('retractions_cleaned.csv')

# # Convert RetractionDate to datetime
# data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

# # Filter out invalid dates
# data = data[data['RetractionDate'].notnull()]

# # Extract the year from the RetractionDate
# data['Year'] = data['RetractionDate'].dt.year

# # Group by year and count the number of retractions per year
# annual_retractions = data.groupby('Year').size().reset_index(name='Retractions')

# # Ensure data is sorted by year
# annual_retractions = annual_retractions.sort_values(by='Year').set_index('Year')
# print(annual_retractions.head())

# # Fit the ARIMA model
# model = sm.tsa.arima.ARIMA(annual_retractions['Retractions'], order=(5, 1, 0))
# model_fit = model.fit()

# # Summary of the model
# print(model_fit.summary())

# # Plot the actual and fitted values
# plt.figure(figsize=(10, 6))
# plt.plot(annual_retractions, label='Actual')
# plt.plot(annual_retractions.index, model_fit.fittedvalues, color='red', label='Fitted')
# plt.title('ARIMA Model - Actual vs Fitted')
# plt.xlabel('Year')
# plt.ylabel('Number of Retractions')
# plt.legend()
# plt.show()

# # Forecast future retractions
# forecast, stderr, conf_int = model_fit.forecast(steps=5)

# # Create a DataFrame for the forecast
# forecast_years = [annual_retractions.index[-1] + i for i in range(1, 6)]
# forecast_df = pd.DataFrame({'Year': forecast_years, 'Forecast': forecast})
# forecast_df.set_index('Year', inplace=True)
# print(forecast_df)

# # Plot the forecast
# plt.figure(figsize=(10, 6))
# plt.plot(annual_retractions, label='Actual')
# plt.plot(forecast_df, color='green', label='Forecast')
# plt.fill_between(forecast_years, conf_int[:, 0], conf_int[:, 1], color='k', alpha=0.1)
# plt.title('ARIMA Model - Forecast of Retractions')
# plt.xlabel('Year')
# plt.ylabel('Number of Retractions')
# plt.legend()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
annual_retractions = annual_retractions.sort_values(by='Year').set_index('Year')
print(annual_retractions.head())

# Fit the ARIMA model
model = sm.tsa.ARIMA(annual_retractions['Retractions'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Plot the actual and fitted values
plt.figure(figsize=(10, 6))
plt.plot(annual_retractions, label='Actual')
plt.plot(annual_retractions.index, model_fit.fittedvalues, color='red', label='Fitted')
plt.title('ARIMA Model - Actual vs Fitted')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.show()

# Forecast future retractions
forecast = model_fit.get_forecast(steps=5)
forecast_df = forecast.conf_int()
forecast_df['Forecast'] = forecast.predicted_mean
forecast_df.index = [annual_retractions.index[-1] + i for i in range(1, 6)]

# Create a DataFrame for the forecast
print(forecast_df)

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(annual_retractions, label='Actual')
plt.plot(forecast_df['Forecast'], color='green', label='Forecast')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='k', alpha=0.1)
plt.title('ARIMA Model - Forecast of Retractions')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.legend()
plt.show()
