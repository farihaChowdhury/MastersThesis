import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load the dataset
file_path = './data/sensor_original.csv'
df = pd.read_csv(file_path)


# print(df.info())
# print(df.head())



# Data types
print(df.dtypes)

df = df.drop(df.columns[0], axis=1)

df = df.drop(columns=['sensor_00', 'sensor_15','sensor_51'])

# Check for missing values
print(df.isnull().sum())

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

# Convert the last column to numeric using mapping
mapping = {'normal': 0, 'broken': 1, 'recovering': 2}
df[df.columns[-1]] = df[df.columns[-1]].map(mapping)

# Drop rows with missing values
df_cleaned = df.dropna()

df_cleaned.set_index(df_cleaned.columns[0], inplace=True)

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Select the column to forecast (assuming the first column here for simplicity)
time_series = df_normalized.iloc[:, 0]

# Split the data into training and testing sets
train_size = int(len(time_series) * 0.8)
train_data, test_data = time_series[:train_size], time_series[train_size:]

# Fit ARIMA model (p, d, q are hyperparameters to be determined)
p, d, q = 5, 1, 0  # Change these based on model selection criteria
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()

# In-sample prediction
train_preds = model_fit.predict(start=0, end=len(train_data)-1, typ='levels')

# Forecast
forecast_steps = len(test_data)
test_preds = model_fit.forecast(steps=forecast_steps)

# Evaluate the model
train_mse = mean_squared_error(train_data[d:], train_preds[d:])
test_mse = mean_squared_error(test_data, test_preds)
print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_data)), train_data, label='Train Data')
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Test Data', color='orange')
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_preds, label='Predictions', color='green')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.show()

# Save the model
joblib.dump(model_fit, '../models/saved_arima_model.pkl')
