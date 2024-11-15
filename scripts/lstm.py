import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the dataset
file_path = './data/machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

# Convert the 'timestamp' column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Prepare the feature 'value'
values = data[['value']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)

# Define function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences for LSTM
SEQ_LENGTH = 50  # Number of previous steps to consider for predicting the next step
X, y = create_sequences(values_scaled, SEQ_LENGTH)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False)

# Make predictions
y_pred = model.predict(X_test)

# Inverse the scaling of predictions and true values
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate the reconstruction error
mse = np.mean(np.power(y_test_inv - y_pred_inv, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # Adjust threshold as needed
anomalies = mse > threshold

# Convert test set timestamps to make comparison with known anomaly periods
test_timestamps = data['timestamp'].iloc[split+SEQ_LENGTH:]

# Mark the ground truth anomalies based on known anomaly periods
y_test_labels = np.zeros_like(anomalies)
anomaly_periods = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]

for start, end in anomaly_periods:
    mask = (test_timestamps >= pd.to_datetime(start)) & (test_timestamps <= pd.to_datetime(end))
    y_test_labels[mask] = 1

# Calculate precision, recall, and f1 score
precision = precision_score(y_test_labels, anomalies)
recall = recall_score(y_test_labels, anomalies)
f1 = f1_score(y_test_labels, anomalies)

# Output the results
print(precision)
print(recall)
print(f1)
