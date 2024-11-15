import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('./data/machine_temperature_system_failure.csv')

# Anomaly periods
anomaly_periods = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]

# Convert anomaly periods to indices
df['timestamp'] = pd.to_datetime(df['timestamp'])
anomaly_indices = []
for start, end in anomaly_periods:
    start_idx = df[df['timestamp'] == pd.to_datetime(start)].index[0]
    end_idx = df[df['timestamp'] == pd.to_datetime(end)].index[0]
    anomaly_indices.extend(range(start_idx, end_idx + 1))

# Drop the timestamp column since we only need the 'value' column
data = df['value'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Assuming the dataset is time-ordered
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

SEQ_LENGTH = 30  # Define sequence length

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# Remove sequences containing anomalies from training data
def remove_anomalous_sequences(data, anomaly_indices, seq_length):
    clean_sequences = []
    for i in range(len(data) - seq_length):
        if not any(point in range(i, i + seq_length) for point in anomaly_indices):
            clean_sequences.append(data[i:i + seq_length])
    return np.array(clean_sequences)

X_train_clean = remove_anomalous_sequences(train_data, anomaly_indices, SEQ_LENGTH)
X_test = create_sequences(test_data, SEQ_LENGTH)

print("1#", X_test)

# Add an extra dimension to indicate the single feature
X_train_clean = np.expand_dims(X_train_clean, -1)
X_test = np.expand_dims(X_test, -1)

# Check and convert data type to float32
X_train_clean = X_train_clean.astype('float32')
X_test = X_test.astype('float32')

print("2#", X_test)

# Build and train the model
# model = Sequential([
#     LSTM(128, activation='relu', input_shape=(SEQ_LENGTH, 1), return_sequences=False),
#     RepeatVector(SEQ_LENGTH),
#     LSTM(128, activation='relu', return_sequences=True),
#     TimeDistributed(Dense(1))
# ])

# model.compile(optimizer='adam', loss='mse')
# history = model.fit(X_train_clean, X_train_clean, epochs=50, batch_size=64, validation_split=0.2)

# model.save('./models/lstm_encoder_new.keras')

model = load_model('./models/lstm_encoder_new.keras')

# Predict the sequences using the trained model
X_train_pred = model.predict(X_train_clean)
X_test_pred = model.predict(X_test)

# Reshape predictions to match the ground truth shape
X_train_pred = X_train_pred.reshape(X_train_clean.shape)
X_test_pred = X_test_pred.reshape(X_test.shape)

# Calculate the mean squared error for each sequence
train_reconstruction_error = np.mean(np.abs(X_train_pred - X_train_clean), axis=(1, 2))
test_reconstruction_error = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))

# Set the threshold as the 95th percentile of the training reconstruction error
threshold = np.percentile(train_reconstruction_error, 98)
print(f"Reconstruction error threshold: {threshold}")

train_anomalies = train_reconstruction_error > threshold
test_anomalies = test_reconstruction_error > threshold

print(f"Number of anomalies in training set: {np.sum(train_anomalies)}")
print(f"Number of anomalies in test set: {np.sum(test_anomalies)}")

# Plot the reconstruction error
# plt.figure(figsize=(10, 6))
# plt.plot(train_reconstruction_error, label='Train Reconstruction Error')
# plt.plot(np.arange(len(train_reconstruction_error), len(train_reconstruction_error) + len(test_reconstruction_error)), test_reconstruction_error, label='Test Reconstruction Error')
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
# plt.legend()
# plt.xlabel('Sequence')
# plt.ylabel('Reconstruction Error')
# plt.title('Reconstruction Error with Threshold')
# plt.show()

anomalous_indices = np.where(test_reconstruction_error > threshold)[0]

# Map these indices back to the original timestamps
anomalous_timestamps = df.iloc[train_size + SEQ_LENGTH + anomalous_indices]['timestamp']

# Print the timestamps with high reconstruction error
print("Timestamps with high reconstruction error:")
print(anomalous_timestamps)

# Convert ground truth anomaly periods to datetime
anomaly_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_periods]

# Filter anomaly periods to only those within the test data range
test_start_time = df['timestamp'].iloc[train_size + SEQ_LENGTH]
test_end_time = df['timestamp'].iloc[-1]

test_anomaly_periods = [(start, end) for start, end in anomaly_periods if start >= test_start_time and end <= test_end_time]

# print("Number of Actual anomalies", test_anomaly_periods.sum())

def is_in_anomaly_period(timestamp, periods):
    for start, end in periods:
        if start <= timestamp <= end:
            return True
    return False

# Create a DataFrame to compare predicted anomalies with ground truth
comparison = pd.DataFrame({
    'Timestamp': anomalous_timestamps,
    'Predicted Anomaly': anomalous_timestamps.apply(lambda x: is_in_anomaly_period(x, anomaly_periods))
})

# print("\nComparison of predicted anomalies with ground truth:")
# print(comparison)

# Calculate false negatives
detected_anomalies = len(anomalous_timestamps)
true_positives = comparison['Predicted Anomaly'].sum()

# Calculate the number of actual anomalies that were not detected (False Negatives)
false_negatives = 0
for start, end in test_anomaly_periods:
    actual_anomaly_range = pd.date_range(start=start, end=end, freq='5T')  # Assuming 5-minute intervals
    false_negatives += len([timestamp for timestamp in actual_anomaly_range if timestamp not in anomalous_timestamps.values])

# Calculate the number of false positives
false_positives = len(comparison) - true_positives

# Calculate precision, recall, and F1 score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Output the results
print(f"true_positives: {true_positives}")
print(f"false_positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Generate timestamps for test data, starting after training set
test_timestamps = df['timestamp'].iloc[train_size + SEQ_LENGTH:].reset_index(drop=True)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Ground truth labels: 1 if in anomaly period, 0 otherwise
ground_truth_labels = test_timestamps.apply(lambda x: int(is_in_anomaly_period(x, test_anomaly_periods)))

# Predicted labels: 1 if detected as anomaly, 0 otherwise
predicted_labels = [1 if error > threshold else 0 for error in test_reconstruction_error]

# Generate confusion matrix
conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

# Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for Anomaly Detection')
# plt.show()


predicted_labels_series = pd.Series(predicted_labels, index=test_timestamps.index)

# Create a boolean mask for the predicted anomalies
anomaly_mask = predicted_labels_series == 1

# Filter timestamps and values based on the anomaly mask
anomalies = test_timestamps[anomaly_mask]
anomaly_values = test_data[SEQ_LENGTH:][anomaly_mask.values]  # Use .values to ensure mask aligns

# Plot the time series and highlight anomalies
plt.figure(figsize=(12, 6))
plt.plot(test_timestamps, test_data[SEQ_LENGTH:], label='Normal', color='blue', alpha=0.5)
plt.scatter(anomalies, anomaly_values, color='red', label='Anomalies', marker='.')

# Set labels and title
plt.xlabel('Timestamp')
plt.ylabel('Temperature Value')
plt.title('Detected Anomalies in Machine Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Convert ground truth anomaly periods to datetime
# anomaly_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_periods]

# # Check if the identified anomalies fall within any of the ground truth periods
# def is_in_anomaly_period(timestamp, periods):
#     for start, end in periods:
#         if start <= timestamp <= end:
#             return True
#     return False

# # Create a DataFrame to compare predicted anomalies with ground truth
# comparison = pd.DataFrame({
#     'Timestamp': anomalous_timestamps,
#     'Predicted Anomaly': anomalous_timestamps.apply(lambda x: is_in_anomaly_period(x, anomaly_periods))
# })

# print("\nComparison of predicted anomalies with ground truth:")
# print(comparison)

# # Calculate performance metrics
# true_positives = comparison['Predicted Anomaly'].sum()
# false_positives = len(comparison) - true_positives
# false_negatives = len([1 for start, end in anomaly_periods for idx in range(len(df)) if start <= df['timestamp'].iloc[idx] <= end]) - true_positives

# precision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)
# f1_score = 2 * (precision * recall) / (precision + recall)

# print(f"true_positives: {true_positives}")
# print(f"false_positives: {false_positives}")
# print(f"False Negatives: {false_negatives}")

# print(f"\nPrecision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1_score:.2f}")