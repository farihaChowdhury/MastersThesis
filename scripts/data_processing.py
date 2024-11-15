import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
file_path = './data/sensor_original.csv'
df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'sensor_15'])

# Interpolate missing values
df.interpolate(method='time', inplace=True)

# Normalize the sensor readings
sensor_columns = df.columns[:-1]  # Exclude the 'machine_status' column
scaler = StandardScaler()
df[sensor_columns] = scaler.fit_transform(df[sensor_columns])

# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

# Parameters
sequence_length = 50  # Length of the sequence for LSTM

# Prepare the data
sensor_data = df[sensor_columns].values
sequences = create_sequences(sensor_data, sequence_length)

# Split the data into training and test sets
X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

# Define the LSTM-based autoencoder model
model = Sequential([
    LSTM(128, input_shape=(sequence_length, len(sensor_columns)), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    RepeatVector(sequence_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    TimeDistributed(Dense(len(sensor_columns)))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1, shuffle=False, callbacks=[early_stopping])

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Detect anomalies
X_test_pred = model.predict(X_test)
test_mse_loss = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))

# Determine threshold for anomalies
threshold = np.percentile(test_mse_loss, 95)  # 95th percentile

# Predict anomalies
test_score_df = pd.DataFrame(index=df.index[sequence_length + len(X_train):])
test_score_df['loss'] = test_mse_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']

# Plot anomalies
plt.figure(figsize=(15, 8))
plt.plot(test_score_df.index, test_score_df['loss'], label='Test Loss')
plt.plot(test_score_df.index, test_score_df['threshold'], label='Threshold', color='r')
plt.scatter(test_score_df[test_score_df['anomaly']].index, test_score_df[test_score_df['anomaly']]['loss'], color='red', label='Anomaly')
plt.title('Test Loss and Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('lstm_autoencoder.h5')
