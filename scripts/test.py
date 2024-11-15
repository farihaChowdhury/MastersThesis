import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

# Load the synthetic data
synthetic_data = pd.read_csv('./data/Synthetic_Dataset.csv')

# Define anomaly periods
anomaly_points = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]

# Convert timestamp to datetime
synthetic_data['timestamp'] = pd.to_datetime(synthetic_data['timestamp'])

# Convert datetime to numerical value (UNIX timestamp)
synthetic_data['timestamp_num'] = synthetic_data['timestamp'].astype('int64') // 10**9

# Manually tune the parameters for the Isolation Forest model
best_model = IsolationForest(n_estimators=200, max_samples=0.8, contamination=0.05, random_state=42)
best_model.fit(synthetic_data[['timestamp_num', 'value']])

# Detect anomalies with the tuned model
synthetic_data['optimized_anomaly'] = best_model.predict(synthetic_data[['timestamp_num', 'value']])
synthetic_data['optimized_anomaly'] = synthetic_data['optimized_anomaly'].map({1: 0, -1: 1})  # Convert to binary 0 for normal, 1 for anomaly

# Mark known anomaly periods
synthetic_data['known_anomaly'] = 0
for start, end in anomaly_points:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    synthetic_data.loc[(synthetic_data['timestamp'] >= start) & (synthetic_data['timestamp'] <= end), 'known_anomaly'] = 1

# Evaluate the optimized model by comparing with known anomalies
# Confusion matrix
conf_matrix_optimized = confusion_matrix(synthetic_data['known_anomaly'], synthetic_data['optimized_anomaly'])

# Display the confusion matrix
conf_matrix_optimized_df = pd.DataFrame(conf_matrix_optimized, index=['Normal', 'Known Anomaly'], columns=['Predicted Normal', 'Predicted Anomaly'])
print(conf_matrix_optimized_df)
