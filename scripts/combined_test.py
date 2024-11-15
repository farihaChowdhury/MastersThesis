import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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

# Feature engineering
synthetic_data['rolling_mean'] = synthetic_data['value'].rolling(window=10).mean()
synthetic_data['rolling_std'] = synthetic_data['value'].rolling(window=10).std()
synthetic_data['value_diff'] = synthetic_data['value'].diff()

# Drop NaN values
synthetic_data = synthetic_data.dropna()

# Scale the data
scaler = StandardScaler()
features = ['timestamp_num', 'value', 'rolling_mean', 'rolling_std', 'value_diff']
synthetic_data[features] = scaler.fit_transform(synthetic_data[features])

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_samples': ['auto', 0.6, 0.8],
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [1, 2, 3, 4, 5]
}

# Create the model
model = IsolationForest(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(synthetic_data[features])

# Get the best model
best_model = grid_search.best_estimator_

# Detect anomalies with the optimized model
synthetic_data['optimized_anomaly'] = best_model.predict(synthetic_data[features])
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
