import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.plotting import show, output_file

# Load the dataset
file_path = './data/machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

# Convert the 'timestamp' column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Drop the 'timestamp' column and prepare the features
X = data[['value']]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

# Apply Isolation Forest for anomaly detection (Unsupervised)
model = IsolationForest(n_estimators=300,contamination=0.01,   max_samples='auto')
model.fit(X_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert the prediction output (-1 for anomalies, 1 for normal) to binary (1 for anomalies, 0 for normal)
y_pred = np.where(y_pred == -1, 1, 0)

# Convert test set timestamps to make comparison with known anomaly periods
test_timestamps = data['timestamp'].iloc[X_test.index]

# Mark the ground truth anomalies based on known anomaly periods
y_test = np.zeros_like(y_pred)
anomaly_periods = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]

for start, end in anomaly_periods:
    mask = (test_timestamps >= pd.to_datetime(start)) & (test_timestamps <= pd.to_datetime(end))
    y_test[mask] = 1

# Calculate precision, recall, and f1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output the results
print(precision)
print(recall)
print(f1)

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Generate the confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title('Confusion Matrix for Isolation Forest Anomaly Detection')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')

# # Show the plot
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

# Plot the results with anomalies marked
plt.figure(figsize=(12, 6))

# Plot the normal data points
plt.plot(test_timestamps, X_test['value'], label='Normal', color='blue', alpha=0.5)

# Plot the anomalies
anomalies = X_test[y_pred == 1]
anomalies_timestamps = data['timestamp'].iloc[anomalies.index]

print(anomalies)
plt.scatter(anomalies_timestamps, anomalies['value'], color='red', label='Anomalies', marker='.')

# Set labels and title
plt.xlabel('Timestamp')
plt.ylabel('Temperature Value')
plt.title('Detected Anomalies in Machine Temperature')
plt.legend()

# Rotate the x-axis labels for better readability
# plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# anomalies = [[tsp, value] for tsp, value in zip(anomalies.index, anomalies['value'])]

# output_file('test.html')
# show(hv.render((hv.Curve(X_test['value'], label="Temperature") * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points"))\
#     .opts(opts.Curve(xlabel="Time", ylabel="Temperature", width=700, height=400,tools=['hover'],show_grid=True))))