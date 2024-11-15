import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

file_path = './data/machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
X = data[['value']]

X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

kmeans = KMeans(n_clusters=2, random_state=42)  
kmeans.fit(X_train)

distances = kmeans.transform(X_test)
closest_cluster_distances = np.min(distances, axis=1)

train_distances = kmeans.transform(X_train)
train_closest_distances = np.min(train_distances, axis=1)
threshold = np.percentile(train_closest_distances, 98)

y_pred = np.where(closest_cluster_distances > threshold, 1, 0)

test_timestamps = data['timestamp'].iloc[X_test.index]
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

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

plt.figure(figsize=(12, 6))

plt.plot(test_timestamps, X_test['value'], label='Normal', color='blue', alpha=0.5)

anomalies = X_test[y_pred == 1]
anomalies_timestamps = data['timestamp'].iloc[anomalies.index]

plt.scatter(anomalies_timestamps, anomalies['value'], color='red', label='Anomalies', marker='.')

plt.xlabel('Timestamp')
plt.ylabel('Temperature Value')
plt.title('Detected Anomalies in Machine Temperature using K-Means')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix for K-Means Anomaly Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
