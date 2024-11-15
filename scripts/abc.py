import matplotlib.pyplot as plt
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

# Convert 'timestamp' from object to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

anomaly_periods = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]


# Convert the anomaly periods to datetime as well
anomaly_periods = pd.DataFrame(anomaly_periods, columns=['start', 'end'])
anomaly_periods['start'] = pd.to_datetime(anomaly_periods['start'])
anomaly_periods['end'] = pd.to_datetime(anomaly_periods['end'])

plt.figure(figsize=(14, 6))

# Plot the normal data in blue
plt.plot(data['timestamp'], data['value'], color='blue', label='Normal', alpha=0.5)

# Highlight the anomalies as smaller red dots
for _, row in anomaly_periods.iterrows():
    anomaly_data = data[(data['timestamp'] >= row['start']) & (data['timestamp'] <= row['end'])]
    plt.plot(anomaly_data['timestamp'], anomaly_data['value'], 'ro', markersize=2, label='Anomalies' if _ == 0 else "")

# Adding labels and title to match the reference style
plt.title("Detected Anomalies in Machine Temperature")
plt.xlabel("Timestamp")
plt.ylabel("Temperature Value")
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()