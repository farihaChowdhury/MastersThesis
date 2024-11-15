import pandas as pd

# Load the synthetic dataset
synthetic_df = pd.read_csv('data/synthetic_machine_temperature_system_failure.csv')

# Convert the timestamp to datetime format
synthetic_df['timestamp'] = pd.to_datetime(synthetic_df['timestamp'])

# Overview of the dataset
total_entries = len(synthetic_df)
min_value = synthetic_df['value'].min()
max_value = synthetic_df['value'].max()

# Define the anomaly periods
anomaly_periods_synthetic = [
    ["2024-01-05 06:25:00", "2024-01-07 05:35:00"],
    ["2024-01-12 17:50:00", "2024-01-14 17:00:00"],
    ["2024-02-02 14:20:00", "2024-02-04 13:30:00"],
    ["2024-02-10 14:55:00", "2024-02-12 14:05:00"]
]

anomaly_periods_synthetic = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_periods_synthetic]
anomaly_mask = pd.Series([False] * total_entries)

for start, end in anomaly_periods_synthetic:
    anomaly_mask |= (synthetic_df['timestamp'] >= start) & (synthetic_df['timestamp'] <= end)

num_anomalies = anomaly_mask.sum()

# Compile the overview
overview = {
    "Total Entries": total_entries,
    "Number of Anomalies": num_anomalies,
    "Minimum Value": min_value,
    "Maximum Value": max_value
}

print(overview)
