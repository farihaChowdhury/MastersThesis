from kafka import KafkaProducer
import pandas as pd
import json
from time import sleep
import logging

from sklearn.preprocessing import MinMaxScaler

# Enable logging
# logging.basicConfig(level=logging.DEBUG)

producer = KafkaProducer(bootstrap_servers='kafka:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'),max_request_size=20971520,api_version = (2, 0, 2))
data = pd.read_csv('./data/machine_temperature_system_failure.csv')

train_size = int(len(data) * 0.8)
# train_data = data[:train_size]
test_data = data[train_size:]


for index, row in test_data.iterrows():
    try:
        producer.send('sensor-data', row.to_dict())
        logging.info("DATA INFO", row)
        producer.flush()
        sleep(0.02)   
    except Exception as e:
        logging.error(f"Error sending message at index {index}: {e}")

