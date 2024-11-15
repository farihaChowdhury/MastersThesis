from kafka import KafkaConsumer
import json
from datetime import datetime
from anomaly_detection import AnomalyDetector
from postgres_client import PostgresClient
from config import KAFKA_SERVER, KAFKA_TOPIC, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT

# Kafka consumer setup
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_SERVER],
    api_version=(0,10),
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Anomaly detector setup
detector = AnomalyDetector(model_path='./models/Isolation1.keras', threshold=0.05)  

# PostgreSQL client setup
postgres_client = PostgresClient(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)

SEQ_LENGTH = 1
sequence = []

for message in consumer:
    data = message.value

    print("DATA###", data)
    sensor_value = data['value']
    timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S")
    
    postgres_client.insert_data(timestamp, sensor_value)
    sequence.append(sensor_value)
    
    if len(sequence) >= SEQ_LENGTH:
        sequence = sequence[-SEQ_LENGTH:]

        # print("sequence###", sequence)
        is_anomaly = detector.detect_anomaly(sequence)
        
        if is_anomaly:
            print(f"Anomaly detected at {data['timestamp']}: value {data['value']}")
            is_anomaly = bool(is_anomaly)
            postgres_client.insert_anomaly(timestamp, is_anomaly)
