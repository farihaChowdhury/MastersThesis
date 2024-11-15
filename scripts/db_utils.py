# scripts/db_utils.py

import psycopg2
from psycopg2 import sql

def connect_db():
    conn = psycopg2.connect(
        dbname='testDB',
        user='fariha',
        password='fariha',
        host='localhost',
        port='5433'
    )
    return conn

def create_tables(conn):
    create_data_table_query = """
    CREATE TABLE IF NOT EXISTS data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ,
        value FLOAT,
        is_anomaly BOOLEAN
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_data_table_query)
        conn.commit()

# CREATE TABLE sensor_data (
#     timestamp TIMESTAMPTZ NOT NULL,
#     sensor_00 DOUBLE PRECISION,
#     sensor_01 DOUBLE PRECISION,
#     ...
#     machine_status TEXT,
#     PRIMARY KEY (timestamp)
# );
# SELECT create_hypertable('sensor_data', 'timestamp');


# -- Connect to your TimescaleDB database
# \c your_database

# -- Create table to store all sensor data
# CREATE TABLE sensor_data (
#     id SERIAL PRIMARY KEY,
#     timestamp TIMESTAMPTZ NOT NULL,
#     sensor_values FLOAT[] NOT NULL
# );
# SELECT create_hypertable('sensor_data', 'timestamp');

# -- Create table to store anomalies
# CREATE TABLE anomalies (
#     id SERIAL PRIMARY KEY,
#     timestamp TIMESTAMPTZ NOT NULL,
#     sensor_values FLOAT[] NOT NULL,
#     is_anomaly BOOLEAN NOT NULL
# );
# SELECT create_hypertable('anomalies', 'timestamp');

