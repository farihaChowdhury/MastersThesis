# real_time/postgres_client.py

import psycopg2
from psycopg2 import sql

class PostgresClient:
    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def insert_data(self, timestamp, sensor_values):
        insert_query = sql.SQL("""
        INSERT INTO sensor_data (timestamp, value) 
        VALUES (%s, %s)
        """)
        with self.conn.cursor() as cursor:
            cursor.execute(insert_query, (timestamp, sensor_values))
            self.conn.commit()

    def insert_anomaly(self, timestamp, is_anomaly):
        update_query = sql.SQL("""
        UPDATE sensor_data SET is_anomaly = %s WHERE timestamp = %s
        """)
        with self.conn.cursor() as cursor:
            cursor.execute(update_query, (is_anomaly, timestamp))
            self.conn.commit()

    def close(self):
        self.conn.close()
