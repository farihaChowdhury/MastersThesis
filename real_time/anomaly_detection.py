import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

class AnomalyDetector:
    def __init__(self, model_path, threshold):
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def preprocess_data(self, sequence):

        sequence_array = np.array(sequence).reshape((-1,  1))
        return sequence_array
    
    def preprocess_data_lstm(self, sequence):

        sequence = np.array(sequence).reshape((-1,  1))
        scaler = MinMaxScaler()
        sequence_array = scaler.fit_transform(sequence)

        sequence_array = np.expand_dims(sequence_array, -1)
        sequence_array = sequence_array.astype('float32')
        return sequence_array

    def detect_anomaly(self, sequence):
    
        preprocessed_data = self.preprocess_data(sequence)

        # print("preprocessed_data", preprocessed_data)
        
        reconstruction = self.model.predict(preprocessed_data)
        # print("reconstruction", reconstruction)

        return reconstruction < 0

    def detect_anomaly_lstm(self, sequence):
        
            preprocessed_data = self.preprocess_data_lstm(sequence)

            # print("preprocessed_data", preprocessed_data)
            
            reconstruction = self.model.predict(preprocessed_data)

            reconstruction = reconstruction.reshape(preprocessed_data.shape)

            # reconstruction = np.squeeze(reconstruction) 
            # print("reconstruction", reconstruction)

            reconstruction_loss = np.mean(np.abs(reconstruction - preprocessed_data), axis=(1, 2))

            # print("reconstruction_loss#", reconstruction_loss)
            return reconstruction_loss > self.threshold
