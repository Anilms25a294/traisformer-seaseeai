"""
Advanced AIS data preprocessor with normalization and time handling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class AdvancedAISPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_columns = ['latitude', 'longitude', 'sog', 'cog']
        
    def preprocess_data(self, df, vessel_id_col='mmsi', timestamp_col='timestamp'):
        """Preprocess the raw AIS data"""
        print("Preprocessing data...")
        
        # Convert timestamp to datetime and sort
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(by=[vessel_id_col, timestamp_col])
        
        # Clean data
        df = self.clean_data(df)
        
        # Normalize data per vessel
        df_normalized = self.normalize_data(df, vessel_id_col)
        
        return df_normalized
    
    def clean_data(self, df):
        """Clean AIS data"""
        # Remove invalid coordinates
        df = df[
            (df['latitude'].between(-90, 90)) & 
            (df['longitude'].between(-180, 180))
        ]
        
        # Remove unrealistic speeds
        df = df[df['sog'] <= 50]
        df = df[df['sog'] >= 0]
        
        # Handle missing values
        df = df.dropna(subset=self.feature_columns)
        
        return df
    
    def normalize_data(self, df, vessel_id_col):
        """Normalize data per vessel"""
        normalized_dfs = []
        
        for vessel_id, vessel_data in df.groupby(vessel_id_col):
            # Initialize scaler for this vessel
            if vessel_id not in self.scalers:
                self.scalers[vessel_id] = StandardScaler()
            
            # Fit and transform the features
            vessel_data[self.feature_columns] = self.scalers[vessel_id].fit_transform(vessel_data[self.feature_columns])
            normalized_dfs.append(vessel_data)
        
        return pd.concat(normalized_dfs)
    
    def create_sequences(self, df, sequence_length=10, prediction_length=5, vessel_id_col='mmsi'):
        """Create sequences for training"""
        sequences = []
        targets = []
        
        for vessel_id, vessel_data in df.groupby(vessel_id_col):
            values = vessel_data[self.feature_columns].values
            
            for i in range(len(values) - sequence_length - prediction_length):
                seq = values[i:i + sequence_length]
                target = values[i + sequence_length:i + sequence_length + prediction_length]
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    # Test the advanced preprocessor
    from sample_data_generator import generate_realistic_ais_data
    
    # Generate sample data
    df = generate_realistic_ais_data(num_vessels=3, hours=24)
    
    # Preprocess
    preprocessor = AdvancedAISPreprocessor()
    processed_df = preprocessor.preprocess_data(df, vessel_id_col='mmsi', timestamp_col='timestamp')
    
    # Create sequences
    sequences, targets = preprocessor.create_sequences(processed_df)
    
    print(f"Sequences: {sequences.shape}, Targets: {targets.shape}")
