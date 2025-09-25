import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class AISPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_columns = ['latitude', 'longitude', 'sog', 'cog']
    
    def clean_data(self, df):
        """Clean AIS data"""
        print("Cleaning data...")
        # Remove invalid coordinates
        df = df[
            (df['latitude'].between(-90, 90)) & 
            (df['longitude'].between(-180, 180))
        ]
        
        # Remove unrealistic speeds
        df = df[df['sog'] <= 50]
        df = df[df['sog'] >= 0]
        
        # Handle missing values
        df = df.dropna()
        
        print(f"After cleaning: {len(df)} records")
        return df
    
    def create_sequences(self, df, sequence_length=10, prediction_length=5):
        """Create sequences for training"""
        sequences = []
        targets = []
        
        for mmsi, vessel_data in df.groupby('mmsi'):
            vessel_data = vessel_data.sort_values('timestamp')
            values = vessel_data[self.feature_columns].values
            
            for i in range(len(values) - sequence_length - prediction_length):
                seq = values[i:i + sequence_length]
                target = values[i + sequence_length:i + sequence_length + prediction_length]
                sequences.append(seq)
                targets.append(target)
        
        print(f"Created {len(sequences)} sequences")
        return np.array(sequences), np.array(targets)

# Test the preprocessor
if __name__ == "__main__":
    from sample_data_generator import generate_realistic_ais_data
    
    # Generate sample data
    df = generate_realistic_ais_data(num_vessels=3, hours=24)
    
    # Test preprocessing
    preprocessor = AISPreprocessor()
    clean_df = preprocessor.clean_data(df)
    sequences, targets = preprocessor.create_sequences(clean_df)
    
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
