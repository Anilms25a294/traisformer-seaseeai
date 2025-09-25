"""
AIS Data Loader for SeaSeeAI Project
"""

import pandas as pd
import numpy as np
from pathlib import Path

class AISDataLoader:
    def __init__(self, data_path="data/raw/"):
        self.data_path = Path(data_path)
        self.data = None
    
    def load_sample_data(self):
        """Load a small sample dataset for testing"""
        # For now, create sample data. We'll replace with real AIS data later.
        sample_data = {
            'mmsi': [123456789, 123456789, 123456789],
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:02:00'],
            'latitude': [37.7749, 37.7750, 37.7751],
            'longitude': [-122.4194, -122.4193, -122.4192],
            'sog': [10.5, 10.6, 10.7],  # Speed Over Ground
            'cog': [45.0, 46.0, 47.0]   # Course Over Ground
        }
        self.data = pd.DataFrame(sample_data)
        return self.data
    
    def explore_data(self):
        """Basic data exploration"""
        if self.data is not None:
            print("Data Overview:")
            print(self.data.head())
            print("\nData Info:")
            print(self.data.info())
            print("\nBasic Statistics:")
            print(self.data.describe())
        else:
            print("No data loaded. Call load_sample_data() first.")

# Test the loader
if __name__ == "__main__":
    loader = AISDataLoader()
    data = loader.load_sample_data()
    loader.explore_data()
