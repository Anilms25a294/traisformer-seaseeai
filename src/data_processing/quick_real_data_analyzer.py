"""
Quick analysis of real AIS data to avoid system hangs
"""

import pandas as pd
import numpy as np
import sys
import os

def quick_analyze_real_data(file_path, sample_size=5000):
    """Quick analysis without loading full dataset"""
    print("=== Quick Real AIS Data Analysis ===")
    
    # Get file size
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")
    
    # Read only sample rows
    print(f"Reading first {sample_size} rows for analysis...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f"Sample shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic info
    print("\n--- Basic Info ---")
    print(df.info())
    
    # Check for key AIS columns
    expected_columns = ['mmsi', 'timestamp', 'latitude', 'longitude', 'sog', 'cog']
    available_columns = [col for col in expected_columns if col in df.columns]
    print(f"\nAvailable AIS columns: {available_columns}")
    
    # Basic statistics for numeric columns
    print("\n--- Basic Statistics ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    # Check for missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Check unique vessels
    if 'mmsi' in df.columns:
        print(f"\nUnique vessels in sample: {df['mmsi'].nunique()}")
    
    return df

if __name__ == "__main__":
    df = quick_analyze_real_data("AIS_2024_12_311.csv", sample_size=3000)
