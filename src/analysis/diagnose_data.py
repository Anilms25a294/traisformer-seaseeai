import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing.data_loader import load_ais_data

def diagnose_data():
    print("=== DATA DIAGNOSIS ===")
    
    # Load the data
    data = load_ais_data()
    print(f"Data shape: {data.shape}")
    print(f"Data type: {type(data)}")
    
    # Check basic statistics
    print("\n--- Basic Statistics ---")
    print(f"Latitude range: [{data[:, 0].min():.6f}, {data[:, 0].max():.6f}]")
    print(f"Longitude range: [{data[:, 1].min():.6f}, {data[:, 1].max():.6f}]")
    print(f"SOG range: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]")
    print(f"COG range: [{data[:, 3].min():.2f}, {data[:, 3].max():.2f}]")
    
    print("\n--- Mean and Std ---")
    print(f"Latitude - Mean: {data[:, 0].mean():.6f}, Std: {data[:, 0].std():.6f}")
    print(f"Longitude - Mean: {data[:, 1].mean():.6f}, Std: {data[:, 1].std():.6f}")
    print(f"SOG - Mean: {data[:, 2].mean():.2f}, Std: {data[:, 2].std():.2f}")
    print(f"COG - Mean: {data[:, 3].mean():.2f}, Std: {data[:, 3].std():.2f}")
    
    # Check for NaN values
    print(f"\nNaN values: {np.isnan(data).sum()}")
    
    # Check first few samples
    print("\n--- First 3 samples ---")
    for i in range(3):
        print(f"Sample {i}: LAT={data[i, 0]:.6f}, LON={data[i, 1]:.6f}, SOG={data[i, 2]:.2f}, COG={data[i, 3]:.2f}")

if __name__ == "__main__":
    diagnose_data()
