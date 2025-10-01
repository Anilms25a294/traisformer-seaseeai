"""
Process real AIS data files
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class RealAISProcessor:
    def __init__(self):
        self.required_columns = ['timestamp', 'latitude', 'longitude', 'sog', 'cog', 'mmsi']
    
    def load_real_data(self, file_path):
        """Load and inspect real AIS data"""
        print(f"Loading real AIS data from: {file_path}")
        
        # Read the file
        try:
            df = pd.read_csv(file_path)
            print(f"Original data shape: {df.shape}")
            print("Original columns:", list(df.columns))
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
        
        return df
    
    def standardize_columns(self, df):
        """Convert real AIS data to our standard format"""
        print("Standardizing column names...")
        
        # Common column mappings for AIS data
        column_mappings = {
            # MarineCadastre format
            'BaseDateTime': 'timestamp',
            'LAT': 'latitude', 
            'LON': 'longitude',
            'SOG': 'sog',
            'COG': 'cog',
            'MMSI': 'mmsi',
            'VesselType': 'vessel_type',
            'Heading': 'heading',
            
            # Other common formats
            'datetime': 'timestamp',
            'lat': 'latitude',
            'lon': 'longitude',
            'speed': 'sog',
            'course': 'cog',
            'imo': 'mmsi'
        }
        
        # Rename columns
        df = df.rename(columns=column_mappings)
        
        # Keep only columns we need
        available_columns = [col for col in self.required_columns if col in df.columns]
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        print(f"Available columns: {available_columns}")
        print(f"Missing columns: {missing_columns}")
        
        # Select only available required columns
        df = df[available_columns]
        
        return df, missing_columns
    
    def clean_real_data(self, df):
        """Clean real AIS data"""
        print("Cleaning real AIS data...")
        
        original_count = len(df)
        
        # Remove rows with missing values
        df = df.dropna()
        print(f"After removing NA: {len(df)} rows (removed {original_count - len(df)})")
        
        # Filter valid coordinates
        df = df[
            (df['latitude'].between(-90, 90)) & 
            (df['longitude'].between(-180, 180))
        ]
        print(f"After coordinate filter: {len(df)} rows")
        
        # Filter valid speeds (0-50 knots)
        df = df[df['sog'].between(0, 50)]
        print(f"After speed filter: {len(df)} rows")
        
        # Filter valid courses (0-360 degrees)
        df = df[df['cog'].between(0, 360)]
        print(f"After course filter: {len(df)} rows")
        
        # Sort by vessel and timestamp
        df = df.sort_values(['mmsi', 'timestamp'])
        
        print(f"Final cleaned data: {len(df)} rows")
        return df
    
    def explore_real_data(self, df):
        """Explore the real AIS data"""
        print("\n=== Real AIS Data Exploration ===")
        print(f"Total records: {len(df):,}")
        print(f"Unique vessels: {df['mmsi'].nunique()}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Data coverage: {len(df) / df['mmsi'].nunique():.1f} points per vessel")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(f"Latitude: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        print(f"Speed: {df['sog'].min():.1f} to {df['sog'].max():.1f} knots")
        print(f"Course: {df['cog'].min():.1f} to {df['cog'].max():.1f} degrees")
        
        return df
    
    def process_file(self, file_path):
        """Complete processing pipeline for real AIS file"""
        # Load data
        df = self.load_real_data(file_path)
        if df is None:
            return None
        
        # Standardize columns
        df, missing = self.standardize_columns(df)
        
        # Check if we have minimum required columns
        min_required = ['timestamp', 'latitude', 'longitude', 'mmsi']
        if not all(col in df.columns for col in min_required):
            print("❌ Missing required columns. Cannot process.")
            return None
        
        # Clean data
        df = self.clean_real_data(df)
        
        # Explore data
        df = self.explore_real_data(df)
        
        return df

# Test with your file
if __name__ == "__main__":
    processor = RealAISProcessor()
    
    # Process your downloaded file
    real_df = processor.process_file("AIS_2024_12_311.csv")
    
    if real_df is not None:
        # Save processed data
        real_df.to_csv("data/real_ais/processed_ais_data.csv", index=False)
        print("✅ Processed real AIS data saved to: data/real_ais/processed_ais_data.csv")
        
        # Show sample of processed data
        print("\nSample of processed data:")
        print(real_df.head())
    else:
        print("❌ Failed to process real AIS data")
