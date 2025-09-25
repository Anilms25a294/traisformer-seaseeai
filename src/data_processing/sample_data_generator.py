"""
Generate realistic sample AIS data for immediate testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_realistic_ais_data(num_vessels=10, hours=72, output_file="data/raw/sample_ais_data.csv"):
    """Generate realistic vessel trajectories with different behaviors"""
    
    # Create directory if it doesn't exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    vessels = []
    vessel_types = ['Cargo', 'Tanker', 'Passenger', 'Fishing', 'Tug']
    
    print(f"Generating realistic AIS data for {num_vessels} vessels over {hours} hours...")
    
    for vessel_id in range(1, num_vessels + 1):
        vessel_type = vessel_types[vessel_id % len(vessel_types)]
        
        # Different starting points based on vessel type
        if vessel_type == 'Cargo':
            base_lat, base_lon = 37.7, -122.4  # Near San Francisco
        elif vessel_type == 'Tanker':
            base_lat, base_lon = 40.7, -74.0   # Near New York
        elif vessel_type == 'Fishing':
            base_lat, base_lon = 47.6, -122.3  # Near Seattle
        else:
            base_lat, base_lon = 34.0, -118.0  # Near Los Angeles
        
        # Add some variation
        base_lat += np.random.uniform(-1, 1)
        base_lon += np.random.uniform(-1, 1)
        
        current_lat, current_lon = base_lat, base_lon
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        
        for hour in range(hours):
            timestamp = start_time + timedelta(hours=hour)
            
            # Different movement patterns based on vessel type
            if vessel_type == 'Cargo':
                # Straight routes, consistent speed
                speed = np.random.uniform(12, 18)
                course_change = np.random.normal(0, 2)
            elif vessel_type == 'Tanker':
                # Slower, more stable
                speed = np.random.uniform(8, 14)
                course_change = np.random.normal(0, 1)
            elif vessel_type == 'Fishing':
                # Erratic movement
                speed = np.random.uniform(2, 8)
                course_change = np.random.normal(0, 10)
            else:  # Passenger, Tug
                speed = np.random.uniform(10, 16)
                course_change = np.random.normal(0, 5)
            
            # Simulate movement
            current_lat += speed * 0.001 * np.random.normal(0.1, 0.01)
            current_lon += speed * 0.001 * np.random.normal(0.1, 0.01)
            
            # Add some realistic noise
            current_lat += np.random.normal(0, 0.001)
            current_lon += np.random.normal(0, 0.001)
            
            vessels.append({
                'mmsi': 100000000 + vessel_id,  # Unique vessel identifier
                'timestamp': timestamp,
                'latitude': current_lat,
                'longitude': current_lon,
                'sog': speed,  # Speed Over Ground (knots)
                'cog': (hour * 5 + course_change) % 360,  # Course Over Ground
                'vessel_type': vessel_type,
                'heading': (hour * 5 + course_change) % 360,
                'draught': np.random.uniform(5, 15)  # How deep the vessel sits
            })
    
    df = pd.DataFrame(vessels)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} data points")
    print(f"Saved to: {output_file}")
    print(f"Vessel distribution: {df['vessel_type'].value_counts().to_dict()}")
    
    return df

def explore_generated_data(file_path="data/raw/sample_ais_data.csv"):
    """Quick exploration of the generated data"""
    df = pd.read_csv(file_path)
    
    print("\n=== Generated Data Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of vessels: {df['mmsi'].nunique()}")
    print(f"Vessel types: {df['vessel_type'].value_counts()}")
    print(f"Speed range: {df['sog'].min():.1f} - {df['sog'].max():.1f} knots")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_realistic_ais_data()
    
    # Explore the data
    explore_generated_data()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_realistic_ais_data(num_vessels=10, hours=72, output_file="data/raw/sample_ais_data.csv"):
    """Generate realistic vessel trajectories with different behaviors"""
    
    # Create directory if it doesn't exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    vessels = []
    vessel_types = ['Cargo', 'Tanker', 'Passenger', 'Fishing', 'Tug']
    
    print(f"Generating realistic AIS data for {num_vessels} vessels over {hours} hours...")
    
    for vessel_id in range(1, num_vessels + 1):
        vessel_type = vessel_types[vessel_id % len(vessel_types)]
        
        # Different starting points based on vessel type
        if vessel_type == 'Cargo':
            base_lat, base_lon = 37.7, -122.4  # Near San Francisco
        elif vessel_type == 'Tanker':
            base_lat, base_lon = 40.7, -74.0   # Near New York
        elif vessel_type == 'Fishing':
            base_lat, base_lon = 47.6, -122.3  # Near Seattle
        else:
            base_lat, base_lon = 34.0, -118.0  # Near Los Angeles
        
        # Add some variation
        base_lat += np.random.uniform(-1, 1)
        base_lon += np.random.uniform(-1, 1)
        
        current_lat, current_lon = base_lat, base_lon
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        
        for hour in range(hours):
            timestamp = start_time + timedelta(hours=hour)
            
            # Different movement patterns based on vessel type
            if vessel_type == 'Cargo':
                # Straight routes, consistent speed
                speed = np.random.uniform(12, 18)
                course_change = np.random.normal(0, 2)
            elif vessel_type == 'Tanker':
                # Slower, more stable
                speed = np.random.uniform(8, 14)
                course_change = np.random.normal(0, 1)
            elif vessel_type == 'Fishing':
                # Erratic movement
                speed = np.random.uniform(2, 8)
                course_change = np.random.normal(0, 10)
            else:  # Passenger, Tug
                speed = np.random.uniform(10, 16)
                course_change = np.random.normal(0, 5)
            
            # Simulate movement
            current_lat += speed * 0.001 * np.random.normal(0.1, 0.01)
            current_lon += speed * 0.001 * np.random.normal(0.1, 0.01)
            
            # Add some realistic noise
            current_lat += np.random.normal(0, 0.001)
            current_lon += np.random.normal(0, 0.001)
            
            vessels.append({
                'mmsi': 100000000 + vessel_id,  # Unique vessel identifier
                'timestamp': timestamp,
                'latitude': current_lat,
                'longitude': current_lon,
                'sog': speed,  # Speed Over Ground (knots)
                'cog': (hour * 5 + course_change) % 360,  # Course Over Ground
                'vessel_type': vessel_type,
                'heading': (hour * 5 + course_change) % 360,
                'draught': np.random.uniform(5, 15)  # How deep the vessel sits
            })
    
    df = pd.DataFrame(vessels)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} data points")
    print(f"Saved to: {output_file}")
    print(f"Vessel distribution: {df['vessel_type'].value_counts().to_dict()}")
    
    return df

def explore_generated_data(file_path="data/raw/sample_ais_data.csv"):
    """Quick exploration of the generated data"""
    df = pd.read_csv(file_path)
    
    print("\n=== Generated Data Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of vessels: {df['mmsi'].nunique()}")
    print(f"Vessel types: {df['vessel_type'].value_counts()}")
    print(f"Speed range: {df['sog'].min():.1f} - {df['sog'].max():.1f} knots")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_realistic_ais_data()
    
    # Explore the data
    explore_generated_data()
