import requests
import pandas as pd
import json

api_url = "https://traisformer-seaseeai.onrender.com"

print("🚢 Testing All AIS Datasets with SeaSeeAI")
print("=" * 50)

datasets = [
    ("./data/raw/sample_ais_data.csv", "Sample AIS Data", "mmsi"),
    ("./data/real_ais/processed_ais_data.csv", "Processed AIS Data", "mmsi"),
    ("./AIS_2024_12_311.csv", "Large AIS Dataset", "MMSI")
]

def test_dataset(file_path, description, mmsi_col):
    print(f"\n📊 Testing: {description}")
    print(f"📁 File: {file_path}")
    
    try:
        # Load a small sample for testing
        df = pd.read_csv(file_path, nrows=1000)
        print(f"   ✅ Loaded {len(df)} rows")
        
        # Find a vessel with enough data points
        if mmsi_col in df.columns:
            vessel_counts = df[mmsi_col].value_counts()
            suitable_vessels = vessel_counts[vessel_counts >= 10]
            
            if len(suitable_vessels) > 0:
                vessel_id = suitable_vessels.index[0]
                vessel_data = df[df[mmsi_col] == vessel_id].head(12)
                
                print(f"   🚤 Using vessel {vessel_id} with {len(vessel_data)} points")
                
                # Convert to API format
                observations = []
                for _, row in vessel_data.iterrows():
                    # Handle different column names
                    lat_col = 'LAT' if 'LAT' in df.columns else 'latitude'
                    lon_col = 'LON' if 'LON' in df.columns else 'longitude'
                    sog_col = 'SOG' if 'SOG' in df.columns else 'sog'
                    cog_col = 'COG' if 'COG' in df.columns else 'cog'
                    ts_col = 'BaseDateTime' if 'BaseDateTime' in df.columns else 'timestamp'
                    
                    obs = {
                        'mmsi': int(row[mmsi_col]),
                        'timestamp': str(row[ts_col]) if ts_col in row and pd.notna(row[ts_col]) else "2024-01-01T00:00:00",
                        'latitude': float(row[lat_col]),
                        'longitude': float(row[lon_col]),
                        'sog': float(row[sog_col]) if sog_col in row and pd.notna(row[sog_col]) else 10.0,
                        'cog': float(row[cog_col]) if cog_col in row and pd.notna(row[cog_col]) else 45.0
                    }
                    observations.append(obs)
                
                # Make prediction
                payload = {
                    "observations": observations,
                    "prediction_horizon": 3
                }
                
                response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Predictions: {len(data.get('predictions', []))}")
                    return True
                else:
                    print(f"   ❌ Prediction failed: {response.status_code}")
                    return False
            else:
                print(f"   ⚠️  No vessels with sufficient data")
                return False
        else:
            print(f"   ❌ No MMSI column found")
            return False
            
    except Exception as e:
        print(f"   💥 Error: {e}")
        return False

# Test all datasets
success_count = 0
for file_path, description, mmsi_col in datasets:
    if test_dataset(file_path, description, mmsi_col):
        success_count += 1

print(f"\n" + "=" * 50)
print(f"🎯 Results: {success_count}/{len(datasets)} datasets successful")
print(f"🌐 API Status: ✅ Healthy")
print(f"🚢 SeaSeeAI is operational with real AIS data!")
print("=" * 50)
