import requests
import pandas as pd
import json
from datetime import datetime
import time

api_url = "https://traisformer-seaseeai.onrender.com"

print("🚢 SeaSeeAI - Testing Predictions with Real AIS Data")
print("=" * 60)

# First, check the model info to see what we're working with
print("1. Checking model information...")
try:
    model_response = requests.get(f"{api_url}/model/info", timeout=10)
    if model_response.status_code == 200:
        model_info = model_response.json()
        print(f"   ✅ Model Type: {model_info.get('model_type')}")
        print(f"   ✅ Model Loaded: {model_info.get('loaded')}")
        print(f"   ✅ Prediction Length: {model_info.get('prediction_length')}")
    else:
        print(f"   ❌ Model info failed: {model_response.status_code}")
except Exception as e:
    print(f"   ❌ Model info error: {e}")

# Test with sample AIS data
print(f"\n2. Testing with sample AIS data...")
file_path = "./data/raw/sample_ais_data.csv"

try:
    df = pd.read_csv(file_path)
    print(f"   📊 Loaded {len(df)} rows from sample data")
    
    # Use vessel 100000001 with exactly 12 observations
    vessel_data = df[df['mmsi'] == 100000001].head(12)
    
    print(f"   🚤 Using vessel {vessel_data['mmsi'].iloc[0]} with {len(vessel_data)} observations")
    
    # Convert to API format
    observations = []
    for _, row in vessel_data.iterrows():
        obs = {
            'mmsi': int(row['mmsi']),
            'timestamp': str(row['timestamp']),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'sog': float(row['sog']),
            'cog': float(row['cog'])
        }
        observations.append(obs)
    
    print(f"   📍 First observation: Lat={observations[0]['latitude']:.4f}, Lon={observations[0]['longitude']:.4f}")
    print(f"   📍 Last observation: Lat={observations[-1]['latitude']:.4f}, Lon={observations[-1]['longitude']:.4f}")
    
    # Test different prediction horizons
    horizons = [3, 5]
    
    for horizon in horizons:
        print(f"\n3. Testing prediction horizon: {horizon} steps")
        
        payload = {
            "observations": observations,
            "prediction_horizon": horizon
        }
        
        start_time = time.time()
        response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ PREDICTION SUCCESS!")
            print(f"      Confidence: {data.get('confidence')}")
            print(f"      Processing Time: {processing_time:.2f}s")
            print(f"      Model Version: {data.get('model_version')}")
            print(f"      Predictions Generated: {len(data.get('predictions', []))}")
            
            predictions = data.get('predictions', [])
            if predictions:
                print(f"\n      📈 Predicted Trajectory (Horizon: {horizon}):")
                last_obs = observations[-1]
                print(f"         Current: Lat={last_obs['latitude']:.4f}, Lon={last_obs['longitude']:.4f}")
                
                for i, pred in enumerate(predictions):
                    lat_diff = pred.get('latitude', 0) - last_obs['latitude']
                    lon_diff = pred.get('longitude', 0) - last_obs['longitude']
                    print(f"         Step {i+1}: Lat={pred.get('latitude', 0):.4f} ({lat_diff:+.4f}), "
                          f"Lon={pred.get('longitude', 0):.4f} ({lon_diff:+.4f})")
        else:
            print(f"   ❌ Prediction failed: {response.status_code} - {response.text}")
            
except Exception as e:
    print(f"   💥 Error: {e}")

print(f"\n" + "=" * 60)
print("🎯 Testing complete!")
