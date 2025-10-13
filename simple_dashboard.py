import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os

# Configuration
API_URL = "https://traisformer-seaseeai.onrender.com"

st.set_page_config(
    page_title="SeaSeeAI - Maritime Trajectory Prediction",
    page_icon="🚢",
    layout="wide"
)

# Simple maritime regions
MARITIME_REGIONS = {
    'gulf_mexico': {
        'name': 'Gulf of Mexico', 
        'center': {'lat': 28.0, 'lon': -90.0}, 
        'zoom': 8
    },
    'atlantic_coast': {
        'name': 'Atlantic Coast',
        'center': {'lat': 35.0, 'lon': -75.0}, 
        'zoom': 7
    },
    'pacific_coast': {
        'name': 'Pacific Coast',
        'center': {'lat': 37.8, 'lon': -122.4},
        'zoom': 7
    }
}

# Vessel type mapping
VESSEL_TYPES = {
    30: 'Fishing', 31: 'Fishing', 32: 'Towing', 33: 'Dredging', 34: 'Diving', 35: 'Military',
    36: 'Sailing', 37: 'Pleasure Craft', 50: 'Pilot Vessel', 51: 'Search and Rescue',
    52: 'Tug', 53: 'Port Tender', 54: 'Anti-pollution', 55: 'Law Enforcement',
    57: 'Passenger', 70: 'Cargo', 71: 'Cargo', 72: 'Cargo', 80: 'Tanker', 90: 'Other'
}

st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        color: #1f77b4; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    .success-box { 
        background-color: #d4edda; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.sidebar.error(f"API connection failed: {e}")
        return None

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def make_prediction(observations, horizon):
    """Make trajectory prediction"""
    try:
        payload = {
            "observations": observations,
            "prediction_horizon": horizon
        }
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def load_real_ais_data():
    """Load AIS data with simple error handling"""
    try:
        data_path = "AIS_2024_12_311.csv"
        if not os.path.exists(data_path):
            st.sidebar.warning("AIS data file not found")
            return None
            
        st.sidebar.info("Loading AIS data...")
        
        # Simple read with minimal processing
        use_cols = ['MMSI', 'LAT', 'LON', 'SOG', 'COG', 'VesselName', 'VesselType']
        df = pd.read_csv(data_path, usecols=use_cols, nrows=1000, low_memory=False)
        
        # Basic cleaning
        df = df.dropna(subset=['LAT', 'LON'])
        df = df[(df['LAT'].between(20, 50)) & (df['LON'].between(-130, -60))]
        
        # Add vessel type names
        df['VesselTypeName'] = df['VesselType'].map(VESSEL_TYPES).fillna('Unknown')
        
        st.sidebar.success(f"Loaded {len(df)} positions")
        return df
        
    except Exception as e:
        st.sidebar.error(f"Data loading error: {str(e)}")
        return None

def generate_simple_track(base_lat, base_lon, base_sog=10.0, base_cog=45.0, num_points=12):
    """Generate maritime tracks that stay in water"""
    
    # Define maritime bounds for Gulf of Mexico
    maritime_bounds = {
        'min_lat': 25.0, 'max_lat': 30.5,
        'min_lon': -97.0, 'max_lon': -88.0
    }
    
    tracks = []
    
    # Start from a known maritime position in Gulf of Mexico
    if not (maritime_bounds['min_lat'] <= base_lat <= maritime_bounds['max_lat'] and 
            maritime_bounds['min_lon'] <= base_lon <= maritime_bounds['max_lon']):
        # If starting position is not in Gulf, use a default maritime position
        base_lat, base_lon = 28.5, -90.5  # Central Gulf of Mexico
    
    for i in range(num_points):
        # Generate realistic maritime movement
        lat_variation = np.random.uniform(-0.02, 0.02)
        lon_variation = np.random.uniform(-0.02, 0.02)
        
        new_lat = base_lat + (i * 0.01) + lat_variation
        new_lon = base_lon + (i * 0.01) + lon_variation
        
        # Ensure positions stay within maritime bounds
        new_lat = np.clip(new_lat, maritime_bounds['min_lat'], maritime_bounds['max_lat'])
        new_lon = np.clip(new_lon, maritime_bounds['min_lon'], maritime_bounds['max_lon'])
        
        # Add realistic speed variations
        sog_variation = np.random.uniform(-2.0, 2.0)
        new_sog = max(0.5, base_sog + sog_variation)
        
        tracks.append({
            'latitude': new_lat,
            'longitude': new_lon,
            'sog': new_sog,
            'cog': base_cog,
            'timestamp': (datetime.now() - timedelta(minutes=(num_points - i) * 10)).isoformat()
        })
    
    return tracks

def create_simple_map(historical_data, predicted_data=None, region='gulf_mexico'):
    """Create a simple, reliable map without complex features"""
    
    if not historical_data:
        st.warning("No data available for map")
        return go.Figure()
    
    # Get map configuration
    map_config = MARITIME_REGIONS.get(region, MARITIME_REGIONS['gulf_mexico'])
    
    fig = go.Figure()
    
    # Historical track (always show)
    if historical_data:
        historical_lats = [obs['latitude'] for obs in historical_data]
        historical_lons = [obs['longitude'] for obs in historical_data]
        
        fig.add_trace(go.Scattermapbox(
            lat=historical_lats,
            lon=historical_lons,
            mode='lines+markers',
            name='Historical Track',
            line=dict(color='blue', width=4),
            marker=dict(size=8, color='blue'),
            hovertemplate='<b>Historical</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
        ))
    
    # Predicted track
    if predicted_data:
        predicted_lats = [pred['latitude'] for pred in predicted_data]
        predicted_lons = [pred['longitude'] for pred in predicted_data]
        
        fig.add_trace(go.Scattermapbox(
            lat=predicted_lats,
            lon=predicted_lons,
            mode='lines+markers',
            name='Predicted Track',
            line=dict(color='red', width=4),
            marker=dict(size=8, color='red', symbol='diamond'),
            hovertemplate='<b>Predicted</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
        ))
    
    # Use last point as center if available
    if historical_data:
        center_lat = historical_data[-1]['latitude']
        center_lon = historical_data[-1]['longitude']
    else:
        # Force Gulf of Mexico center for all tracks
        center_lat, center_lon = 28.5, -90.5  # Central Gulf of Mexico
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=map_config['zoom'],
            center=dict(lat=center_lat, lon=center_lon)
        ),
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">🚢 SeaSeeAI - Maritime Trajectory Prediction</div>', unsafe_allow_html=True)
    
    # Initialize session state for data persistence
    if 'observations' not in st.session_state:
        st.session_state.observations = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Sidebar - Simplified
    st.sidebar.title("Configuration")
    
    # API Health Check
    if st.sidebar.button("Check API Status"):
        health_data = check_api_health()
        if health_data:
            st.sidebar.success("✅ API Healthy")
            st.sidebar.info(f"Uptime: {health_data.get('uptime', 0):.0f}s")
        else:
            st.sidebar.error("❌ API Unavailable")
    
    # Load data once
    if 'ais_data' not in st.session_state:
        st.session_state.ais_data = load_real_ais_data()
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Setup")
        
        # Simple controls
        horizon = st.slider("Prediction Steps", 1, 10, 5)
        
        data_source = st.radio("Choose Data Source:", 
                              ["Manual Input", "Real AIS Data", "Sample Data"])
        
        observations = []
        
        if data_source == "Real AIS Data" and st.session_state.ais_data is not None:
            st.success("Using Real AIS Data")
            
            # Simple vessel selection
            vessels = st.session_state.ais_data['MMSI'].unique()[:20]
            selected_mmsi = st.selectbox("Select Vessel", vessels)
            
            if selected_mmsi:
                vessel_data = st.session_state.ais_data[
                    st.session_state.ais_data['MMSI'] == selected_mmsi
                ].iloc[0]
                
                vessel_name = vessel_data['VesselName'] if pd.notna(vessel_data['VesselName']) else "Unknown"
                vessel_type = vessel_data['VesselTypeName']
                
                st.info(f"**{vessel_name}** ({vessel_type})")
                st.info(f"Position: {vessel_data['LAT']:.4f}°N, {vessel_data['LON']:.4f}°W")
                
                # Generate simple track
                simple_track = generate_simple_track(
                    vessel_data['LAT'],
                    vessel_data['LON'],
                    vessel_data['SOG'] if pd.notna(vessel_data['SOG']) else 10.0,
                    vessel_data['COG'] if pd.notna(vessel_data['COG']) else 45.0
                )
                
                # Convert to observations
                for point in simple_track:
                    observations.append({
                        'mmsi': int(selected_mmsi),
                        'timestamp': point['timestamp'],
                        'latitude': point['latitude'],
                        'longitude': point['longitude'],
                        'sog': point['sog'],
                        'cog': point['cog']
                    })
                
                st.success(f"✅ Ready with {len(observations)} points")
                
        elif data_source == "Sample Data":
            st.success("Using Sample Data")
            
            # Simple sample track in Gulf of Mexico
            sample_track = generate_simple_track(29.74423, -93.86838, 12.0, 45.0)
            
            for point in sample_track:
                observations.append({
                    'mmsi': 100000001,
                    'timestamp': point['timestamp'],
                    'latitude': point['latitude'],
                    'longitude': point['longitude'],
                    'sog': point['sog'],
                    'cog': point['cog']
                })
            
            st.success(f"✅ Created {len(observations)} sample points")
            
        else:  # Manual Input
            st.success("Manual Input")
            
            # Simple manual input
            base_lat = st.number_input("Latitude", value=29.74423, format="%.5f")
            base_lon = st.number_input("Longitude", value=-93.86838, format="%.5f")
            base_sog = st.number_input("Speed (knots)", value=12.0)
            base_cog = st.number_input("Course (degrees)", value=45.0)
            
            num_points = st.slider("Track Points", 5, 15, 12)
            
            manual_track = generate_simple_track(base_lat, base_lon, base_sog, base_cog, num_points)
            
            for point in manual_track:
                observations.append({
                    'mmsi': 100000001,
                    'timestamp': point['timestamp'],
                    'latitude': point['latitude'],
                    'longitude': point['longitude'],
                    'sog': point['sog'],
                    'cog': point['cog']
                })
            
            st.success(f"✅ Created {len(observations)} manual points")
        
        # Store observations in session state
        st.session_state.observations = observations
        
        # Prediction button
        if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
            if observations:
                with st.spinner("Making predictions..."):
                    result = make_prediction(observations, horizon)
                    st.session_state.last_result = result
            else:
                st.error("Please create observations first")
    
    with col2:
        st.subheader("🎯 Results")
        
        # Show results if available
        if st.session_state.last_result:
            result = st.session_state.last_result
            
            st.success("✅ Predictions Generated!")
            
            # Simple metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
            with col2:
                st.metric("Time", f"{result.get('processing_time', 0):.3f}s")
            with col3:
                predictions = result.get('predictions', [])
                st.metric("Predictions", len(predictions))
            
            # Show predictions table
            if predictions:
                st.subheader("📈 Predicted Positions")
                pred_df = pd.DataFrame(predictions)
                st.dataframe(pred_df[['step', 'latitude', 'longitude', 'sog', 'cog']])
                
                # Show map
                st.subheader("🗺️ Trajectory Map")
                
                # Determine region based on last observation
                last_obs = st.session_state.observations[-1]
                region = 'gulf_mexico'  # Default
                if 30.0 <= last_obs['latitude'] <= 45.0 and -80.0 <= last_obs['longitude'] <= -70.0:
                    region = 'atlantic_coast'
                elif 32.0 <= last_obs['latitude'] <= 48.0 and -125.0 <= last_obs['longitude'] <= -117.0:
                    region = 'pacific_coast'
                
                # Create and display map
                fig = create_simple_map(st.session_state.observations, predictions, region)
                st.plotly_chart(fig, use_container_width=True)
                
                # Simple analysis
                st.subheader("📊 Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    start_lat = st.session_state.observations[0]['latitude']
                    start_lon = st.session_state.observations[0]['longitude']
                    end_lat = predictions[-1]['latitude'] if predictions else start_lat
                    end_lon = predictions[-1]['longitude'] if predictions else start_lon
                    
                    st.metric("Lat Change", f"{(end_lat - start_lat):.4f}°")
                    st.metric("Lon Change", f"{(end_lon - start_lon):.4f}°")
                
                with col2:
                    if predictions:
                        avg_sog = np.mean([p['sog'] for p in predictions])
                        avg_cog = np.mean([p['cog'] for p in predictions])
                    else:
                        avg_sog = np.mean([o['sog'] for o in st.session_state.observations])
                        avg_cog = np.mean([o['cog'] for o in st.session_state.observations])
                    
                    st.metric("Avg Speed", f"{avg_sog:.1f} kn")
                    st.metric("Avg Course", f"{avg_cog:.1f}°")
        else:
            st.info("Configure settings and generate predictions to see results")
    
    # Data overview at bottom
    if st.session_state.ais_data is not None:
        st.markdown("---")
        st.subheader("📈 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vessels", len(st.session_state.ais_data['MMSI'].unique()))
        with col2:
            st.metric("Data Points", len(st.session_state.ais_data))
        with col3:
            moving = len(st.session_state.ais_data[st.session_state.ais_data['SOG'] > 1.0])
            st.metric("Moving Vessels", moving)

if __name__ == "__main__":
    main()