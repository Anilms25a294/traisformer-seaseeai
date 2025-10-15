import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuration
API_URL = "https://traisformer-seaseeai.onrender.com"  # Live API

st.set_page_config(
    page_title="SeaSeeAI - Maritime Trajectory Prediction",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS
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
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
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
            return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.markdown('<div class="main-header">🚢 SeaSeeAI - Maritime Trajectory Prediction</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Health Check
    with st.sidebar:
        st.subheader("API Status")
        health_data = check_api_health()
        if health_data:
            st.success(f"✅ API Healthy")
            st.info(f"Uptime: {health_data.get('uptime', 0):.0f}s")
        else:
            st.error("❌ API Unavailable")
            return
    
    # Model Information
    model_info = get_model_info()
    if model_info:
        with st.sidebar:
            st.subheader("Model Info")
            st.write(f"Type: {model_info.get('model_type', 'N/A')}")
            st.write(f"Loaded: {model_info.get('loaded', False)}")
            st.write(f"Prediction Length: {model_info.get('prediction_length', 'N/A')}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Prediction Configuration")
        
        # Prediction horizon
        horizon = st.slider("Prediction Horizon (steps)", 1, 10, 5)
        
        # Data source selection
        data_source = st.radio("Data Source", 
                              ["Sample Data", "Manual Input"])
        
        if data_source == "Sample Data":
            st.info("Using built-in sample AIS data")
            try:
                df = pd.read_csv("./data/raw/sample_ais_data.csv")
                vessel_data = df[df['mmsi'] == 100000001].head(12)
                
                observations = []
                for _, row in vessel_data.iterrows():
                    observations.append({
                        'mmsi': int(row['mmsi']),
                        'timestamp': str(row['timestamp']),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'sog': float(row['sog']),
                        'cog': float(row['cog'])
                    })
                
                st.success(f"Loaded {len(observations)} observations from vessel 100000001")
                
                # Show the loaded data
                with st.expander("View Loaded Data"):
                    st.dataframe(vessel_data[['timestamp', 'latitude', 'longitude', 'sog', 'cog']])
                
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
                return
                
        else:  # Manual Input
            st.info("Enter vessel observation data manually")
            num_obs = st.number_input("Number of observations", 10, 20, 12)
            observations = []
            
            # Create a simple form for manual input
            base_lat = st.number_input("Base Latitude", value=40.8980, format="%.6f")
            base_lon = st.number_input("Base Longitude", value=-74.6890, format="%.6f")
            base_sog = st.number_input("Speed (SOG)", value=10.0)
            base_cog = st.number_input("Course (COG)", value=45.0)
            
            for i in range(num_obs):
                obs = {
                    'mmsi': 100000001,
                    'timestamp': (datetime.now()).isoformat(),
                    'latitude': base_lat + (i * 0.001),
                    'longitude': base_lon + (i * 0.001),
                    'sog': base_sog,
                    'cog': base_cog
                }
                observations.append(obs)
            
            st.success(f"Created {len(observations)} synthetic observations")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("Generating trajectory predictions..."):
                result = make_prediction(observations, horizon)
                
                if result:
                    st.success("✅ Predictions Generated Successfully!")
                    
                    # Display results
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    with col_metric2:
                        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                    with col_metric3:
                        st.metric("Predictions", len(result.get('predictions', [])))
                    
                    # Show predictions table
                    predictions = result.get('predictions', [])
                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        st.subheader("📈 Predicted Positions")
                        st.dataframe(pred_df[['step', 'latitude', 'longitude', 'sog', 'cog']])
                    
                    # Create visualization
                    st.subheader("🗺️ Trajectory Map")
                    
                    # Extract coordinates
                    historical_lats = [obs['latitude'] for obs in observations]
                    historical_lons = [obs['longitude'] for obs in observations]
                    predicted_lats = [pred['latitude'] for pred in predictions]
                    predicted_lons = [pred['longitude'] for pred in predictions]
                    
                    # Create map - FIXED: Using Scattermap instead of Scattermapbox
                    fig = go.Figure()
                    
                    # Historical track - using solid line
                    fig.add_trace(go.Scattermap(
                        lat=historical_lats,
                        lon=historical_lons,
                        mode='lines+markers',
                        name='Historical Track',
                        line=dict(color='blue', width=4),  # Removed 'dash' property
                        marker=dict(size=8, color='blue')
                    ))
                    
                    # Predicted track - using different color but solid line
                    if predicted_lats:
                        fig.add_trace(go.Scattermap(
                            lat=predicted_lats,
                            lon=predicted_lons,
                            mode='lines+markers',
                            name='Predicted Track',
                            line=dict(color='red', width=4),  # Removed 'dash' property
                            marker=dict(size=8, color='red', symbol='diamond')
                        ))
                    
                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            zoom=10,
                            center=dict(lat=historical_lats[-1], lon=historical_lons[-1])
                        ),
                        height=500,
                        margin={"r":0,"t":0,"l":0,"b":0},
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional analysis
                    st.markdown("<br>", unsafe_allow_html=True)  # Add vertical spacing
                    st.divider()  # Add a visual separator
                    st.markdown("<h3>📊 Movement Analysis</h3>", unsafe_allow_html=True)
                    if len(predictions) > 0:
                        start_lat = observations[-1]['latitude']
                        start_lon = observations[-1]['longitude']
                        end_lat = predictions[-1]['latitude']
                        end_lon = predictions[-1]['longitude']
                        
                        col_analysis1, col_analysis2 = st.columns(2)
                        with col_analysis1:
                            st.metric("Latitude Change", f"{(end_lat - start_lat):.4f}°")
                            st.metric("Longitude Change", f"{(end_lon - start_lon):.4f}°")
                        with col_analysis2:
                            avg_sog = sum(pred['sog'] for pred in predictions) / len(predictions)
                            avg_cog = sum(pred['cog'] for pred in predictions) / len(predictions)
                            st.metric("Average Speed", f"{avg_sog:.1f} knots")
                            st.metric("Average Course", f"{avg_cog:.1f}°")
                    
                else:
                    st.error("❌ Failed to generate predictions")

    # Footer
    st.markdown("---")
    st.markdown("### 📍 Live API Endpoint")
    st.code(f"{API_URL}")
    st.markdown("""
    **Available Endpoints:**
    - `GET /health` - API health check
    - `GET /model/info` - Model information  
    - `POST /predict` - Make trajectory predictions
    """)

if __name__ == "__main__":
    main()
