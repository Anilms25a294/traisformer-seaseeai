"""
Simple SeaSeeAI Dashboard
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="SeaSeeAI Dashboard",
    page_icon="🚢",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚢 SeaSeeAI Production Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Real-time Prediction", "API Testing"])

API_URL = "http://localhost:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def generate_sample_observations(count=20):
    """Generate sample AIS observations"""
    observations = []
    base_lat = 37.7749
    base_lon = -122.4194
    
    for i in range(count):
        observations.append({
            'mmsi': 123456789,
            'timestamp': (datetime.now() - timedelta(hours=count-i)).isoformat(),
            'latitude': base_lat + i * 0.001,
            'longitude': base_lon + i * 0.001,
            'sog': 10.0 + i * 0.1,
            'cog': 45.0 + i * 1.0
        })
    
    return observations

if page == "Dashboard":
    st.header("📊 Production Dashboard")
    
    # Health status
    col1, col2, col3 = st.columns(3)
    
    api_healthy, health_data = check_api_health()
    
    with col1:
        st.metric("API Status", "🟢 Healthy" if api_healthy else "🔴 Offline")
    
    with col2:
        st.metric("Model Status", "✅ Loaded" if health_data and health_data.get('model_loaded') else "❌ Not Loaded")
    
    with col3:
        st.metric("Uptime", f"{health_data.get('uptime', 0):.0f}s" if health_data else "N/A")
    
    # Quick prediction demo
    st.header("🎯 Quick Prediction Demo")
    
    if st.button("Generate Sample Prediction"):
        with st.spinner("Making prediction..."):
            try:
                sample_obs = generate_sample_observations(20)
                prediction_request = {
                    'observations': sample_obs,
                    'prediction_horizon': 5
                }
                
                response = requests.post(f"{API_URL}/predict", json=prediction_request, timeout=10)
                
                if response.status_code == 200:
                    prediction_data = response.json()
                    st.success("✅ Prediction successful!")
                    
                    # Display predictions
                    pred_df = pd.DataFrame(prediction_data['predictions'])
                    st.dataframe(pred_df)
                    
                    # Show on map
                    st.subheader("📍 Trajectory Map")
                    
                    fig = go.Figure()
                    
                    # Historical path
                    hist_lats = [obs['latitude'] for obs in sample_obs]
                    hist_lons = [obs['longitude'] for obs in sample_obs]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=hist_lats,
                        lon=hist_lons,
                        mode='lines+markers',
                        name='Historical Path',
                        line=dict(color='blue', width=4),
                        marker=dict(size=8)
                    ))
                    
                    # Predicted path
                    pred_lats = [p['latitude'] for p in prediction_data['predictions']]
                    pred_lons = [p['longitude'] for p in prediction_data['predictions']]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=pred_lats,
                        lon=pred_lons,
                        mode='lines+markers',
                        name='Predicted Path',
                        line=dict(color='red', width=4),
                        marker=dict(size=10, symbol='star')
                    ))
                    
                    fig.update_layout(
                        mapbox=dict(
                            style='open-street-map',
                            center=dict(lat=hist_lats[-1], lon=hist_lons[-1]),
                            zoom=12
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif page == "Real-time Prediction":
    st.header("🎯 Real-time Prediction")
    
    if not check_api_health()[0]:
        st.error("❌ API is not available. Please start the API server first.")
        st.info("💡 Run: `python simple_api.py` in another terminal")
    else:
        st.success("✅ API is ready!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_observations = st.slider("Number of Observations", 10, 50, 20)
            prediction_horizon = st.slider("Prediction Horizon", 1, 10, 5)
        
        if st.button("Generate and Predict"):
            with st.spinner("Generating sample data and making prediction..."):
                observations = generate_sample_observations(num_observations)
                
                st.subheader("Sample Observations")
                st.json(observations[:5])  # Show first 5
                
                prediction_request = {
                    'observations': observations,
                    'prediction_horizon': prediction_horizon
                }
                
                try:
                    response = requests.post(f"{API_URL}/predict", json=prediction_request, timeout=10)
                    
                    if response.status_code == 200:
                        prediction_data = response.json()
                        st.success("✅ Prediction successful!")
                        
                        # Show prediction details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Confidence", f"{prediction_data['confidence']:.1%}")
                            st.metric("Processing Time", f"{prediction_data['processing_time']:.3f}s")
                        
                        with col2:
                            st.metric("Predictions Generated", len(prediction_data['predictions']))
                            st.metric("Model Version", prediction_data['model_version'])
                        
                        # Show predictions table
                        st.subheader("📋 Predictions")
                        pred_df = pd.DataFrame(prediction_data['predictions'])
                        st.dataframe(pred_df)
                        
                    else:
                        st.error(f"❌ Prediction failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif page == "API Testing":
    st.header("🔧 API Testing")
    
    st.info("""
    **API Endpoints:**
    - `GET /health` - Health check
    - `GET /` - Root endpoint  
    - `GET /model/info` - Model information
    - `POST /predict` - Make trajectory prediction
    """)
    
    if st.button("Test API Health"):
        healthy, data = check_api_health()
        if healthy:
            st.success("✅ API is healthy!")
            st.json(data)
        else:
            st.error("❌ API is not responding")
    
    if st.button("Test Model Info"):
        try:
            response = requests.get(f"{API_URL}/model/info", timeout=5)
            if response.status_code == 200:
                st.success("✅ Model info retrieved!")
                st.json(response.json())
            else:
                st.error(f"❌ Failed to get model info: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### System Information")
st.sidebar.write(f"**API URL**: {API_URL}")
st.sidebar.write(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    pass
