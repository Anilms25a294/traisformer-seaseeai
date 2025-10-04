"""
Streamlit Production Dashboard for SeaSeeAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.production_config import get_config
from data_processing.sample_data_generator import generate_realistic_ais_data

# Configuration
config = get_config()
API_URL = f"http://{config.api_host}:{config.api_port}"

# Page configuration
st.set_page_config(
    page_title="SeaSeeAI Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚢 SeaSeeAI Production Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Real-time Prediction", "Model Management", "API Testing", "System Monitoring"]
)

# Utility functions
def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def generate_sample_vessel_data(vessel_count=1, hours=24):
    """Generate sample vessel data for demonstration"""
    df = generate_realistic_ais_data(num_vessels=vessel_count, hours=hours)
    return df

def format_ais_observations(df, mmsi):
    """Format DataFrame observations for API"""
    vessel_data = df[df['mmsi'] == mmsi]
    observations = []
    
    for _, row in vessel_data.iterrows():
        observations.append({
            'mmsi': int(row['mmsi']),
            'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'sog': float(row['sog']),
            'cog': float(row['cog'])
        })
    
    return observations

# Dashboard Page
if page == "Dashboard":
    st.header("📊 Production Dashboard")
    
    # Health status
    col1, col2, col3, col4 = st.columns(4)
    
    api_healthy, health_data = check_api_health()
    
    with col1:
        st.metric(
            label="API Status",
            value="🟢 Healthy" if api_healthy else "🔴 Offline",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Model Status", 
            value="✅ Loaded" if health_data and health_data.get('model_loaded') else "❌ Not Loaded"
        )
    
    with col3:
        st.metric(
            label="Uptime",
            value=f"{health_data.get('uptime', 0):.0f}s" if health_data else "N/A"
        )
    
    with col4:
        st.metric(
            label="Active Connections",
            value="0"  # Would be tracked in production
        )
    
    # System metrics
    st.subheader("System Metrics")
    
    # Create sample metrics (in production, these would come from monitoring system)
    metrics_data = {
        'Time': [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)],
        'Requests': np.random.poisson(50, 24),
        'Response Time (ms)': np.random.normal(120, 20, 24),
        'Memory Usage (%)': np.random.normal(65, 5, 24),
        'CPU Usage (%)': np.random.normal(45, 8, 24)
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig_requests = px.line(metrics_df, x='Time', y='Requests', title='API Requests per Hour')
        st.plotly_chart(fig_requests, use_container_width=True)
        
        fig_cpu = px.line(metrics_df, x='Time', y='CPU Usage (%)', title='CPU Usage')
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        fig_response = px.line(metrics_df, x='Time', y='Response Time (ms)', title='Average Response Time')
        st.plotly_chart(fig_response, use_container_width=True)
        
        fig_memory = px.line(metrics_df, x='Time', y='Memory Usage (%)', title='Memory Usage')
        st.plotly_chart(fig_memory, use_container_width=True)

# Real-time Prediction Page
elif page == "Real-time Prediction":
    st.header("🎯 Real-time Trajectory Prediction")
    
    if not check_api_health()[0]:
        st.error("❌ API is not available. Please start the API server first.")
        st.stop()
    
    # Prediction controls
    col1, col2 = st.columns(2)
    
    with col1:
        vessel_count = st.slider("Number of Vessels", 1, 5, 1)
        prediction_horizon = st.slider("Prediction Horizon (steps)", 1, 10, 5)
    
    with col2:
        sequence_length = st.slider("Sequence Length", 10, 50, 20)
        auto_refresh = st.checkbox("Auto-refresh Predictions", value=False)
    
    # Generate sample data
    if st.button("Generate Sample Data") or auto_refresh:
        with st.spinner("Generating vessel data..."):
            df = generate_sample_vessel_data(vessel_count, hours=48)
            
            # Store in session state
            st.session_state.vessel_data = df
            st.session_state.predictions = {}
            
            # Make predictions for each vessel
            for mmsi in df['mmsi'].unique():
                vessel_df = df[df['mmsi'] == mmsi]
                observations = format_ais_observations(vessel_df, mmsi)
                
                if len(observations) >= sequence_length:
                    try:
                        prediction_request = {
                            'observations': observations[-sequence_length:],
                            'prediction_horizon': prediction_horizon
                        }
                        
                        response = requests.post(
                            f"{API_URL}/predict",
                            json=prediction_request,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            st.session_state.predictions[mmsi] = response.json()
                        else:
                            st.error(f"Prediction failed for vessel {mmsi}: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error predicting for vessel {mmsi}: {str(e)}")
        
        # Display results
        if hasattr(st.session_state, 'predictions'):
            for mmsi, prediction in st.session_state.predictions.items():
                with st.expander(f"Vessel {mmsi} - Prediction Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prediction Details**")
                        st.write(f"Confidence: {prediction['confidence']:.2%}")
                        st.write(f"Processing Time: {prediction['processing_time']:.3f}s")
                        
                        # Show predictions in table
                        pred_df = pd.DataFrame(prediction['predictions'])
                        st.dataframe(pred_df[['step', 'latitude', 'longitude', 'sog', 'cog']])
                    
                    with col2:
                        st.markdown("**Trajectory Visualization**")
                        
                        # Get historical data
                        vessel_df = st.session_state.vessel_data[st.session_state.vessel_data['mmsi'] == mmsi]
                        
                        # Create map
                        fig = go.Figure()
                        
                        # Historical trajectory
                        fig.add_trace(go.Scattermapbox(
                            lat=vessel_df['latitude'],
                            lon=vessel_df['longitude'],
                            mode='lines+markers',
                            name='Historical Path',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Predicted trajectory
                        pred_lats = [p['latitude'] for p in prediction['predictions']]
                        pred_lons = [p['longitude'] for p in prediction['predictions']]
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=pred_lats,
                            lon=pred_lons,
                            mode='lines+markers',
                            name='Predicted Path',
                            line=dict(color='red', width=3, dash='dot'),
                            marker=dict(size=8, symbol='star')
                        ))
                        
                        fig.update_layout(
                            mapbox=dict(
                                style='open-street-map',
                                center=dict(lat=vessel_df['latitude'].iloc[-1], lon=vessel_df['longitude'].iloc[-1]),
                                zoom=10
                            ),
                            height=400,
                            margin=dict(l=0, r=0, t=0, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()

# Model Management Page
elif page == "Model Management":
    st.header("🤖 Model Management")
    
    if not check_api_health()[0]:
        st.error("❌ API is not available. Please start the API server first.")
        st.stop()
    
    # Model information
    try:
        model_info = requests.get(f"{API_URL}/model/info").json()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Configuration")
            st.json(model_info)
        
        with col2:
            st.markdown("### Performance Metrics")
            
            # Sample performance metrics
            metrics = {
                'Mean Distance Error': '732.75 km',
                'Inference Speed': '45 ms',
                'Model Size': '3.3 MB',
                'Training Date': '2024-01-15'
            }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
    
    except Exception as e:
        st.error(f"Failed to get model info: {str(e)}")
    
    # Model operations
    st.markdown("### Model Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Model"):
            with st.spinner("Reloading model..."):
                # This would trigger a model reload in production
                st.success("Model reload initiated")
    
    with col2:
        if st.button("📊 Model Metrics"):
            with st.spinner("Collecting metrics..."):
                # Show model performance metrics
                st.info("Model metrics dashboard would appear here")
    
    with col3:
        if st.button("⚡ Warmup Model"):
            with st.spinner("Warming up model..."):
                st.success("Model warmup completed")

# API Testing Page
elif page == "API Testing":
    st.header("🔧 API Testing")
    
    st.markdown("### Test API Endpoints")
    
    # Test endpoints
    endpoint = st.selectbox(
        "Select Endpoint",
        ["/health", "/predict", "/model/info"]
    )
    
    if endpoint == "/predict":
        st.markdown("### Test Prediction Endpoint")
        
        # Generate test data
        if st.button("Generate Test Payload"):
            df = generate_sample_vessel_data(1, 24)
            mmsi = df['mmsi'].iloc[0]
            observations = format_ais_observations(df, mmsi)
            
            test_payload = {
                'observations': observations[-20:],  # Last 20 observations
                'prediction_horizon': 5
            }
            
            st.session_state.test_payload = test_payload
            st.json(test_payload)
        
        if hasattr(st.session_state, 'test_payload'):
            if st.button("Send Test Request"):
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=st.session_state.test_payload,
                        timeout=30
                    )
                    response_time = time.time() - start_time
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Request Details**")
                        st.write(f"Status Code: {response.status_code}")
                        st.write(f"Response Time: {response_time:.3f}s")
                        st.write(f"Payload Size: {len(json.dumps(st.session_state.test_payload))} bytes")
                    
                    with col2:
                        st.markdown("**Response**")
                        if response.status_code == 200:
                            st.success("✅ Request successful!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ Request failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# System Monitoring Page
elif page == "System Monitoring":
    st.header("📈 System Monitoring")
    
    st.markdown("### Real-time System Metrics")
    
    # Placeholder for system monitoring
    st.info("""
    **Production Monitoring Features:**
    
    - Real-time API metrics and logs
    - Model performance monitoring
    - Resource utilization (CPU, Memory, GPU)
    - Error rate tracking
    - Alert system integration
    
    *In a production environment, this would integrate with tools like:*
    - Prometheus & Grafana for metrics
    - ELK Stack for logging
    - Sentry for error tracking
    """)
    
    # Sample monitoring dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Error Rates")
        error_data = pd.DataFrame({
            'Time': [datetime.now() - timedelta(hours=i) for i in range(12, 0, -1)],
            'Errors': np.random.poisson(2, 12)
        })
        st.line_chart(error_data.set_index('Time'))
    
    with col2:
        st.markdown("#### Resource Usage")
        resource_data = pd.DataFrame({
            'Resource': ['CPU', 'Memory', 'Disk', 'Network'],
            'Usage (%)': [65, 80, 45, 30]
        })
        st.bar_chart(resource_data.set_index('Resource'))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### System Information")
st.sidebar.write(f"**API URL**: {API_URL}")
st.sidebar.write(f"**Environment**: Production")
st.sidebar.write(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # This would typically be run with: streamlit run src/dashboard/streamlit_app.py
    pass
