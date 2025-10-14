# SeaSeeAI Project Summary

## Project Overview
SeaSeeAI is a maritime trajectory prediction system that uses AI to predict vessel movements based on AIS (Automatic Identification System) data. The project consists of two main components: a FastAPI-based prediction service and a Streamlit dashboard for visualization.

## Current State (v1.1.0-stable)
This stable version provides core functionality for maritime trajectory prediction with a working API and dashboard interface.

### Key Components

#### 1. API Service (`simple_api.py`)
- **Framework**: FastAPI
- **Core Functionality**: 
  - Trajectory prediction endpoints
  - Health monitoring
  - Model information
- **Endpoints**:
  - `GET /health` - System health check
  - `GET /` - Root endpoint with API information
  - `GET /model/info` - Model metadata and configuration
  - `POST /predict` - Trajectory prediction endpoint

#### 2. Dashboard (`simple_dashboard.py`)
- **Framework**: Streamlit
- **Features**:
  - Real-time vessel tracking visualization
  - Interactive prediction interface
  - Movement analysis with metrics
  - API status monitoring
  - Map-based trajectory visualization

### Data Structures

#### Input Format (Observation)
```python
{
    "mmsi": Optional[int],         # Maritime Mobile Service Identity
    "timestamp": str,              # ISO format timestamp
    "latitude": float,             # Current latitude
    "longitude": float,            # Current longitude
    "sog": float,                  # Speed Over Ground (knots)
    "cog": float                   # Course Over Ground (degrees)
}
```

#### Prediction Response
```python
{
    "predictions": [
        {
            "step": int,           # Prediction step number
            "latitude": float,     # Predicted latitude
            "longitude": float,    # Predicted longitude
            "sog": float,         # Predicted Speed Over Ground
            "cog": float,         # Predicted Course Over Ground
            "timestamp": str      # Predicted timestamp
        }
    ],
    "confidence": float,          # Prediction confidence score
    "processing_time": float,     # API processing time
    "model_version": str         # Model version identifier
}
```

### Deployment
- Production API URL: https://traisformer-seaseeai.onrender.com
- Docker support with automated health checks
- Environment variable configuration for flexible deployment

### Project Structure
```
traisformer-seaseeai/
├── simple_api.py           # Main API service
├── simple_dashboard.py     # Interactive dashboard
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Docker compose setup
├── tests/                 # Test suites
│   └── production/        # Production tests
├── models/               # Model files
├── data/                # Data directory
│   ├── processed/       # Processed datasets
│   └── raw/            # Raw AIS data
└── scripts/            # Utility scripts
```

## Technical Details

### API Configuration
- Default port: 8000
- CORS enabled
- JSON request/response format
- Automatic API documentation at `/docs`

### Dashboard Features
- Real-time API status monitoring
- Interactive map visualization using Plotly
- Movement analysis metrics:
  - Latitude/Longitude changes
  - Average speed and course
  - Prediction confidence visualization

### Dependencies
Core requirements:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
streamlit==1.28.0
pandas==2.0.3
plotly==5.15.0
requests==2.31.0
numpy==1.24.3
```

## Future Development Areas

### Potential Enhancements
1. **Model Improvements**
   - Integration of more sophisticated prediction models
   - Support for different vessel types
   - Uncertainty estimation in predictions

2. **Dashboard Enhancements**
   - Advanced filtering options
   - Historical data analysis
   - Batch prediction support
   - Custom visualization options

3. **API Extensions**
   - Batch processing endpoints
   - Weather data integration
   - Model retraining endpoints
   - Performance metrics API

4. **System Features**
   - User authentication
   - Data validation pipeline
   - Real-time data streaming
   - Model version control

### Technical Considerations
- Current API response time target: < 1 second
- Prediction horizon: configurable, default 5 steps
- Minimum required observations: 10
- Map visualization uses OpenStreetMap base layer

## Development Guidelines

### Code Structure
- Use FastAPI for API extensions
- Streamlit for dashboard components
- Maintain type hints and docstrings
- Follow existing error handling patterns

### Testing
- API endpoints have existing test coverage
- Use pytest for new test cases
- Include both unit and integration tests
- Test both API and dashboard components

### Documentation
- Update API documentation for new endpoints
- Maintain consistent docstring format
- Document configuration changes
- Update deployment guides as needed

## Current Limitations
1. Mock predictions in current version
2. Fixed prediction horizon
3. Basic error handling
4. Limited historical data support

This summary represents the stable v1.0.0 state of the project, providing a foundation for future development while maintaining existing functionality.