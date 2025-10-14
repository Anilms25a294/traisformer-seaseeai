# 🏗️ SeaSeeAI Project Architecture Diagram

## **System Overview**
```
┌─────────────────────────────────────────────────────────────────┐
│                        SeaSeeAI System                         │
├─────────────────────────────────────────────────────────────────┤
│  🌐 Web Interface    │  🔌 API Service    │  🤖 AI Models      │
│  ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────┐ │
│  │ Streamlit       │ │ │ FastAPI         │ │ │ TrAISformer     │ │
│  │ Dashboard       │ │ │ Server          │ │ │ (Transformer)   │ │
│  │                 │ │ │                 │ │ │                 │ │
│  │ • Ship Selection│ │ │ • /health       │ │ │ • Position      │ │
│  │ • Map Display   │ │ │ • /predict      │ │ │   Prediction    │ │
│  │ • Visualization │ │ │ • /model/info   │ │ │ • Multi-head    │ │
│  └─────────────────┘ │ └─────────────────┘ │ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  📊 Data Sources    │  🔄 Processing    │  💾 Storage         │
│  ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────┐ │
│  │ Real AIS Data   │ │ │ AISPreprocessor │ │ │ Processed Data  │ │
│  │ (1M+ records)   │ │ │                 │ │ │                 │ │
│  │                 │ │ │ • Data Cleaning │ │ │ • CSV Files     │ │
│  │ • MarineCadastre│ │ │ • Sequence      │ │ │ • Model Files   │ │
│  │ • Vessel Tracks │ │ │   Creation      │ │ │ • Configs       │ │
│  │ • Coordinates   │ │ │ • Normalization │ │ │                 │ │
│  └─────────────────┘ │ └─────────────────┘ │ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## **Module Structure**
```
src/
├── 📁 models/                    # AI/ML Models
│   ├── traisformer.py           # Main Transformer Model
│   ├── baseline.py              # LSTM & Basic Models
│   └── transformer_model.py     # Alternative Transformers
│
├── 📁 data_processing/           # Data Pipeline
│   ├── preprocessor.py          # Core Data Processing
│   ├── real_data_processor.py   # Real AIS Data Handler
│   └── sample_data_generator.py # Synthetic Data Generator
│
├── 📁 training/                  # Model Training
│   ├── working_train.py         # Working Training Script
│   ├── train_traisformer.py     # TrAISformer Training
│   └── compare_models.py        # Model Comparison
│
├── 📁 evaluation/                # Model Evaluation
│   ├── evaluate_model.py        # General Evaluation
│   └── evaluate_real_data.py    # Real Data Evaluation
│
├── 📁 api/                       # API Services
│   ├── fastapi_server.py        # Main API Server
│   └── fastapi_server_fixed.py  # Fixed API Version
│
├── 📁 dashboard/                 # Web Interface
│   └── streamlit_app.py         # Advanced Dashboard
│
├── 📁 inference/                 # Real-time Prediction
│   └── real_time_predictor.py   # Live Prediction Handler
│
├── 📁 utils/                     # Utilities
│   ├── config.py                # Configuration
│   └── production_config.py     # Production Settings
│
└── 📁 analysis/                  # Data Analysis
    └── diagnose_data.py         # Data Diagnostics
```

## **Data Flow**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw AIS Data  │───▶│  Data Cleaning  │───▶│  Sequence       │
│                 │    │                 │    │  Creation       │
│ • 1M+ records   │    │ • Validation    │    │ • Time series   │
│ • Coordinates   │    │ • Filtering     │    │ • Training      │
│ • Vessel info   │    │ • Normalization │    │   sequences     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Training │◀───│  Data Splitting │◀───│  Feature        │
│                 │    │                 │    │  Engineering    │
│ • TrAISformer   │    │ • Train/Val/Test│    │ • Scaling       │
│ • LSTM          │    │ • Cross-val     │    │ • Encoding      │
│ • Baseline      │    │ • Time series   │    │ • Augmentation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Serving  │◀───│  Model          │◀───│  Evaluation     │
│                 │    │  Deployment     │    │                 │
│ • API Endpoints │    │ • Model Loading │    │ • Metrics       │
│ • Predictions   │    │ • Caching       │    │ • Validation    │
│ • Real-time     │    │ • Monitoring    │    │ • Testing       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **API Endpoints**
```
FastAPI Server (Port 8000)
├── GET  /health          # Health check
├── GET  /                # Root endpoint
├── GET  /model/info      # Model information
└── POST /predict         # Trajectory prediction
    └── Input: AIS observations
    └── Output: Predicted trajectory
```

## **Dashboard Features**
```
Streamlit Dashboard (Port 8501)
├── 🚢 Vessel Selection
│   ├── Dropdown menu
│   ├── Search functionality
│   └── Vessel information
├── 🗺️ Map Visualization
│   ├── Historical tracks
│   ├── Predicted trajectories
│   └── Interactive maps
├── 📊 Analysis
│   ├── Prediction metrics
│   ├── Confidence scores
│   └── Performance data
└── ⚙️ Configuration
    ├── Prediction horizon
    ├── Data source selection
    └── Model parameters
```

## **Deployment Architecture**
```
Production Deployment
├── 🌐 Load Balancer (Nginx)
├── 🔌 API Servers (FastAPI)
├── 📊 Dashboard (Streamlit)
├── 💾 Database (PostgreSQL)
├── 🗄️ Cache (Redis)
├── 📈 Monitoring (Prometheus)
└── 📊 Visualization (Grafana)
```

## **Key Technologies**
```
Frontend:     Streamlit, Plotly, HTML/CSS
Backend:      FastAPI, Uvicorn, Python
AI/ML:        PyTorch, Transformers, LSTM
Data:         Pandas, NumPy, SciPy
Visualization: Plotly, Matplotlib
Deployment:   Docker, Docker Compose
Monitoring:   Prometheus, Grafana
Database:     PostgreSQL, Redis
```

This architecture provides a complete maritime trajectory prediction system with real-time capabilities, scalable deployment, and comprehensive monitoring.
