# 🚢 SeaSeeAI Project Structure & Documentation

## 📋 **Project Overview**
SeaSeeAI is a maritime trajectory prediction system that uses AI to predict vessel movements based on AIS (Automatic Identification System) data. The project consists of multiple components working together to provide real-time vessel trajectory predictions.

---

## 🏗️ **Project Architecture**

```
SeaSeeAI Project
├── 🌐 Web Interface (Streamlit Dashboard)
├── 🔌 API Service (FastAPI)
├── 🤖 AI Models (TrAISformer, LSTM)
├── 📊 Data Processing Pipeline
├── 📈 Model Training & Evaluation
└── 🚀 Deployment & Monitoring
```

---

## 📁 **Root Directory Files**

### **Core Application Files**
| File | Purpose | Description |
|------|---------|-------------|
| `simple_api.py` | **Main API Server** | FastAPI-based prediction service with health checks and prediction endpoints |
| `simple_dashboard.py` | **Main Dashboard** | Streamlit web interface for vessel trajectory visualization |
| `simple_demo.py` | **Demo Script** | Quick demonstration of the system capabilities |

### **Data Files**
| File | Purpose | Description |
|------|---------|-------------|
| `AIS_2024_12_311.csv` | **Real AIS Data** | 1M+ real maritime records from MarineCadastre (main dataset) |
| `data/raw/sample_ais_data.csv` | **Sample Data** | Generated synthetic AIS data for testing |
| `data/real_ais/processed_ais_data.csv` | **Processed Data** | Cleaned and standardized real AIS data |

### **Model Files** (`models/` directory)
| File | Purpose | Description |
|------|---------|-------------|
| `best_traisformer.pth` | **Best TrAISformer** | Highest performing transformer model |
| `final_traisformer.pth` | **Final TrAISformer** | Production-ready transformer model |
| `working_model.pth` | **Working Model** | Simple neural network for basic predictions |
| `lstm_real_data.pth` | **LSTM Real Data** | LSTM model trained on real AIS data |
| `baseline_lstm.pth` | **Baseline LSTM** | Basic LSTM model for comparison |

### **Configuration Files**
| File | Purpose | Description |
|------|---------|-------------|
| `requirements.txt` | **Dependencies** | Python package requirements |
| `pyproject.toml` | **Build Config** | Project build and packaging configuration |
| `Dockerfile` | **Container Config** | Docker container setup |
| `docker-compose.yml` | **Multi-Service** | Docker services orchestration |
| `render.yaml` | **Deployment** | Render.com deployment configuration |

### **Documentation Files**
| File | Purpose | Description |
|------|---------|-------------|
| `PROJECT_SUMMARY.md` | **Project Overview** | Comprehensive project status and features |
| `DEPLOYMENT_GUIDE.md` | **Deployment Guide** | Production deployment instructions |
| `README.md` | **Main Documentation** | Project introduction and setup |

---

## 📂 **Source Code Structure** (`src/` directory)

### **1. Models Module** (`src/models/`)
**Purpose**: AI/ML model implementations

| File | Purpose | Description |
|------|---------|-------------|
| `traisformer.py` | **TrAISformer Model** | Main transformer-based model for trajectory prediction |
| `baseline.py` | **Baseline Models** | Simple LSTM and basic neural network models |
| `transformer_model.py` | **Transformer Variants** | Alternative transformer implementations |
| `simple_lstm_model.py` | **LSTM Implementation** | Long Short-Term Memory network for sequences |

**Key Classes:**
- `TrAISformer`: Main transformer model with positional encoding
- `MultiHeadTraisformer`: Enhanced version with multiple prediction heads
- `SimpleLSTMModel`: Basic LSTM for trajectory prediction

### **2. Data Processing Module** (`src/data_processing/`)
**Purpose**: Data loading, cleaning, and preprocessing

| File | Purpose | Description |
|------|---------|-------------|
| `preprocessor.py` | **Main Preprocessor** | Core data cleaning and sequence creation |
| `real_data_processor.py` | **Real Data Handler** | Processes real AIS data from MarineCadastre |
| `sample_data_generator.py` | **Data Generator** | Creates synthetic AIS data for testing |
| `ais_data_loader.py` | **Data Loader** | Loads and manages AIS datasets |
| `advanced_preprocessor.py` | **Advanced Processing** | Enhanced preprocessing with more features |

**Key Classes:**
- `AISPreprocessor`: Main data cleaning and sequence creation
- `RealAISProcessor`: Handles real maritime data processing
- `AISDataLoader`: Manages data loading operations

### **3. Training Module** (`src/training/`)
**Purpose**: Model training scripts and utilities

| File | Purpose | Description |
|------|---------|-------------|
| `working_train.py` | **Working Training** | Simple training script that works reliably |
| `train_traisformer.py` | **TrAISformer Training** | Trains the main transformer model |
| `train_real_data.py` | **Real Data Training** | Trains models on real AIS data |
| `compare_models.py` | **Model Comparison** | Compares different model performances |
| `hyperparameter_tuner.py` | **Hyperparameter Tuning** | Optimizes model parameters |

**Training Scripts:**
- `train_baseline.py`: Trains baseline LSTM models
- `train_real_data_simple.py`: Simplified real data training
- `train_real_minimal.py`: Minimal training for quick testing

### **4. Evaluation Module** (`src/evaluation/`)
**Purpose**: Model evaluation and performance metrics

| File | Purpose | Description |
|------|---------|-------------|
| `evaluate_model.py` | **Model Evaluator** | General model evaluation framework |
| `evaluate_real_data.py` | **Real Data Evaluation** | Evaluates models on real AIS data |
| `model_evaluator.py` | **Evaluation Metrics** | Calculates accuracy and error metrics |

**Evaluation Features:**
- Position error calculation (degrees and kilometers)
- Prediction confidence scoring
- Model performance comparison
- Visualization of results

### **5. API Module** (`src/api/`)
**Purpose**: API server implementations

| File | Purpose | Description |
|------|---------|-------------|
| `fastapi_server.py` | **Main API Server** | Complete FastAPI implementation |
| `fastapi_server_fixed.py` | **Fixed API Server** | Bug-fixed version of the API |

**API Features:**
- Health check endpoints
- Model information endpoints
- Prediction endpoints
- Error handling and validation

### **6. Dashboard Module** (`src/dashboard/`)
**Purpose**: Web interface implementations

| File | Purpose | Description |
|------|---------|-------------|
| `streamlit_app.py` | **Streamlit App** | Advanced dashboard implementation |

### **7. Inference Module** (`src/inference/`)
**Purpose**: Real-time prediction capabilities

| File | Purpose | Description |
|------|---------|-------------|
| `real_time_predictor.py` | **Real-time Predictor** | Handles live prediction requests |

### **8. Utils Module** (`src/utils/`)
**Purpose**: Utility functions and configuration

| File | Purpose | Description |
|------|---------|-------------|
| `config.py` | **Configuration** | General project configuration |
| `production_config.py` | **Production Config** | Production environment settings |

### **9. Analysis Module** (`src/analysis/`)
**Purpose**: Data analysis and diagnostics

| File | Purpose | Description |
|------|---------|-------------|
| `diagnose_data.py` | **Data Diagnostics** | Analyzes data quality and patterns |

---

## 🧪 **Testing Files**

### **Test Scripts** (Root directory)
| File | Purpose | Description |
|------|---------|-------------|
| `test_ais_predictions.py` | **API Testing** | Tests prediction API endpoints |
| `test_all_datasets.py` | **Dataset Testing** | Tests all available datasets |
| `test_fixes.py` | **Fix Testing** | Tests bug fixes and improvements |
| `test_model_fix.py` | **Model Testing** | Tests model functionality |

### **Production Tests** (`tests/production/`)
| File | Purpose | Description |
|------|---------|-------------|
| Production test suites for deployment validation |

---

## 📊 **Data Directory Structure**

### **Raw Data** (`data/raw/`)
- `sample_ais_data.csv`: Generated synthetic AIS data

### **Processed Data** (`data/processed/`)
- Processed and cleaned datasets

### **Real AIS Data** (`data/real_ais/`)
- `AIS_2024_12_311.csv`: Original real AIS data (1M+ records)
- `processed_ais_data.csv`: Cleaned and standardized real data

---

## 🚀 **Deployment Files**

### **Docker Configuration**
| File | Purpose | Description |
|------|---------|-------------|
| `Dockerfile` | **Container Build** | Docker image configuration |
| `docker-compose.yml` | **Multi-Service** | Orchestrates API and dashboard services |

### **Deployment Scripts** (`scripts/`)
| File | Purpose | Description |
|------|---------|-------------|
| `deploy_production.sh` | **Production Deploy** | Automated production deployment |
| `start_production.py` | **Production Start** | Starts production services |

### **Configuration** (`config/`)
| File | Purpose | Description |
|------|---------|-------------|
| `production.yaml` | **Production Config** | Production environment settings |

---

## 📈 **Visualization Files**

### **Generated Images**
| File | Purpose | Description |
|------|---------|-------------|
| `demo_training.png` | **Demo Training** | Training progress visualization |
| `model_comparison.png` | **Model Comparison** | Performance comparison charts |
| `real_data_training.png` | **Real Data Training** | Real data training progress |
| `working_training.png` | **Working Training** | Working model training results |

---

## 🔧 **Project Requirements**

### **Core Dependencies** (`requirements.txt`)
```
fastapi==0.104.1          # Web API framework
uvicorn[standard]==0.24.0  # ASGI server
streamlit==1.28.0         # Web dashboard
pandas==2.0.3             # Data manipulation
plotly==5.15.0            # Interactive visualizations
requests==2.31.0          # HTTP requests
numpy==1.24.3             # Numerical computing
torch==2.1.0              # Deep learning framework
scipy==1.10.1             # Scientific computing
```

### **System Requirements**
- **Python**: 3.9+
- **Memory**: 4GB RAM minimum
- **Storage**: 10GB disk space
- **OS**: Windows, Linux, macOS

### **Optional Dependencies**
- **GPU**: CUDA-compatible GPU for faster training
- **Docker**: For containerized deployment
- **Jupyter**: For notebook development

---

## 🎯 **Key Features**

### **1. AI Models**
- **TrAISformer**: Transformer-based trajectory prediction
- **LSTM Models**: Long Short-Term Memory networks
- **Baseline Models**: Simple neural networks for comparison

### **2. Data Processing**
- **Real AIS Data**: 1M+ maritime records
- **Data Cleaning**: Automatic data validation and cleaning
- **Sequence Creation**: Time-series data preparation

### **3. Web Interface**
- **Interactive Dashboard**: Streamlit-based visualization
- **Real-time Predictions**: Live trajectory predictions
- **Map Visualization**: Geographic trajectory plotting

### **4. API Service**
- **REST API**: FastAPI-based prediction service
- **Health Monitoring**: System status and performance
- **Model Information**: Model metadata and configuration

### **5. Deployment**
- **Docker Support**: Containerized deployment
- **Production Ready**: Scalable production configuration
- **Monitoring**: System monitoring and logging

---

## 🚀 **Quick Start Guide**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv seaseeai_env
source seaseeai_env/bin/activate  # Linux/Mac
# or
seaseeai_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **2. Run Dashboard**
```bash
streamlit run simple_dashboard.py
```

### **3. Run API Server**
```bash
python simple_api.py
```

### **4. Train Models**
```bash
python src/training/working_train.py
```

---

## 📝 **Development Workflow**

1. **Data Processing**: Use `src/data_processing/` modules
2. **Model Training**: Use `src/training/` scripts
3. **Model Evaluation**: Use `src/evaluation/` tools
4. **API Development**: Modify `src/api/` files
5. **Dashboard Development**: Update `src/dashboard/` files
6. **Testing**: Use test scripts in root directory
7. **Deployment**: Use `scripts/` and `config/` files

---

This comprehensive guide provides a complete understanding of the SeaSeeAI project structure, making it easy to navigate and understand each component's purpose and functionality.
