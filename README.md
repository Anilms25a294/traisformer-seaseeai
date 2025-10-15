# SeaSeeAI - Maritime Trajectory Prediction

## Overview
SeaSeeAI is a maritime vessel trajectory prediction system that uses advanced AI models to predict vessel movements based on AIS (Automatic Identification System) data. The system provides both an API for predictions and an interactive dashboard for visualization.

## Key Features
- FastAPI-based prediction service
- Interactive Streamlit dashboard
- TrAISformer model for accurate predictions
- Docker containerization
- Production-ready deployment configurations

## Quick Start

### Using Docker
```bash
# Build and start the services
docker-compose up -d

# Access the services
API: http://localhost:8000
Dashboard: http://localhost:8501
```

### Manual Setup
1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API:
```bash
python simple_api.py
```

4. Start the dashboard:
```bash
streamlit run simple_dashboard.py
```

## Project Structure
- `simple_api.py` - Main API service
- `simple_dashboard.py` - Interactive dashboard
- `config/` - Configuration files
- `src/` - Source code
- `models/` - Model files
- `data/` - Data directory
- `docs/` - Documentation
- `tests/` - Test suites
- `notebooks/` - Jupyter notebooks
- `scripts/` - Utility scripts

## Documentation
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `PROJECT_SUMMARY.md` - Detailed project overview
- `docs/` - Additional documentation

## Development
The project uses a stable release branch strategy:
- `main` - Development branch
- `release/v1.1.0-stable` - Current stable release

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
MIT License

## Contact
Maintainer: Anil M S
