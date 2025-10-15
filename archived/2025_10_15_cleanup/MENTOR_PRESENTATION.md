# SeaSeeAI — Mentor Presentation

## 1) What we built
- **SeaSeeAI** is a maritime trajectory prediction system that forecasts future vessel positions from AIS streams.
- It includes:
  - **API service** for predictions (FastAPI)
  - **Interactive dashboard** for visualization (Streamlit + Plotly)
  - **AI models** (TrAISformer + LSTM baselines) with training and evaluation pipelines
  - **Data processing** for real AIS datasets (MarineCadastre format)
  - **Deployment & monitoring** artifacts (Docker, Prometheus)

## 2) Why it matters (problem → impact)
- Maritime operators need situational awareness: predict where ships will be to optimize routing, port operations, and safety.
- Predictive trajectories help with ETA estimation, congestion avoidance, and early anomaly detection.

## 3) Data we use
- Primary dataset: `AIS_2024_12_311.csv` (1M+ rows, MarineCadastre US AIS)
  - Columns: `MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`, `Heading`, `VesselType`, etc.
- Processed/standardized to: `timestamp`, `latitude`, `longitude`, `sog`, `cog`, `mmsi` → saved as `data/real_ais/processed_ais_data.csv`.
- Synthetic data: `data/raw/sample_ais_data.csv` (quick demos, testing edge cases).

## 4) How the system works (end‑to‑end)
1. Ingest AIS CSV → standardize columns and clean invalid rows.
2. Create time‑series sequences (past N observations → predict next K steps).
3. Train models (baseline LSTM and TrAISformer transformer).
4. Serve predictions via FastAPI (`POST /predict`).
5. Visualize historical + predicted tracks on the dashboard (map + metrics).

### High‑level architecture
```
Streamlit Dashboard  ⇄  FastAPI Service  ⇄  Inference (loaded model)
        │                    │                  │
        └──────────────→ Data Processing / Sequences ←──────────────┘
                                  │
                             Model Training
```

## 5) Modules and responsibilities
- `src/data_processing/`
  - `preprocessor.py`: cleaning, sequence creation
  - `real_data_processor.py`: MarineCadastre → internal schema + validation
  - `sample_data_generator.py`: synthetic tracks for demos
- `src/models/`
  - `traisformer.py`: transformer with positional encoding and multi‑head variant
  - `baseline.py`, `simple_lstm_model.py`: LSTM baselines
- `src/training/`
  - `train_traisformer.py`, `train_real_data.py`, `working_train.py`: training flows
  - `compare_models.py`: LSTM vs Transformer comparison
- `src/evaluation/`
  - `evaluate_*`: metrics, plots, reports
- `src/api/`
  - `fastapi_server.py`: production‑style API with schemas/validation
- `src/dashboard/`
  - `streamlit_app.py`: advanced dashboard (the root `simple_dashboard.py` is a streamlined variant)
- `src/inference/`
  - `real_time_predictor.py`: wraps model for online inference

## 6) API and dashboard (demo)
- Live API (Render): `https://traisformer-seaseeai.onrender.com`
  - `GET /health` — service health and uptime
  - `GET /model/info` — model metadata
  - `POST /predict` — returns multi‑step trajectory predictions
- Local dashboard: `streamlit run simple_dashboard.py`
  - Choose data source → send observations → visualize historical vs predicted tracks

## 7) Results (current state)
- Visual artifacts in repo:
  - `model_comparison.png` — LSTM vs Transformer training curves
  - `working_training.png`, `real_data_training.png` — training progress
  - `working_model_evaluation.png` — evaluation snapshot
- Report: `model_evaluation_report.txt`
  - TrAISformer outperforms LSTM baseline on internal metrics

## 8) Requirements (runtime)
From `requirements.txt`:
- fastapi 0.104.1, uvicorn 0.24.0
- streamlit 1.28.0, plotly 5.15.0
- pandas 2.0.3, numpy 1.24.3, scipy 1.10.1
- torch 2.1.0
System: Python 3.9+, ≥4GB RAM, ≥10GB disk. Optional: CUDA GPU for faster training.

## 9) What the mentor should look at first
- Code tour: `PROJECT_STRUCTURE_GUIDE.md` and `ARCHITECTURE_DIAGRAM.md`
- Data: `AIS_2024_12_311.csv` → `data/real_ais/processed_ais_data.csv`
- Model: `src/models/traisformer.py`
- API: `simple_api.py` (demo) or `src/api/fastapi_server_fixed.py` (full)
- Dashboard: `simple_dashboard.py`

## 10) Roadmap (near term)
- Replace mock API predictions with the trained TrAISformer model
- Dashboard: ship selection from real AIS, maritime‑only coordinate validation
- Batch prediction + confidence intervals
- Monitoring (Prometheus) dashboards for latency and error rates

## 11) Risks and mitigations
- Large AIS files → memory constraints → chunked processing + sampling
- Data quality (on‑land points) → geographic validation + filters
- Model drift → periodic retraining on latest data

## 12) Demo script (talk‑track)
1. Show `GET /health` and `GET /model/info` on the live API.
2. Load the dashboard, pick a vessel, set horizon, request predictions.
3. Explain how the model consumes the last N points to output K future points.
4. Open `model_comparison.png` and summarize why TrAISformer is preferred.
5. Outline the roadmap to plug the trained model into the API for end‑to‑end real predictions.

---
Questions welcome. We can drill down into any module or metric as needed.
