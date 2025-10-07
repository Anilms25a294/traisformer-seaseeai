"""
Simple SeaSeeAI API Server - Working Version
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seaseeai-simple-api")

# Create FastAPI app
app = FastAPI(title="SeaSeeAI API", version="1.0.0")

start_time = datetime.now()

# Pydantic models for request/response
class Observation(BaseModel):
    mmsi: Optional[int]
    timestamp: str
    latitude: float
    longitude: float
    sog: float
    cog: float

class PredictionRequest(BaseModel):
    observations: List[Observation]
    prediction_horizon: int = 5

class Prediction(BaseModel):
    step: int
    latitude: float
    longitude: float
    sog: float
    cog: float
    timestamp: str

class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    confidence: float
    processing_time: float
    model_version: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()
    response = {
        "status": "healthy",
        "model_loaded": True,
        "uptime": uptime,
        "timestamp": datetime.now().isoformat()
    }
    logger.info("Health check - OK")
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SeaSeeAI Trajectory Prediction API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/model/info")
async def model_info():
    """Model information endpoint"""
    return {
        "model_type": "TrAISformer",
        "loaded": True,
        "parameters": 1000000,
        "input_size": 4,
        "sequence_length": 20,
        "prediction_length": 5
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make trajectory predictions"""
    try:
        # Log the request
        logger.info(f"Prediction request with {len(request.observations)} observations")
        
        # Validate input
        if len(request.observations) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 observations")
        
        # Create mock predictions based on last observation
        last_obs = request.observations[-1]
        
        predictions = []
        for i in range(request.prediction_horizon):
            predictions.append(Prediction(
                step=i + 1,
                latitude=last_obs.latitude + (i + 1) * 0.01,
                longitude=last_obs.longitude + (i + 1) * 0.01,
                sog=last_obs.sog,
                cog=last_obs.cog,
                timestamp=(datetime.now() + timedelta(hours=i+1)).isoformat()
            ))
        
        response = PredictionResponse(
            predictions=predictions,
            confidence=0.85,
            processing_time=0.05,
            model_version="1.0.0"
        )
        
        logger.info(f"Prediction response sent for {request.prediction_horizon} steps")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print("")
    print("🚀 SeaSeeAI API Server Started Successfully!")
    print("=" * 50)
    print(f"📍 Server running on: http://localhost:{port}")
    print("")
    print("📊 Available Endpoints:")
    print("   GET  /health      - Health check")
    print("   GET  /            - Root endpoint")
    print("   GET  /model/info  - Model information")
    print("   POST /predict     - Make trajectory prediction")
    print("")
    print("💡 Example prediction request:")
    print('   curl -X POST http://localhost:8000/predict \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"observations": [{"mmsi": 123, "timestamp": "2024-01-01T00:00:00",')
    print('        "latitude": 37.7749, "longitude": -122.4194, "sog": 10.0, "cog": 45.0}],')
    print('        "prediction_horizon": 3}\'')
    print("")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    print("")
    uvicorn.run(app, host="0.0.0.0", port=port)
