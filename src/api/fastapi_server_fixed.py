"""
Fixed FastAPI Production Server for SeaSeeAI
"""

import sys
import os

# Fix Python path for Windows multiprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import uvicorn

# Import from our project
try:
    from models.traisformer import TrAISformer
    from utils.production_config import get_config
    from data_processing.preprocessor import AISPreprocessor
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("seaseeai-api")

# Configuration
config = get_config()
config.validate()

# Global variables
model = None
preprocessor = AISPreprocessor()

# Pydantic models for API
class AISObservation(BaseModel):
    mmsi: int = Field(..., description="Maritime Mobile Service Identity")
    timestamp: str = Field(..., description="ISO format timestamp")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    sog: float = Field(..., ge=0, le=50, description="Speed Over Ground in knots")
    cog: float = Field(..., ge=0, le=360, description="Course Over Ground in degrees")

class PredictionRequest(BaseModel):
    observations: List[AISObservation] = Field(..., min_items=config.sequence_length)
    prediction_horizon: Optional[int] = Field(config.prediction_length, ge=1, le=10)

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float
    timestamp: str

# Model management
def load_model():
    """Load the trained model"""
    global model
    try:
        logger.info(f"Loading model from {config.model_checkpoint}")
        checkpoint = torch.load(config.model_checkpoint, map_location='cpu')
        model = TrAISformer(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# API lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SeaSeeAI API Server")
    
    # Load model
    if not load_model():
        raise RuntimeError("Failed to load model during startup")
    
    yield  # API is running
    
    # Shutdown
    logger.info("Shutting down SeaSeeAI API Server")

# Create FastAPI app
app = FastAPI(
    title="SeaSeeAI Trajectory Prediction API",
    description="Production API for vessel trajectory prediction using Transformer models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def prepare_sequence(observations: List[AISObservation]) -> np.ndarray:
    """Convert observations to model input sequence"""
    features = []
    for obs in observations:
        features.append([obs.latitude, obs.longitude, obs.sog, obs.cog])
    return np.array(features)

def predict_trajectory(sequence: np.ndarray, horizon: int) -> Dict[str, Any]:
    """Make prediction using the loaded model"""
    start_time = time.time()
    
    try:
        # Prepare input tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.cpu().numpy().squeeze()
        
        # Convert to readable format
        predictions = []
        for i in range(min(horizon, prediction.shape[0])):
            pred = {
                'step': i + 1,
                'latitude': float(prediction[i, 0]),
                'longitude': float(prediction[i, 1]),
                'sog': float(prediction[i, 2]),
                'cog': float(prediction[i, 3]),
                'timestamp': (datetime.now() + timedelta(hours=i)).isoformat()
            }
            predictions.append(pred)
        
        processing_time = time.time() - start_time
        
        # Calculate confidence (simplified)
        confidence = max(0.0, min(1.0, 1.0 - (processing_time * 0.1)))
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "SeaSeeAI Trajectory Prediction API", "status": "healthy"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        uptime=0.0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make trajectory prediction for a single vessel"""
    logger.info(f"Prediction request for {len(request.observations)} observations")
    
    # Validate sequence length
    if len(request.observations) < config.sequence_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least {config.sequence_length} observations, got {len(request.observations)}"
        )
    
    # Prepare sequence (use last N observations)
    sequence = prepare_sequence(request.observations[-config.sequence_length:])
    
    # Make prediction
    result = predict_trajectory(sequence, request.prediction_horizon)
    
    return PredictionResponse(
        predictions=result['predictions'],
        confidence=result['confidence'],
        processing_time=result['processing_time'],
        model_version="1.0.0"
    )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    checkpoint = torch.load(config.model_checkpoint, map_location='cpu')
    
    return {
        "model_type": "TrAISformer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_size": checkpoint['config']['input_size'],
        "sequence_length": config.sequence_length,
        "prediction_length": config.prediction_length,
        "checkpoint": config.model_checkpoint,
        "loaded": model is not None
    }

# Server startup with single worker to avoid multiprocessing issues
def start_server():
    """Start the production server with single worker"""
    uvicorn.run(
        "src.api.fastapi_server_fixed:app",
        host=config.api_host,
        port=config.api_port,
        workers=1,  # Single worker to avoid multiprocessing issues
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    start_server()
