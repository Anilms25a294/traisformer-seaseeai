"""
FastAPI Production Server for SeaSeeAI
"""

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
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.traisformer import TrAISformer
from utils.production_config import get_config
from data_processing.preprocessor import AISPreprocessor

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

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., max_items=config.max_batch_size)

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processing_time: float

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

def warmup_model():
    """Warm up the model with dummy data"""
    if model is None:
        return False
    
    try:
        logger.info("Warming up model...")
        dummy_input = torch.randn(1, config.sequence_length, 4)
        with torch.no_grad():
            _ = model(dummy_input)
        logger.info("Model warmup completed")
        return True
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        return False

# API lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SeaSeeAI API Server")
    
    # Load model
    if not load_model():
        raise RuntimeError("Failed to load model during startup")
    
    # Warm up model if configured
    if config.model_warmup:
        warmup_model()
    
    yield  # API is running
    
    # Shutdown
    logger.info("Shutting down SeaSeeAI API Server")
    # Cleanup resources if needed

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
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware (placeholder for API key validation)
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if config.api_key_required and not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    # Add API key validation logic here
    return True

# Rate limiting (simplified version)
request_times = {}

async def check_rate_limit(client_id: str = "default"):
    if client_id not in request_times:
        request_times[client_id] = []
    
    now = time.time()
    window_start = now - config.rate_limit_window
    
    # Remove old requests
    request_times[client_id] = [t for t in request_times[client_id] if t > window_start]
    
    if len(request_times[client_id]) >= config.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_times[client_id].append(now)
    return True

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
        
        # Calculate confidence (simplified - could be based on prediction variance)
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
        uptime=0.0,  # Would track actual uptime in production
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: bool = Depends(verify_api_key),
    rate_limit: bool = Depends(check_rate_limit)
):
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

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    api_key: bool = Depends(verify_api_key),
    rate_limit: bool = Depends(check_rate_limit)
):
    """Make batch predictions for multiple vessels"""
    logger.info(f"Batch prediction request for {len(request.requests)} vessels")
    
    start_time = time.time()
    results = []
    
    for i, pred_request in enumerate(request.requests):
        try:
            # Prepare sequence
            sequence = prepare_sequence(pred_request.observations[-config.sequence_length:])
            
            # Make prediction
            result = predict_trajectory(sequence, pred_request.prediction_horizon)
            
            results.append(PredictionResponse(
                predictions=result['predictions'],
                confidence=result['confidence'],
                processing_time=result['processing_time'],
                model_version="1.0.0"
            ))
            
        except Exception as e:
            logger.error(f"Batch prediction failed for request {i}: {e}")
            # Continue with other predictions
    
    total_time = time.time() - start_time
    
    return BatchPredictionResponse(
        results=results,
        total_processing_time=total_time
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

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Server startup
def start_server():
    """Start the production server"""
    uvicorn.run(
        "src.api.fastapi_server:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        timeout_keep_alive=config.api_timeout,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    start_server()
