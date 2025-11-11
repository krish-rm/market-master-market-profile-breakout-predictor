""" 
FastAPI service for Market Profile predictions. 
 
Run with: uvicorn scripts.serve:app --host 0.0.0.0 --port 9696 
""" 

import logging
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from scripts.predict import PredictionService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Market Master – Market Profile Breakout API",
    description="API for predicting market profile breakouts",
    version="1.0.0"
)

# Load prediction service globally
try:
    service = PredictionService(model_dir="models")
    logger.info("Prediction service loaded successfully")
except Exception as e:
    logger.error(f"Error loading prediction service: {e}")
    service = None


# Request/Response models
class SessionFeatures(BaseModel):
    """Market Profile session features for prediction."""
    
    session_poc: float = Field(..., description="Point of Control")
    session_vah: float = Field(..., description="Value Area High")
    session_val: float = Field(..., description="Value Area Low")
    va_range_width: float = Field(..., description="VA range width")
    balance_flag: int = Field(..., description="Balance flag (0 or 1)")
    volume_imbalance: float = Field(..., description="Volume imbalance ratio")
    one_day_return: float = Field(..., description="1-day return")
    three_day_return: float = Field(..., description="3-day return")
    atr_14: float = Field(..., description="Average True Range (14-period)")
    rsi_14: float = Field(..., description="RSI (14-period)")
    session_volume: float = Field(..., description="Session volume")
    
    class Config:
        schema_extra = {
            "example": {
                "session_poc": 42500.0,
                "session_vah": 42750.0,
                "session_val": 42250.0,
                "va_range_width": 500.0,
                "balance_flag": 1,
                "volume_imbalance": 0.52,
                "one_day_return": 0.01,
                "three_day_return": 0.025,
                "atr_14": 350.0,
                "rsi_14": 60.5,
                "session_volume": 1500000.0
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request."""
    features: SessionFeatures


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., description="Probability of class 1")
    confidence: float = Field(..., description="Confidence level")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    if service is None:
        return {
            "status": "unhealthy",
            "message": "Prediction service not loaded"
        }
    
    return {
        "status": "healthy",
        "message": "API is running and ready for predictions"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction for a single session.
    
    Args:
        request: SessionFeatures with market profile data
    
    Returns:
        Prediction and probability
    """
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    try:
        # Convert request to dictionary
        features_dict = request.features.dict()
        
        # Make prediction
        result = service.predict_single(features_dict)
        
        # Calculate confidence
        prob = result['probability']
        confidence = max(prob, 1 - prob)
        
        return {
            "prediction": result['prediction'],
            "probability": round(prob, 4),
            "confidence": round(confidence, 4)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-predict")
async def batch_predict(features_list: List[SessionFeatures]):
    """
    Make predictions for multiple sessions.
    
    Args:
        features_list: List of SessionFeatures
    
    Returns:
        List of predictions
    """
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    try:
        # Convert to DataFrame
        data = [f.dict() for f in features_list]
        df = pd.DataFrame(data)
        
        # Make predictions
        results = service.predict(df)
        
        # Format response
        predictions = []
        for i in range(len(df)):
            prob = results['probabilities'][i]
            predictions.append({
                "prediction": results['predictions'][i],
                "probability": round(prob, 4),
                "confidence": round(max(prob, 1 - prob), 4)
            })
        
        return {
            "num_samples": len(df),
            "predictions": predictions
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/info")
async def info():
    """
    Get API information and feature schema.
    
    Returns:
        API information and required fields
    """
    # Load schema from config
    schema_path = Path("configs/api_schema.json")
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    else:
        schema = {}
    
    return {
        "title": "Market Master – Market Profile Breakout API",
        "version": "1.0.0",
        "description": "Predict market profile breakouts using ML model",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Single prediction",
            "POST /batch-predict": "Batch predictions",
            "GET /info": "API information"
        },
        "feature_schema": schema
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Market Master – Market Profile Breakout API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9696,
        log_level="info"
    )

