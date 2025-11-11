"""
Prediction script for Market Profile model.

Usage:
    python scripts/predict.py --input data.json --output predictions.json
    python scripts/predict.py --input data.csv
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PredictionService:
    """Load model and make predictions."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize prediction service."""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        # Load model
        model_path = self.model_dir / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler if exists
        scaler_path = self.model_dir / "preprocessor.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load feature names
        features_path = self.model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Feature names loaded: {self.feature_names}")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on input features.
        
        Args:
            features: DataFrame with required features
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training
            features = features[self.feature_names]
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(features)
        else:
            X = features.values
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'num_samples': len(features)
        }
    
    def predict_single(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single sample.
        
        Args:
            features_dict: Dictionary with feature values
        
        Returns:
            Dictionary with prediction and probability
        """
        df = pd.DataFrame([features_dict])
        result = self.predict(df)
        
        return {
            'prediction': result['predictions'][0],
            'probability': result['probabilities'][0]
        }


def load_input(input_path: str) -> pd.DataFrame:
    """Load input data from CSV or JSON."""
    path = Path(input_path)
    
    if path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return df


def save_predictions(
    results: Dict[str, Any],
    output_path: str = None,
    input_path: str = None
) -> None:
    """Save predictions to file."""
    if output_path is None:
        if input_path:
            output_path = Path(input_path).stem + "_predictions.json"
        else:
            output_path = "predictions.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Predictions saved to {output_path}")


def main(
    input_path: str,
    output_path: str = None,
    model_dir: str = "models"
):
    """Main prediction pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Loading prediction service...")
    service = PredictionService(model_dir=model_dir)
    
    logger.info(f"Loading input data from {input_path}...")
    features = load_input(input_path)
    
    logger.info(f"Making predictions on {len(features)} samples...")
    results = service.predict(features)
    
    logger.info(f"Predictions completed")
    logger.info(f"Results:\n{json.dumps(results, indent=2)}")
    
    # Save predictions
    save_predictions(results, output_path, input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data file (CSV or JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output predictions file"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory with model artifacts"
    )
    args = parser.parse_args()
    
    main(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir
    )

