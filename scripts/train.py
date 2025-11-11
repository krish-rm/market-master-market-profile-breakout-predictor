"""
Model training script for Market Profile prediction.

Usage:
    python scripts/train.py --config configs/train.yaml
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import xgboost as xgb

from src.data.loader import DataLoader
from src.features.market_profile import engineer_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(config: Dict[str, Any]) -> tuple:
    """Load and prepare data for training."""
    logger.info("Loading data...")
    
    # Load raw data
    loader = DataLoader(
        ticker=config['data']['ticker'],
        period=config['data']['period'],
        interval=config['data']['interval']
    )
    raw_data = loader.fetch_data(
        use_cache=True,
        cache_dir=config['data']['raw_dir']
    )
    
    # Clean data
    raw_data = loader.clean_data(raw_data)
    logger.info(f"Loaded {len(raw_data)} hourly candles")
    
    # Engineer features
    logger.info("Engineering features...")
    features = engineer_features(
        raw_data,
        target_threshold=config['label']['breakout_threshold']
    )
    
    # Save processed features
    output_path = Path(config['data']['processed_dir']) / config['data']['output_file']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    logger.info(f"Features saved to {output_path}")
    logger.info(f"Total samples: {len(features)}")
    
    # Print target distribution
    target_counts = features['breaks_above_vah'].value_counts()
    logger.info(f"Target distribution:\n{target_counts}")
    logger.info(f"Positive class ratio: {target_counts.get(1, 0) / len(features):.2%}")
    
    return features


def prepare_train_test_split(
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> tuple:
    """Prepare stratified train/validation/test split."""
    
    X = features.drop(['breaks_above_vah', 'date', 'next_day_high', 'profile_type', 
                       'poc_volume', 'day_high', 'day_low'], axis=1, errors='ignore')
    y = features['breaks_above_vah']
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"Feature columns: {list(X.columns)}")
    logger.info(f"Total features: {X.shape[1]}")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state'],
        stratify=y
    )
    
    # Second split: train vs val from temp
    val_ratio = config['model']['validation_size'] / (1 - config['model']['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=config['model']['random_state'],
        stratify=y_temp
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train multiple models and return best one."""
    
    # Initialize scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {
        'C': config['model']['logistic_regression']['C'],
        'max_iter': [config['model']['logistic_regression']['max_iter']]
    }
    
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=config['model']['random_state']),
        {'C': lr_params['C']},
        cv=config['model']['cv_folds'],
        scoring='roc_auc',
        n_jobs=-1
    )
    lr_grid.fit(X_train_scaled, y_train)
    
    lr_best = lr_grid.best_estimator_
    lr_pred = lr_best.predict_proba(X_val_scaled)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_pred)
    
    results['logistic_regression'] = {
        'model': lr_best,
        'scaler': scaler,
        'auc': lr_auc,
        'params': lr_grid.best_params_
    }
    logger.info(f"Logistic Regression - AUC: {lr_auc:.4f}, Best params: {lr_grid.best_params_}")
    
    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=config['model']['random_state'], n_jobs=-1),
        {
            'n_estimators': config['model']['random_forest']['n_estimators'],
            'max_depth': config['model']['random_forest']['max_depth'],
            'min_samples_split': config['model']['random_forest']['min_samples_split'],
            'min_samples_leaf': config['model']['random_forest']['min_samples_leaf']
        },
        cv=config['model']['cv_folds'],
        scoring='roc_auc',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    
    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)
    
    results['random_forest'] = {
        'model': rf_best,
        'scaler': None,
        'auc': rf_auc,
        'params': rf_grid.best_params_
    }
    logger.info(f"Random Forest - AUC: {rf_auc:.4f}, Best params: {rf_grid.best_params_}")
    
    # 3. XGBoost
    logger.info("Training XGBoost...")
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=config['model']['random_state'], use_label_encoder=False),
        {
            'n_estimators': config['model']['xgboost']['n_estimators'],
            'max_depth': config['model']['xgboost']['max_depth'],
            'learning_rate': config['model']['xgboost']['learning_rate'],
            'subsample': config['model']['xgboost']['subsample']
        },
        cv=config['model']['cv_folds'],
        scoring='roc_auc',
        n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    
    xgb_best = xgb_grid.best_estimator_
    xgb_pred = xgb_best.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    
    results['xgboost'] = {
        'model': xgb_best,
        'scaler': None,
        'auc': xgb_auc,
        'params': xgb_grid.best_params_
    }
    logger.info(f"XGBoost - AUC: {xgb_auc:.4f}, Best params: {xgb_grid.best_params_}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    logger.info(f"Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    
    return results, best_model_name


def evaluate_model(
    model: Any,
    scaler: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"\n{model_name} - Test Set Performance:")
    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        'model_name': model_name,
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }


def save_artifacts(
    model: Any,
    scaler: Any,
    X_train: pd.DataFrame,
    metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Save model artifacts."""
    
    model_dir = Path(config['artifacts']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / config['artifacts']['model']
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor/scaler if exists
    if scaler is not None:
        scaler_path = model_dir / config['artifacts']['preprocessor']
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    
    # Save metrics
    metrics_path = model_dir / config['artifacts']['metrics']
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature names
    features_path = model_dir / 'feature_names.json'
    with open(features_path, 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    logger.info(f"Feature names saved to {features_path}")


def main(config_path: str = "configs/train.yaml"):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("Market Master – Market Profile Breakout Predictor - Training Pipeline")
    logger.info("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Set random seed
    np.random.seed(config['model']['random_state'])
    
    # Load and prepare data
    features = load_and_prepare_data(config)
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        features, config
    )
    
    # Train models
    results, best_model_name = train_models(
        X_train, X_val, y_train, y_val, config
    )
    
    # Evaluate best model on test set
    best_result = results[best_model_name]
    metrics = evaluate_model(
        best_result['model'],
        best_result['scaler'],
        X_test,
        y_test,
        best_model_name
    )
    
    # Save artifacts
    save_artifacts(
        best_result['model'],
        best_result['scaler'],
        X_train,
        metrics,
        config
    )
    
    logger.info("="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Market Master Market Profile Breakout model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML"
    )
    args = parser.parse_args()
    
    main(config_path=args.config)

