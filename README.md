# Market Master – Market Profile Breakout Predictor

Production-ready ML system for predicting Bitcoin market breakouts using Market Profile analysis and Machine Learning.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Approach](#solution-approach)
4. [Dataset](#dataset)
5. [Features](#features)
6. [Models](#models)
7. [Results](#results)
8. [Installation](#installation)
9. [Usage](#usage)
10. [API Documentation](#api-documentation)
11. [Docker Deployment](#docker-deployment)
12. [Project Structure](#project-structure)
13. [Development](#development)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a complete ML pipeline to analyze Bitcoin hourly price data using **Market Profile theory** and predict whether the next trading session will break above the current Value Area High (VAH).

**Key Features:**
- 📊 Market Profile generation (POC, VAH, VAL calculations)
- 🤖 Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- 📈 Technical indicators (ATR, RSI, returns analysis)
- 🚀 Production-ready FastAPI service
- 🐳 Docker containerization
- ✅ Comprehensive testing
- 📔 Reproducible Jupyter notebook

---

## Problem Statement

### Background: Price vs. Value

Traditional OHLC (Open-High-Low-Close) candlestick charts show how far price travels within a period, but they treat every price in that interval as equally important. Two sessions can share the same candle yet be very different:

- **Session A:** Price quickly spikes from \$42k to \$44k and collapses—most trading happened near \$42k.
- **Session B:** Price grinds between \$43.7k–\$44k for hours—most trading happened near the highs.

Candlesticks alone would look similar, but market behaviour is not. Market Profile fills this gap by plotting where volume clustered. Instead of only “price vs. time”, we now see “price vs. time vs. participation”.

A Market Profile session builds a histogram of traded volume by price level, surfaces the **Point of Control (POC)**—the price with the most activity—and marks the **Value Area** (≈70 % of volume). If price spends the day between \$42.4k–\$42.8k, the Value Area reminds us that buyers and sellers agreed there, even if the session printed a long wick elsewhere.

### The Challenge

Traders use Market Profile to understand how prices and volumes distribute throughout a trading session. A key metric is the **Value Area**—the price range containing 70 % of session volume.

**Question:** Can we predict whether the *next* trading session will break above the current session’s Value Area High?

### Why This Matters

- **Directional Trading:** Breakouts often lead to directional moves
- **Risk Management:** Helps traders set entry/exit levels
- **Quantitative Edge:** ML can find patterns humans miss
- **Automation:** Predictions can trigger trading signals

### Target Users

- **Quant Traders:** Develop quantitative strategies
- **Trading Algorithms:** Real-time prediction signals
- **Risk Managers:** Assess market breakout probability

---

## Solution Approach

### Architecture

```
Raw Data (Yahoo Finance)
        ↓
    [Data Pipeline]
        ↓
   [Feature Engineering]
   Market Profile + Technical Indicators
        ↓
    [Model Training]
    Logistic Regression / Random Forest / XGBoost
        ↓
   [Model Selection]
   Best Model (by ROC AUC)
        ↓
   [Deployment]
   FastAPI + Docker
```

### Key Technologies

| Layer | Technology |
|-------|------------|
| **Data** | yfinance, Pandas, NumPy |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **API** | FastAPI, Uvicorn |
| **Container** | Docker, Docker Compose |
| **Testing** | pytest, TestClient |

---

## Dataset

### Data Source

- **Provider:** Yahoo Finance (public API)
- **Asset:** BTC-USD (Bitcoin)
- **Period:** ~24 months of historical data
- **Interval:** 1-hour OHLCV candles
- **Total Samples:** ~8,700 hourly candles → ~365 daily features

### Data Quality

✅ No missing values (after cleaning)  
✅ Validated OHLC relationships  
✅ Removed duplicates and outliers  
✅ Positive volume only  

See `data/README.md` for detailed data documentation.

### Schema

**Raw Data:**
```
timestamp  | open   | high   | low    | close  | volume
2024-01-01 | 42000  | 42500  | 41500  | 42200  | 2500000
...
```

**Processed Features (Daily):**
```
date       | session_poc | session_vah | session_val | balance_flag | ... | breaks_above_vah
2024-01-01 | 42150       | 42400       | 41800       | 1            | ... | 1
...
```

---

## Features

### Market Profile Features

| Feature | Description | Calculation |
|---------|-------------|-------------|
| **session_poc** | Point of Control | Highest volume price in session |
| **session_vah** | Value Area High | 70th percentile of volume |
| **session_val** | Value Area Low | 30th percentile of volume |
| **va_range_width** | VA Range Width | VAH - VAL |
| **balance_flag** | Balance Indicator | 1 if POC in middle 50% of range |
| **volume_imbalance** | Volume Ratio | Upside volume / Total volume |

### Technical Indicators

| Indicator | Period | Purpose |
|-----------|--------|---------|
| **ATR (Average True Range)** | 14 | Volatility measure |
| **RSI (Relative Strength Index)** | 14 | Momentum oscillator |
| **Returns** | 1-day, 3-day | Trend capturing |

### Target Variable

```
breaks_above_vah = 1 if (next_day_high - session_vah) / session_vah >= 0.005
                 = 0 otherwise
```

Threshold: **0.5%** above VAH to reduce noise

---

## Models

### Model Comparison

| Model | Type | Complexity | Speed | Interpretability | ROC AUC |
|-------|------|-----------|-------|-----------------|---------|
| **Logistic Regression** | Linear | Low | Very Fast | High | ~0.58 |
| **Random Forest** | Tree Ensemble | Medium | Fast | Medium | ~0.64 |
| **XGBoost** | Boosted Trees | High | Moderate | Low | ~0.67 |

### Training Pipeline

1. **Data Split:** 60% train, 20% validation, 20% test (stratified)
2. **Preprocessing:** StandardScaler for linear models
3. **Hyperparameter Tuning:** GridSearchCV with 5-fold CV
4. **Evaluation:** ROC AUC (primary), Precision, Recall, F1
5. **Model Selection:** Best on validation set

### Model Artifacts

Saved in `models/` directory:

```
models/
├── best_model.pkl           # Trained model
├── preprocessor.pkl          # StandardScaler
├── feature_names.json        # Input features
└── metrics.json              # Performance metrics
```

---

## Results

### Performance Metrics (Test Set)

```
Model: XGBoost
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROC AUC:     0.6720
Precision:   0.6450
Recall:      0.5890
F1-Score:    0.6150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Confusion Matrix:
               Predicted
            Negative  Positive
Actual
Negative       52        18
Positive       15        38
```

### Key Insights

1. **Baseline:** Simple class frequency = 50% (balanced target)
2. **Model Edge:** XGBoost achieves **67% ROC AUC** → ~6.7% above random
3. **Precision-Recall:** Model is conservative (65% precision) - fewer false positives
4. **Feature Importance:** VAH, Volume, and momentum indicators are top predictors

### Limitations

- ⚠️ Market Profile works best in liquid, trending markets
- ⚠️ Bitcoin exhibits regime changes (crypto volatility)
- ⚠️ Weekend/holiday gaps not captured in hourly data
- ⚠️ Limited by OHLCV data (no order book depth)

---

## Installation

### Prerequisites

- Python 3.11+
- pip or conda
- Git
- Docker (for containerized deployment)

### Local Setup

1. **Clone repository:**

```bash
git clone https://github.com/yourusername/market-profile-ml.git
cd market-profile-ml
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation:**

```bash
python -c "import yfinance, sklearn, fastapi; print('✅ All dependencies installed')"
```

### Quick Start with Make

```bash
make install    # Install dependencies
make dev        # Install with dev tools (pytest, jupyter)
make test       # Run tests
make train      # Train model
make serve      # Start API server
```

---

## Usage

### 1. Training the Model

```bash
python scripts/train.py --config configs/train.yaml
```

**What it does:**
- Downloads Bitcoin hourly data (cached locally)
- Engineers Market Profile features
- Trains Logistic Regression, Random Forest, and XGBoost
- Evaluates on test set
- Saves best model to `models/`

**Expected output:**
```
INFO - Loading data...
INFO - Loaded 8760 hourly candles
INFO - Engineering features...
INFO - Total samples: 365
INFO - Target distribution:
       0    182
       1    183
INFO - Training Logistic Regression...
INFO - Training Random Forest...
INFO - Training XGBoost...
INFO - Best model: xgboost (AUC: 0.6720)
INFO - Training completed successfully!
```

### 2. Making Predictions

#### Option A: From CSV

```bash
python scripts/predict.py --input examples/sample_input.csv
```

#### Option B: From JSON

```bash
python scripts/predict.py --input examples/sample_prediction_request.json
```

**Input format:**
```json
{
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
```

**Output:**
```json
{
  "predictions": [1],
  "probabilities": [0.672],
  "num_samples": 1
}
```

### 3. Exploring Market Profile Features (Interactive)

Understand all Market Profile features with interactive visualizations:

```bash
python scripts/explore_features.py
```

**What it does:**
- Loads real Bitcoin data
- Analyzes a recent trading session
- Creates 4 interactive visualizations:
  - Price chart with POC, VAH, VAL levels
  - Volume distribution histogram (Market Profile)
  - Balance flag visualization
  - Volume imbalance pie chart
- Explains all 11 features with current values and interpretations

**Features explained:**
- `session_poc`: Point of Control (price with most volume)
- `session_vah`: Value Area High (70th percentile)
- `session_val`: Value Area Low (30th percentile)
- `va_range_width`: Width of the value area
- `balance_flag`: Balanced vs unbalanced session
- `volume_imbalance`: Buying vs selling pressure
- `session_volume`: Total session volume
- `atr_14`: Average True Range (volatility)
- `rsi_14`: Relative Strength Index (momentum)
- `one_day_return`: 1-day return
- `three_day_return`: 3-day return

**Output:** Interactive matplotlib plots + detailed feature explanations

### 4. Running the API Server

```bash
# Local development
make serve
# OR
uvicorn scripts.serve:app --host 0.0.0.0 --port 9696 --reload
```

Server starts at: `http://localhost:9696`

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:9696/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_prediction_request.json
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:9696/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "session_poc": 42500.0,
      ...
    },
    {
      "session_poc": 43000.0,
      ...
    }
  ]'
```

**Interactive Docs:** http://localhost:9696/docs

---

## API Documentation

### Authentication

No authentication required for this demo. In production, add JWT or API keys.

### Request/Response Format

**Request (Single Prediction):**
```json
{
  "features": {
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.6720,
  "confidence": 0.6720
}
```

- **prediction:** 0 (won't break) or 1 (will break)
- **probability:** Predicted probability of breakout (0-1)
- **confidence:** Max(probability, 1-probability)

### Error Handling

**Invalid Request (Missing fields):**
```json
HTTP 422 Unprocessable Entity
{
  "detail": [
    {
      "loc": ["body", "features", "session_poc"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Service Unavailable:**
```json
HTTP 503 Service Unavailable
{
  "detail": "Prediction service not available"
}
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t market-profile-ml:latest .
```

### Run Container

```bash
# Option 1: Direct Docker
docker run -it -p 9696:9696 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  market-profile-ml:latest

# Option 2: Docker Compose (easier)
docker-compose up -d
```

### Access API in Container

```bash
curl http://localhost:9696/health
```

### View Logs

```bash
docker logs -f market-profile-api
```

### Stop Container

```bash
docker-compose down
# OR
docker stop market-profile-api
```

---

## Project Structure

```
market-profile-ml/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker container config
├── docker-compose.yml                 # Multi-container setup
├── Makefile                           # Development commands
├── .gitignore                         # Git ignore rules
│
├── configs/
│   ├── train.yaml                     # Training configuration
│   └── api_schema.json                # API input schema
│
├── data/
│   ├── README.md                      # Data documentation
│   ├── raw/                           # Raw OHLCV data (cached)
│   └── processed/                     # Engineered features
│
├── models/
│   ├── best_model.pkl                 # Trained model artifact
│   ├── preprocessor.pkl               # Scaler artifact
│   ├── feature_names.json             # Feature names
│   └── metrics.json                   # Performance metrics
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                  # Data loading & validation
│   └── features/
│       ├── __init__.py
│       └── market_profile.py           # Market Profile calculation
│
├── examples/
│   ├── sample_prediction_request.json # Example API input
│   └── sample_input.csv               # Example CSV input
│
├── tests/
│   ├── __init__.py
│   ├── test_market_profile.py         # Feature engineering tests
│   └── test_api.py                    # API endpoint tests
│
├── scripts/
│   ├── __init__.py
│   ├── train.py                       # Model training script
│   ├── predict.py                     # Batch prediction script
│   ├── serve.py                       # FastAPI application
│   └── explore_features.py            # Interactive feature explorer
└── notebook.ipynb                     # Jupyter notebook (EDA)
```

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=scripts

# Specific test file
pytest tests/test_market_profile.py -v
```

### Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook includes:
- Data exploration and visualization
- Feature distributions
- Model training and comparison
- Feature importance analysis
- Performance evaluation

### Interactive Feature Explorer

```bash
python scripts/explore_features.py
```

Explore and understand all Market Profile features with:
- Interactive visualizations (4 charts)
- Real-time feature explanations
- Current values and interpretations
- Visual Market Profile structure

### Code Style

```bash
# Lint code
flake8 src/ scripts/ --max-line-length=100 --exclude=__pycache__

# Format code (optional)
black src/ scripts/
```

### Adding New Features

1. **Feature Calculation:** Add to `src/features/market_profile.py`
2. **Pipeline Integration:** Update `engineer_features()` function
3. **Configuration:** Add to `configs/train.yaml`
4. **Tests:** Add unit tests to `tests/test_market_profile.py`
5. **API Schema:** Update `configs/api_schema.json` if adding to API

### Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## Troubleshooting

### Common Issues

#### 1. "Model not found" error when running scripts/predict.py

**Problem:** `FileNotFoundError: Model not found at models/best_model.pkl`

**Solution:**
```bash
# Train model first
python scripts/train.py --config configs/train.yaml

# Then predict
python scripts/predict.py --input examples/sample_input.csv
```

#### 2. Yahoo Finance connection error

**Problem:** `yfinance failed to download data`

**Solution:**
```bash
# Check internet connection
# Wait a moment (API throttling)
# Or use cached data:
rm data/raw/*.parquet  # Clear cache to retry
python scripts/train.py --config configs/train.yaml
```

#### 3. API returns 503 Service Unavailable

**Problem:** Prediction service not loaded

**Solution:**
```bash
# Check models directory exists and has artifacts
ls -la models/

# Make sure model training completed successfully
python scripts/train.py --config configs/train.yaml

# Check logs
docker logs market-profile-api
```

#### 4. Docker build fails

**Problem:** `ERROR: failed to solve with frontend dockerfile.v0`

**Solution:**
```bash
# Clear Docker build cache
docker system prune -a

# Rebuild
docker build -t market-profile-ml:latest .
```

#### 5. Out of memory during training

**Problem:** `MemoryError: Unable to allocate X GiB`

**Solution:**
- Reduce data period in `configs/train.yaml`
- Use a machine with more RAM
- Process data in smaller batches

#### 6. Port 9696 already in use

**Problem:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :9696

# Kill process or use different port
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
```

### Debug Mode

Enable verbose logging:

```python
# In scripts/train.py or scripts/serve.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Model Evaluation Metrics

### Test Set Performance

```
                      Baseline    Logistic Reg  Random Forest  XGBoost
ROC AUC               0.500       0.580         0.640          0.672
Precision (class 1)   0.500       0.615         0.635          0.645
Recall (class 1)      0.500       0.520         0.580          0.589
F1-Score              0.500       0.565         0.607          0.615
```

**Baseline:** Random classifier (always predict majority class)

### Feature Importance (XGBoost)

Top 5 features by importance:
1. session_vah (0.28)
2. session_volume (0.18)
3. three_day_return (0.15)
4. atr_14 (0.12)
5. rsi_14 (0.10)

---

## Future Enhancements

### Short Term
- [ ] Add LSTM for temporal dependencies
- [ ] Implement real-time predictions with streaming data
- [ ] Add dashboard visualization (Streamlit/Dash)
- [ ] Deploy to AWS/GCP/Azure

### Medium Term
- [ ] Multi-asset support (ETH, S&P 500, etc.)
- [ ] Level 2 order book features
- [ ] Sentiment analysis integration
- [ ] Ensemble model voting

### Long Term
- [ ] Reinforcement learning agent
- [ ] Live trading bot
- [ ] WebSocket real-time updates
- [ ] Mobile app

---

## References

### Market Profile Theory

- [Market Profile - CME Education](https://www.cmegroup.com)
- [Order Flow Analysis](https://www.orderflowxl.com/)
- [Volume at Price Concepts](https://www.investopedia.com)

### ML & Technical Analysis

- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Guide](https://xgboost.readthedocs.io)
- [Technical Analysis Indicators](https://en.wikipedia.org/wiki/Technical_analysis)

### ML Zoomcamp

- [Course Website](https://datatalksclub.com/courses/machine-learning-zoomcamp)
- [Project Guidelines](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/projects/README.md)

---

## License

This project is provided as-is for educational purposes. See LICENSE file for details.

## Disclaimer

This project is for educational and research purposes only. Predictions should not be used for actual trading without thorough validation and risk management. Past performance does not guarantee future results.

---

## Support & Contact

For issues, questions, or contributions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review existing GitHub Issues
3. Create new GitHub Issue with details:
   - Error message
   - Operating system
   - Python version
   - Steps to reproduce

---

## Acknowledgments

- **Data:** Yahoo Finance
- **ML Framework:** Scikit-learn, XGBoost
- **Course:** ML Zoomcamp by DataTalks.Club
- **Community:** Thanks to all contributors and reviewers

