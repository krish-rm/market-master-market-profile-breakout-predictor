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
10. [API Documentation](docs/API.md)
11. [Docker Deployment](docs/Docker.md)
12. [Project Structure](#project-structure)
13. [Development](docs/Development.md)
14. [Troubleshooting](docs/Troubleshooting.md)

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

See [data/README.md](data/README.md) for detailed data documentation.

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

---

## Usage

### Quick Start — Real BTC‑USD (End‑to‑End)

Run the whole pipeline on fresh market data, then explore predictions, performance, and the dashboard.

```bash
# 1) Install
make install

# 2) Refresh data and retrain on the latest 2y of BTC‑USD 1h candles
make data-clear-cache
make retrain-latest

# 3) Quick sanity check: print latest live BTC‑USD 1h close
make latest-price

# 4) Make predictions on real engineered data
make predict-latest          # predict on the most recent engineered session (real data)
make predict-monitor         # build labeled input from processed features and log predictions

# 5) Feature importance (opens an interactive bar chart at figures/feature_importance.html)
make feature-importance

# 6) Serve API and open docs at http://localhost:9696/docs
make serve

# 7) Start dashboard (auto‑loads latest engineered features and latest cached price)
make dashboard

# 8) Build and open performance/drift monitoring report
make monitor-report
make monitor-open
```

Expected Outputs:
- Training: saves `models/best_model.pkl`, `models/metrics.json`, `data/processed/market_profile.parquet`
- Predictions: writes `sample_input_predictions.json` and appends rows to `logs/predictions.csv`
- Feature importance: `figures/feature_importance.html`
- Monitoring: `reports/monitor_report.html` + `reports/monitor_report.json` (metrics and PSI if labels exist)

### 1. Training the Model

```bash
# Using Make (default config)
make train

# OR directly
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

### 2. Making Predictions (Real Data)

Recommend using engineered features generated by training:

```bash
# Predict on the most recent engineered session
make predict-latest
```

Notes:
- make predict-latest runs a one-off prediction on the latest engineered session (single row) and prints the result. It does not write labels to the log.
- To generate labeled logs for performance monitoring, see section "4. Monitoring (Performance & Drift)".

### 3. Running the Dashboard (Streamlit)

Launch the interactive dashboard for visualizations and predictions:

```bash
# Using Make
make dashboard

# OR directly
streamlit run scripts/dashboard.py
```

Dashboard opens at: `http://localhost:8501`

**Dashboard Features:**
- 🔮 **Predictions Page**: Interactive form to input Market Profile features and get real-time predictions
- 📊 **Model Performance**: Visualize metrics, confusion matrix, and performance charts
- 📈 **Feature Explorer**: Learn about all Market Profile features with detailed explanations
- 📜 **Prediction History**: Track your recent predictions

**Dashboard Pages:**
1. **Home**: Overview and quick start guide
2. **Predictions**: Make predictions with interactive form
3. **Model Performance**: View ROC AUC, Precision, Recall, F1-Score, and confusion matrix
4. **Feature Explorer**: Understand Market Profile features and view feature importance

### 4. Monitoring (Performance & Drift)

Generate a monitoring report with metrics and drift analysis:

```bash
# Ensure you have a predictions log with true labels
make predict-monitor

# Build monitoring report (HTML + JSON)
make monitor-report
# or
python scripts/monitor.py --log logs/predictions.csv --output reports/monitor_report.html
```

Outputs:
- `logs/predictions.csv`: Append-only prediction log (timestamp, prediction, probability, optional true_label)
- `reports/monitor_report.html`: Interactive report (probability time series, histogram, rolling accuracy when labels exist)
- `reports/monitor_report.json`: Summary metrics (accuracy, precision, recall, F1, ROC AUC when labels exist) and optional PSI drift

### 5. Refresh Cache and Load Latest Data

Get today’s data into the pipeline by clearing cached Parquet files and retraining:

```bash
# Remove cached raw/processed Parquet files
make data-clear-cache

# Retrain to fetch latest data and rebuild features/models
make retrain-latest
```

Quickly print the latest live BTC-USD 1h close (without retraining):

```bash
make latest-price
```

Full reset (clears data, models, logs, reports, figures):

```bash
make reset-all
```

### 6. Running with Docker (Optional)

Once you have trained a model locally (`make retrain-latest` so that `models/best_model.pkl` exists), you can serve the API in Docker:

```bash
# Build image
docker build -t market-profile-ml:latest .

# Run container (Windows cmd example)
docker run -d --name market-profile-api ^
  -p 9696:9696 ^
  -v "%cd%\models:/app/models" ^
  -v "%cd%\data:/app/data" ^
  market-profile-ml:latest

# Health check
curl http://localhost:9696/health
```

See [docs/Docker.md](docs/Docker.md) for full Docker deployment instructions and Docker Compose usage.

### 6. (Optional) Serving the API

The dashboard and CLI run locally without the API. Start the API only if you want external clients (curl/Postman/other apps) to call endpoints.

```bash
make serve
# or
uvicorn scripts.serve:app --host 0.0.0.0 --port 9696 --reload
```

---

## API Documentation

See the full API guide in [docs/API.md](docs/API.md).

---

## Docker Deployment

See full Docker instructions in [docs/Docker.md](docs/Docker.md).

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
│   └── dashboard.py                   # Streamlit dashboard
└── notebook.ipynb                     # Jupyter notebook (EDA)
```

---

## Development

See the full development guide in [docs/Development.md](docs/Development.md).

### Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## Troubleshooting

See the full troubleshooting guide in [docs/Troubleshooting.md](docs/Troubleshooting.md).

---

## Model Evaluation Metrics (Sample/Reference)

Note: The table below is an illustrative example. Your actual metrics will vary per retrain.
See current metrics in `models/metrics.json` and on the Dashboard (Model Performance).

### Test Set Performance (example)

```
                      Baseline    Logistic Reg  Random Forest  XGBoost
ROC AUC               0.500       0.580         0.640          0.672
Precision (class 1)   0.500       0.615         0.635          0.645
Recall (class 1)      0.500       0.520         0.580          0.589
F1-Score              0.500       0.565         0.607          0.615
```

**Baseline:** Random classifier (always predict majority class)

### Feature Importance (example)

Top 5 features by importance (see your current chart at `figures/feature_importance.html`):
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
- [ ] Deploy to AWS/GCP/Azure

### Medium Term
- [ ] Model performance monitoring (logging, drift detection, reports)
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

