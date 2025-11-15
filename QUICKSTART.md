# Quick Start Guide

## 📦 Installation

### Option 1: Local Development

```bash
# Clone repository
git clone <repo-url>
cd market-profile-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import yfinance, sklearn, fastapi; print('✅ Ready to go!')"
```

### Option 2: Using Makefile

```bash
make install       # Install dependencies
make dev           # Install with dev tools (Jupyter, pytest)
```

---

## 🚀 Running the Complete Pipeline

### 1. Train the Model

```bash
python scripts/train.py --config configs/train.yaml
```

**What happens:**
- Downloads Bitcoin hourly data from Yahoo Finance (cached locally)
- Engineers Market Profile features (POC, VAH, VAL, etc.)
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Evaluates models and saves best one to `models/`
- Outputs metrics to `models/metrics.json`

**Time:** ~5-10 minutes first run, ~2 minutes with cache

**Output:**
```
INFO - Loading data...
INFO - Loaded 8760 hourly candles
INFO - Engineering features...
INFO - Total samples: 365
INFO - Training XGBoost...
INFO - Best model: xgboost (AUC: 0.6720)
INFO - Training completed successfully!
```

---

## 🤖 Making Predictions

### Option A: From CSV File

```bash
python scripts/predict.py --input examples/sample_input.csv
```

### Option B: From JSON File

```bash
python scripts/predict.py --input examples/sample_prediction_request.json
```

### Output Example

```
{
  "predictions": [1, 0, 1],
  "probabilities": [0.672, 0.421, 0.598],
  "num_samples": 3
}
```

---

## 📊 Exploring Market Profile Features (Interactive)

Understand all Market Profile features with interactive visualizations and explanations:

```bash
python scripts/explore_features.py
```

**What you'll see:**
- 📈 **4 Interactive Visualizations:**
  - Price chart with POC, VAH, VAL levels marked
  - Volume distribution histogram (Market Profile structure)
  - Balance flag visualization (balanced vs unbalanced sessions)
  - Volume imbalance pie chart (buying vs selling pressure)

- 📝 **Detailed Feature Explanations:**
  - `session_poc`: Point of Control (price with most volume)
  - `session_vah`: Value Area High (70th percentile)
  - `session_val`: Value Area Low (30th percentile)
  - `va_range_width`: Width of the value area
  - `balance_flag`: Balanced vs unbalanced session indicator
  - `volume_imbalance`: Buying vs selling pressure ratio
  - `session_volume`: Total session volume
  - `atr_14`: Average True Range (volatility measure)
  - `rsi_14`: Relative Strength Index (momentum oscillator)
  - `one_day_return`: 1-day return (short-term trend)
  - `three_day_return`: 3-day return (medium-term trend)

**Output:** Interactive matplotlib plots + detailed feature explanations with real data values

**Example Output:**
```
📊 SESSION_POC (POINT OF CONTROL)
Description: The price level where the most trading volume occurred...
Current Value: 101,563.44
💡 Interpretation: Most trading happened at $101,563.44...
```

---

## 🌐 Running the API

### Local Development

```bash
# Option 1: Direct
python scripts/serve.py

# Option 2: Using uvicorn
uvicorn scripts.serve:app --host 0.0.0.0 --port 9696 --reload

# Option 3: Using Make
make serve
```

API runs at: `http://localhost:9696`

### API Endpoints

**Health Check:**
```bash
curl http://localhost:9696/health
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_prediction_request.json
```

**Interactive Docs:**
```
http://localhost:9696/docs
```

---

## 🐳 Docker Deployment

### Option 1: Build and Run

```bash
# Build image
docker build -t market-profile-ml:latest .

# Run container
docker run -it -p 9696:9696 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  market-profile-ml:latest
```

### Option 2: Docker Compose (Recommended)

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

API available at: `http://localhost:9696`

---

## 📔 Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook includes:
- Data exploration and visualization
- Feature engineering walkthrough
- Model training and comparison
- Performance evaluation
- Feature importance analysis

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=scripts

# Specific test
pytest tests/test_market_profile.py -v
```

---

## 📊 Project Structure

```
market-profile-ml/
├── scripts/
│   ├── __init__.py
│   ├── train.py          # Model training script
│   ├── predict.py        # Batch prediction script
│   └── serve.py          # FastAPI service
├── notebook.ipynb        # Jupyter notebook
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
├── requirements.txt      # Python dependencies
│
├── src/
│   ├── data/
│   │   └── loader.py     # Data loading utilities
│   └── features/
│       └── market_profile.py  # Feature engineering
│
├── configs/
│   ├── train.yaml        # Training configuration
│   └── api_schema.json   # API input schema
│
├── data/
│   ├── raw/              # Raw OHLCV data (auto-cached)
│   └── processed/        # Engineered features
│
├── models/
│   ├── best_model.pkl    # Trained model artifact
│   ├── preprocessor.pkl  # Scaler artifact
│   ├── feature_names.json
│   └── metrics.json
│
├── examples/
│   ├── sample_input.csv
│   └── sample_prediction_request.json
│
└── tests/
    ├── test_market_profile.py
    └── test_api.py
```

---

## 🔧 Configuration

Edit `configs/train.yaml` to customize:

```yaml
data:
  ticker: "BTC-USD"        # Change cryptocurrency/stock
  period: "2y"             # Data period (1y, 2y, 5y, etc)
  interval: "1h"           # Candle interval (1h, 4h, 1d, etc)

model:
  test_size: 0.2           # Test set proportion
  random_state: 42         # Reproducibility seed
```

---

## 📈 Performance Metrics

After training, check `models/metrics.json`:

```json
{
  "model_name": "xgboost",
  "auc": 0.6720,
  "precision": 0.6450,
  "recall": 0.5890,
  "f1": 0.6150,
  "confusion_matrix": [[52, 18], [15, 38]]
}
```

---

## 🐛 Troubleshooting

### "Model not found" error
```bash
# Train model first
python scripts/train.py --config configs/train.yaml
```

### Port 9696 already in use
```bash
# Use different port
uvicorn scripts.serve:app --port 8000
```

### Yahoo Finance download fails
```bash
# Check connection or wait (rate limiting)
# Data is cached, so retry later
```

### Out of memory
```bash
# Reduce data period in configs/train.yaml
# Or run on machine with more RAM
```

---

## 📚 Documentation

- **Full README:** `README.md` - Complete project documentation
- **Data Documentation:** `data/README.md` - Schema and preprocessing
- **API Schema:** `configs/api_schema.json` - Input specifications

---

## 🎯 Common Workflows

### Development Cycle

```bash
make install              # First time setup
python scripts/train.py   # Train model
make test                 # Run tests
make serve                # Start API
# Make changes...
docker build -t market-profile-ml .  # Build image
docker run -p 9696:9696 market-profile-ml  # Test container
```

### Production Deployment

```bash
# Prepare
make install
python scripts/train.py --config configs/train.yaml

# Build image
docker build -t market-profile-ml:prod .

# Deploy
docker run -d -p 9696:9696 \
  -v /path/to/models:/app/models \
  market-profile-ml:prod

# Monitor
curl http://localhost:9696/health
```

### Model Retraining

```bash
# Retrain with new data
rm data/raw/BTC-USD_2y_1h.parquet  # Clear cache
python scripts/train.py --config configs/train.yaml

# Update API (restart if using Docker)
docker-compose restart api
```

---

## 📞 Support

1. Check README.md for detailed documentation
2. Review test files for usage examples
3. Check logs for error details
4. Open GitHub issue with error message

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
✓ python -c "import yfinance, sklearn, fastapi"
✓ python scripts/train.py --config configs/train.yaml  
✓ python scripts/predict.py --input examples/sample_input.csv
✓ python scripts/serve.py  # Should start without errors
✓ curl http://localhost:9696/health
✓ pytest tests/ -v  # All tests pass
✓ docker-compose up -d  # Containers start
```

---

## 🎓 Learning Resources

- **Market Profile Theory:** See README.md → References
- **ML Zoomcamp:** Course materials at datatalksclub.com
- **Technical Analysis:** investopedia.com and tradingview.com



