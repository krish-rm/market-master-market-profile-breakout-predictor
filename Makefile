.PHONY: help install dev test train predict serve docker-build docker-run docker-stop clean clean-all lint

help:
	@echo "Market Master – Market Profile Breakout Predictor - Available Commands"
	@echo "========================================"
	@echo "make install       - Install dependencies"
	@echo "make dev           - Install dev dependencies"
	@echo "make test          - Run unit tests"
	@echo "make train         - Train model"
	@echo "make predict       - Make predictions (requires --input=FILE)"
	@echo "make serve         - Start FastAPI server locally"
	@echo "make dashboard     - Start Streamlit dashboard"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run Docker container"
	@echo "make docker-stop   - Stop Docker container"
	@echo "make lint          - Run linting"
	@echo "make clean         - Clean up cache files and build artifacts"
	@echo "make clean-all     - Clean everything including data and models"
	@echo "make feature-importance - Generate feature importance chart"
	@echo "make data-clear-cache  - Remove cached raw/processed data (parquet)"
	@echo "make retrain-latest    - Clear raw cache and retrain to load latest data"
	@echo "make latest-price      - Print latest BTC-USD 1h candle close via yfinance"
	@echo "make predict-sample    - Run predictions on examples/sample_input.csv"
	@echo "make build-monitor-input - Create labeled input from processed features"
	@echo "make predict-monitor   - Predict on labeled monitor input (logs with labels)"
	@echo "make predict-latest    - Predict on most recent session from processed features"
	@echo "make monitor-open      - Open the latest monitoring report in default browser"

install:
	pip install -r requirements.txt

dev: install
	pip install pytest pytest-cov jupyter ipython

test:
	pytest tests/ -v --cov=src --cov=scripts

train:
	python scripts/train.py --config configs/train.yaml

predict:
	@if [ -z "$(input)" ]; then echo "Error: Use 'make predict input=FILE'"; exit 1; fi
	python scripts/predict.py --input $(input)

serve:
	uvicorn scripts.serve:app --host 0.0.0.0 --port 9696 --reload

dashboard:
	streamlit run scripts/dashboard.py --server.port 8501

feature-importance:
	python scripts/feature_importance.py --output figures/feature_importance.html --print-json

monitor-report:
	python scripts/monitor.py --log logs/predictions.csv --output reports/monitor_report.html

data-clear-cache:
	python -c "import os, glob, shutil; [os.remove(p) for p in glob.glob('data/raw/*.parquet') if os.path.exists(p)]; [os.remove(p) for p in glob.glob('data/processed/*.parquet') if os.path.exists(p)]"

retrain-latest: data-clear-cache
	python scripts/train.py --config configs/train.yaml

latest-price:
	python -c "import yfinance as yf, pandas as pd; df=yf.Ticker('BTC-USD').history(period='7d', interval='1h', auto_adjust=True); df=df.rename(columns=str.lower); print(f\"Latest candle: {df.index.max()}  Close: {float(df['close'].iloc[-1]):.2f} USD\")"

predict-sample:
	python scripts/predict.py --input examples/sample_input.csv

build-monitor-input:
	python -c "import json, pandas as pd; df=pd.read_parquet('data/processed/market_profile.parquet').sort_values('date'); feat=json.load(open('models/feature_names.json')); out=df[feat].copy(); out['true_label']=df['breaks_above_vah'].astype(int); out.to_csv('examples/monitor_input.csv', index=False)"

predict-monitor: build-monitor-input
	python scripts/predict.py --input examples/monitor_input.csv

predict-latest:
	python -c "import json, pandas as pd; from scripts.predict import PredictionService; feat=json.load(open('models/feature_names.json')); df=pd.read_parquet('data/processed/market_profile.parquet').sort_values('date'); X=pd.DataFrame([{k: float(df.iloc[-1][k]) for k in feat}]); print(PredictionService('models').predict(X))"

monitor-open:
	python -c "import webbrowser, pathlib; p=pathlib.Path('reports/monitor_report.html'); webbrowser.open_new_tab(p.resolve().as_uri())"

docker-build:
	docker build -t market-profile-ml:latest .

docker-run:
	docker run -it -p 9696:9696 -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data market-profile-ml:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-logs:
	docker logs -f market-profile-api

docker-stop:
	docker stop market-profile-api || true
	docker rm market-profile-api || true

lint:
	flake8 src/ scripts/ --max-line-length=100 --exclude=__pycache__

clean:
	python -c "import os, shutil, glob; [os.remove(f) if os.path.exists(f) else None for root, dirs, files in os.walk('.') for f in [os.path.join(root, file) for file in files if file.endswith('.pyc')]]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/.pytest_cache', recursive=True)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/.ipynb_checkpoints', recursive=True)]; [shutil.rmtree(d, ignore_errors=True) for d in ['build', 'dist'] + glob.glob('*.egg-info') if os.path.exists(d)]"

clean-all: clean
	python -c "import os, shutil, glob; [shutil.rmtree(d, ignore_errors=True) for d in ['data/raw', 'data/processed', 'models'] if os.path.exists(d)]; [os.remove(f) for f in glob.glob('data/raw/*.parquet') + glob.glob('data/processed/*.parquet') + glob.glob('models/*.pkl') + glob.glob('models/*.json') if os.path.exists(f)]"

notebook:
	jupyter notebook notebook.ipynb

