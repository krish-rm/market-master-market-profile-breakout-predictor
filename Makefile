.PHONY: help install dev test train predict serve docker-build docker-run docker-stop clean lint

help:
	@echo "Market Master – Market Profile Breakout Predictor - Available Commands"
	@echo "========================================"
	@echo "make install       - Install dependencies"
	@echo "make dev           - Install dev dependencies"
	@echo "make test          - Run unit tests"
	@echo "make train         - Train model"
	@echo "make predict       - Make predictions (requires --input=FILE)"
	@echo "make serve         - Start FastAPI server locally"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run Docker container"
	@echo "make docker-stop   - Stop Docker container"
	@echo "make lint          - Run linting"
	@echo "make clean         - Clean up artifacts"

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
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.ipynb_checkpoints' -delete
	rm -rf build/ dist/ *.egg-info/

notebook:
	jupyter notebook notebook.ipynb

