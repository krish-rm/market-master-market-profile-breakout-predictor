## Troubleshooting

### Common Issues

#### 1. "Model not found" error when running scripts/predict.py

Problem: `FileNotFoundError: Model not found at models/best_model.pkl`

Solution:
```bash
# Train model first
python scripts/train.py --config configs/train.yaml

# Then predict
python scripts/predict.py --input examples/sample_input.csv
```

#### 2. Yahoo Finance connection error

Problem: `yfinance failed to download data`

Solution:
```bash
# Check internet connection
# Wait a moment (API throttling)
# Or use cached data:
rm data/raw/*.parquet  # Clear cache to retry
python scripts/train.py --config configs/train.yaml
```

#### 3. API returns 503 Service Unavailable

Problem: Prediction service not loaded

Solution:
```bash
# Check models directory exists and has artifacts
ls -la models/

# Make sure model training completed successfully
python scripts/train.py --config configs/train.yaml

# Check logs
docker logs market-profile-api
```

#### 4. Docker build fails

Problem: `ERROR: failed to solve with frontend dockerfile.v0`

Solution:
```bash
# Clear Docker build cache
docker system prune -a

# Rebuild
docker build -t market-profile-ml:latest .
```

#### 5. Out of memory during training

Problem: `MemoryError: Unable to allocate X GiB`

Solution:
- Reduce data period in `configs/train.yaml`
- Use a machine with more RAM
- Process data in smaller batches

#### 6. Port 9696 already in use

Problem: `Address already in use`

Solution:
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


