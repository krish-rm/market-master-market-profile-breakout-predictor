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

### Feature Importance Visualization

Understand which features drive predictions and their relative importance:

```bash
make feature-importance
# or
python scripts/feature_importance.py --output figures/feature_importance.html --top 15 --print-json
```

This generates an interactive bar chart at `figures/feature_importance.html` and optionally prints a JSON list of features sorted by importance.

### Code Style

```bash
# Lint code
flake8 src/ scripts/ --max-line-length=100 --exclude=__pycache__

# Format code (optional)
black src/ scripts/
```

### Adding New Features

1. Feature Calculation: add to `src/features/market_profile.py`
2. Pipeline Integration: update `engineer_features()` function
3. Configuration: add to `configs/train.yaml`
4. Tests: add unit tests to `tests/test_market_profile.py`
5. API Schema: update `configs/api_schema.json` if adding to API


