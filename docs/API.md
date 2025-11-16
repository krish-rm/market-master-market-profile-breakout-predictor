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

- prediction: 0 (won't break) or 1 (will break)
- probability: Predicted probability of breakout (0-1)
- confidence: Max(probability, 1-probability)

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


