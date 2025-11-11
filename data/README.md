# Data Documentation

## Overview

This directory contains raw and processed data for the Market Master – Market Profile Breakout Predictor project.

### Directory Structure

```
data/
├── raw/               # Raw OHLCV data from Yahoo Finance
├── processed/         # Engineered features for model training
└── README.md          # This file
```

## Data Source

**Provider:** Yahoo Finance  
**Ticker:** BTC-USD (Bitcoin vs US Dollar)  
**Interval:** 1 hour  
**Historical Period:** ~24 months  

### Data Collection

The raw data is automatically fetched using `yfinance` when running `scripts/train.py`:

```bash
python scripts/train.py --config configs/train.yaml
```

Data is cached locally to avoid repeated downloads.

## Schema

### Raw Data (OHLCV)

| Column  | Type    | Description                    |
|---------|---------|--------------------------------|
| open    | float   | Opening price for the candle   |
| high    | float   | Highest price during candle    |
| low     | float   | Lowest price during candle     |
| close   | float   | Closing price for the candle   |
| volume  | integer | Trading volume during candle   |

### Processed Data (Features)

Features are engineered at the daily session level from hourly candles:

| Column              | Type    | Description                                     |
|---------------------|---------|-----------------------------------------------  |
| session_poc         | float   | Point of Control (highest volume price)         |
| session_vah         | float   | Value Area High (70th percentile)               |
| session_val         | float   | Value Area Low (30th percentile)                |
| va_range_width      | float   | Value Area range width (VAH - VAL)              |
| balance_flag        | int     | 1 if balanced session, 0 otherwise              |
| volume_imbalance    | float   | Ratio of upside volume to total volume          |
| one_day_return      | float   | 1-day lagged return                             |
| three_day_return    | float   | 3-day lagged return                             |
| atr_14              | float   | Average True Range (14-period)                  |
| rsi_14              | float   | Relative Strength Index (14-period)             |
| session_volume      | float   | Total volume for the session                    |
| breaks_above_vah    | int     | Target: 1 if next day breaks above current VAH |

## Data Validation

Quality checks are performed during data loading:

1. **Completeness:** No missing values after removal of incomplete candles
2. **Consistency:** OHLC relationships validated (High ≥ Close/Open, Low ≤ Close/Open)
3. **Duplicates:** Removed duplicate timestamps
4. **Volume:** Only positive volume retained

See `src/data/loader.py::DataLoader.validate_data()` for implementation details.

## Preprocessing Steps

1. **Cleaning:** Remove NaN values, duplicates, invalid prices
2. **Market Profile:** Partition daily sessions into 30-minute TPOs
3. **Feature Extraction:** Calculate POC, VAH, VAL, technical indicators
4. **Normalization:** StandardScaler applied during model training

## Data Refresh

To refresh data with latest values:

```bash
rm data/raw/BTC-USD_2y_1h.parquet  # Remove cache
python scripts/train.py --config configs/train.yaml  # Re-fetch and retrain
```

## Target Variable

**breaks_above_vah:** Binary classification target

- **1 (Positive):** Next day's high breaks above current session's VAH by ≥ 0.5%
- **0 (Negative):** Otherwise

Computed as:
```
(next_day_high - vah) / vah >= 0.005
```

## Size & Distribution

- **Time Series Length:** ~18 months of hourly data
- **Daily Samples:** ~365 (one feature vector per trading day)
- **Total Features:** 11 input features + 1 target
- **Missing Values:** None (after preprocessing)

## Storage Format

- **Raw:** Parquet (efficient compression and random access)
- **Processed:** Parquet (same advantages)

Both can be loaded with:

```python
import pandas as pd
df = pd.read_parquet('data/processed/market_profile.parquet')
```

## Privacy & Attribution

Data is sourced from Yahoo Finance public API. No personal data is used.

## Reproducibility

All data operations are deterministic:

1. Ticker and period are fixed in `configs/train.yaml`
2. Feature engineering uses deterministic algorithms
3. Random seed (42) ensures consistent train/test splits

To reproduce exact results:
1. Use the same `configs/train.yaml`
2. Install exact dependency versions from `requirements.txt`
3. Use the same random seed (already configured)

## Troubleshooting

### Yahoo Finance Connection Issues

**Error:** "Failed to download data"

**Solution:**
```bash
# Check internet connection
# Wait and retry (API throttling)
# Or manually download and place in data/raw/
```

### Out of Memory

**Error:** "MemoryError during data loading"

**Solution:**
- Reduce `period` in `configs/train.yaml` (e.g., "1y")
- Process data in chunks
- Use a machine with more RAM

### Data Validation Failures

Check `src/data/loader.py::DataLoader.validate_data()` for details on what failed.

## Contact & Support

For data-related issues, check:
1. Yahoo Finance API status
2. Network connectivity
3. Disk space availability

