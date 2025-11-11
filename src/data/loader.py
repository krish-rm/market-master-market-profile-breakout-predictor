"""Data loading and preprocessing utilities."""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess market data from Yahoo Finance."""
    
    def __init__(self, ticker: str = "BTC-USD", period: str = "2y", interval: str = "1h"):
        """
        Initialize DataLoader.
        
        Args:
            ticker: Stock ticker (default BTC-USD)
            period: Data period (default 2y)
            interval: Candle interval (default 1h)
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
    
    def fetch_data(self, use_cache: bool = True, cache_dir: str = "data/raw") -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_dir: Directory to cache raw data
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_path = Path(cache_dir) / f"{self.ticker}_{self.period}_{self.interval}.parquet"
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Fetch from Yahoo Finance
        logger.info(f"Fetching data for {self.ticker} from Yahoo Finance...")
        try:
            data = yf.download(
                self.ticker,
                period=self.period,
                interval=self.interval,
                progress=False
            )
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Clean column names (lowercase)
        data.columns = data.columns.str.lower()
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path)
        logger.info(f"Data saved to {cache_path}")
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing_cols = required_cols - set(data.columns.str.lower())
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for duplicates
        if data.index.duplicated().any():
            issues.append(f"Duplicate timestamps: {data.index.duplicated().sum()}")
        
        # Check for missing values
        missing_pct = (data.isnull().sum() / len(data) * 100)
        if (missing_pct > 0).any():
            issues.append(f"Missing values: {missing_pct[missing_pct > 0].to_dict()}")
        
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(f"Column {col} is not numeric")
        
        # Check price order
        invalid_prices = (data['high'] < data['low']).sum() + \
                         (data['high'] < data['close']).sum() + \
                         (data['low'] > data['close']).sum()
        if invalid_prices > 0:
            issues.append(f"Invalid price relationships: {invalid_prices} rows")
        
        return len(issues) == 0, issues
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            data: Raw data to clean
        
        Returns:
            Cleaned DataFrame
        """
        data = data.copy()
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with invalid prices
        data = data[
            (data['high'] >= data['low']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['close'])
        ]
        
        # Ensure positive volume
        data = data[data['volume'] > 0]
        
        return data
    
    def save_processed_data(
        self, 
        data: pd.DataFrame, 
        output_dir: str = "data/processed",
        filename: str = "market_profile.parquet"
    ) -> Path:
        """
        Save processed data to disk.
        
        Args:
            data: DataFrame to save
            output_dir: Output directory
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_path)
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path


def load_data(
    ticker: str = "BTC-USD",
    period: str = "2y",
    interval: str = "1h",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load and clean data.
    
    Args:
        ticker: Stock ticker
        period: Data period
        interval: Candle interval
        use_cache: Whether to use cached data
    
    Returns:
        Cleaned DataFrame
    """
    loader = DataLoader(ticker=ticker, period=period, interval=interval)
    data = loader.fetch_data(use_cache=use_cache)
    
    is_valid, issues = loader.validate_data(data)
    if not is_valid:
        logger.warning(f"Data validation issues: {issues}")
    
    data = loader.clean_data(data)
    
    return data

