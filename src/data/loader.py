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
        
        from datetime import datetime, timedelta
        
        # Calculate start/end dates
        end_date = datetime.now()
        if self.period.endswith('y'):
            years = int(self.period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        elif self.period.endswith('mo'):
            months = int(self.period[:-2])
            start_date = end_date - timedelta(days=months * 30)
        elif self.period.endswith('d'):
            days = int(self.period[:-1])
            start_date = end_date - timedelta(days=days)
        else:
            start_date = end_date - timedelta(days=730)  # Default to 2 years
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = None
        methods_tried = []
        
        # Method 1: Try download() with list of tickers (sometimes more reliable)
        try:
            logger.info("Trying method 1: yf.download() with ticker list...")
            methods_tried.append("download(list)")
            data = yf.download(
                [self.ticker],
                start=start_str,
                end=end_str,
                interval=self.interval,
                progress=False,
                show_errors=False
            )
            if not data.empty:
                # Handle MultiIndex from download()
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(1, axis=1)
                logger.info("Method 1 succeeded!")
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")
            data = None
        
        # Method 2: Try Ticker.history() with date strings
        if data is None or data.empty:
            try:
                logger.info("Trying method 2: Ticker.history() with date strings...")
                methods_tried.append("Ticker.history(dates)")
                ticker_obj = yf.Ticker(self.ticker)
                data = ticker_obj.history(
                    start=start_str,
                    end=end_str,
                    interval=self.interval,
                    auto_adjust=True,
                    prepost=False
                )
                if not data.empty:
                    logger.info("Method 2 succeeded!")
            except Exception as e:
                logger.debug(f"Method 2 failed: {e}")
                data = None
        
        # Method 3: Try Ticker.history() with period (may fail but worth trying)
        if data is None or data.empty:
            try:
                logger.info("Trying method 3: Ticker.history() with period...")
                methods_tried.append("Ticker.history(period)")
                ticker_obj = yf.Ticker(self.ticker)
                data = ticker_obj.history(
                    period=self.period,
                    interval=self.interval,
                    auto_adjust=True,
                    prepost=False
                )
                if not data.empty:
                    logger.info("Method 3 succeeded!")
            except Exception as e:
                logger.debug(f"Method 3 failed: {e}")
                data = None
        
        # Method 4: Try download() with period (last resort)
        if data is None or data.empty:
            try:
                logger.info("Trying method 4: yf.download() with period...")
                methods_tried.append("download(period)")
                data = yf.download(
                    self.ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False,
                    show_errors=False
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data.droplevel(1, axis=1)
                    logger.info("Method 4 succeeded!")
            except Exception as e:
                logger.debug(f"Method 4 failed: {e}")
                data = None
        
        if data is None or data.empty:
            error_msg = f"No data returned for {self.ticker} after trying methods: {', '.join(methods_tried)}"
            logger.error(error_msg)
            logger.error("This may be due to:")
            logger.error("1. Yahoo Finance API issues")
            logger.error("2. Network connectivity problems")
            logger.error("3. yfinance library datetime bug (try: pip install --upgrade yfinance)")
            raise ValueError(error_msg)
        
        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Drop extra columns that history() may return (Dividends, Stock Splits)
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in columns_to_keep if col in data.columns]]
        
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

