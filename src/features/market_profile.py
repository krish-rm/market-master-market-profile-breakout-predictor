"""Market Profile feature engineering module."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class MarketProfileEngine:
    """
    Market Profile analysis engine.
    
    Computes Point of Control (POC), Value Area High (VAH), Value Area Low (VAL),
    and other market profile features from hourly OHLCV data.
    """
    
    def __init__(self, tpo_size: int = 30, vol_percentile: float = 70):
        """
        Initialize Market Profile Engine.
        
        Args:
            tpo_size: Time Price Opportunity size in minutes (default 30)
            vol_percentile: Volume percentile for VA calculation (default 70)
        """
        self.tpo_size = tpo_size
        self.vol_percentile = vol_percentile
    
    def compute_daily_profile(
        self, 
        daily_data: pd.DataFrame
    ) -> Dict:
        """
        Compute market profile for a single trading day.
        
        Args:
            daily_data: DataFrame with columns [timestamp, open, high, low, close, volume]
                       All candles should be from the same day
        
        Returns:
            Dictionary with POC, VAH, VAL, balance flag, and other metrics
        """
        if daily_data.empty:
            return self._empty_profile()
        
        # Create price bins (e.g., $1 bins or use decimal precision)
        price_bins = self._create_price_bins(daily_data)
        
        # Distribute volume to price bins based on OHLC
        volume_dist = self._distribute_volume(daily_data, price_bins)
        
        # Calculate Profile of Control (POC) - price with highest volume
        poc_price = price_bins[np.argmax(volume_dist)]
        poc_volume = np.max(volume_dist)
        
        # Calculate Value Area (70% of total volume)
        va_range = self._calculate_value_area(price_bins, volume_dist)
        vah = va_range['high']
        val = va_range['low']
        va_width = vah - val
        
        # Calculate balance flag (1 if POC in middle 50% of range, 0 otherwise)
        day_high = daily_data['high'].max()
        day_low = daily_data['low'].min()
        day_range = day_high - day_low
        poc_range_pct = (poc_price - day_low) / (day_range + 1e-6)
        balance_flag = 1 if 0.25 <= poc_range_pct <= 0.75 else 0
        
        # Volume imbalance (buying vs selling pressure)
        volume_imbalance = self._calculate_volume_imbalance(daily_data)
        
        # Total session volume
        session_volume = daily_data['volume'].sum()
        
        return {
            'poc': poc_price,
            'poc_volume': poc_volume,
            'vah': vah,
            'val': val,
            'va_range_width': va_width,
            'balance_flag': balance_flag,
            'volume_imbalance': volume_imbalance,
            'session_volume': session_volume,
            'day_high': day_high,
            'day_low': day_low,
            'profile_type': self._classify_profile_type(poc_price, vah, val, day_high, day_low)
        }
    
    def _create_price_bins(self, daily_data: pd.DataFrame, bin_size: float = 1.0) -> np.ndarray:
        """Create price bins for volume distribution."""
        day_high = daily_data['high'].max()
        day_low = daily_data['low'].min()
        
        # Create bins with 1 unit precision (adjust as needed)
        num_bins = int((day_high - day_low) / bin_size) + 2
        bins = np.linspace(day_low - bin_size, day_high + bin_size, num_bins)
        return bins
    
    def _distribute_volume(
        self, 
        daily_data: pd.DataFrame, 
        price_bins: np.ndarray
    ) -> np.ndarray:
        """Distribute volume from each candle to price bins."""
        volume_dist = np.zeros(len(price_bins) - 1)
        
        for _, row in daily_data.iterrows():
            candle_vol = row['volume']
            candle_high = row['high']
            candle_low = row['low']
            candle_range = candle_high - candle_low + 1e-6
            
            # Find bins that overlap with this candle
            for i, (bin_start, bin_end) in enumerate(zip(price_bins[:-1], price_bins[1:])):
                # Check if bin overlaps with candle range
                if bin_end > candle_low and bin_start < candle_high:
                    # Calculate overlap proportion
                    overlap_start = max(bin_start, candle_low)
                    overlap_end = min(bin_end, candle_high)
                    overlap_pct = (overlap_end - overlap_start) / candle_range
                    volume_dist[i] += candle_vol * overlap_pct
        
        return volume_dist
    
    def _calculate_value_area(
        self, 
        price_bins: np.ndarray, 
        volume_dist: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Value Area High and Low (area containing 70% of volume).
        
        Strategy: Sort prices by volume and accumulate until reaching target %.
        """
        total_volume = volume_dist.sum()
        target_volume = total_volume * (self.vol_percentile / 100.0)
        
        # Get indices sorted by volume (descending)
        sorted_indices = np.argsort(-volume_dist)
        
        # Accumulate volume from highest to lowest
        accumulated_vol = 0
        va_indices = []
        for idx in sorted_indices:
            va_indices.append(idx)
            accumulated_vol += volume_dist[idx]
            if accumulated_vol >= target_volume:
                break
        
        # Get min and max price for VA
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        va_prices = bin_centers[va_indices]
        
        vah = va_prices.max()
        val = va_prices.min()
        
        return {'high': vah, 'low': val}
    
    def _calculate_volume_imbalance(self, daily_data: pd.DataFrame) -> float:
        """
        Calculate volume imbalance (buying vs selling pressure).
        
        Simple heuristic: use close vs open to estimate direction,
        accumulate upbar volumes vs downbar volumes.
        """
        upside_vol = daily_data[daily_data['close'] >= daily_data['open']]['volume'].sum()
        downside_vol = daily_data[daily_data['close'] < daily_data['open']]['volume'].sum()
        total_vol = upside_vol + downside_vol
        
        if total_vol == 0:
            return 0.5
        
        return upside_vol / total_vol
    
    def _classify_profile_type(
        self, 
        poc: float, 
        vah: float, 
        val: float, 
        day_high: float, 
        day_low: float
    ) -> str:
        """
        Classify profile type: P-profile (balanced), b-profile (bottom), t-profile (top).
        """
        poc_pct = (poc - val) / (vah - val + 1e-6)
        
        if 0.3 < poc_pct < 0.7:
            return "P"  # Balanced
        elif poc_pct <= 0.3:
            return "b"  # Bottom (selling pressure)
        else:
            return "t"  # Top (buying pressure)
    
    def _empty_profile(self) -> Dict:
        """Return empty profile structure."""
        return {
            'poc': 0.0,
            'poc_volume': 0.0,
            'vah': 0.0,
            'val': 0.0,
            'va_range_width': 0.0,
            'balance_flag': 0,
            'volume_imbalance': 0.5,
            'session_volume': 0.0,
            'day_high': 0.0,
            'day_low': 0.0,
            'profile_type': 'N'
        }


class TechnicalFeatures:
    """Compute technical indicators for feature engineering."""
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: Lookback period (default 14)
        
        Returns:
            Series of ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            close: Series of close prices
            period: Lookback period (default 14)
        
        Returns:
            Series of RSI values
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_returns(close: pd.Series, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate lagged returns.
        
        Args:
            close: Series of close prices
            periods: List of periods to calculate (default [1, 3])
        
        Returns:
            Dictionary of return series
        """
        if periods is None:
            periods = [1, 3]
        
        returns = {}
        for period in periods:
            ret = close.pct_change(periods=period)
            returns[f'return_{period}d'] = ret
        
        return returns


def engineer_features(
    data: pd.DataFrame,
    target_threshold: float = 0.005
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        data: DataFrame with hourly OHLCV data (indexed by timestamp)
        target_threshold: Breakout threshold (default 0.5%)
    
    Returns:
        DataFrame with engineered features and target label
    """
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    data = data.sort_index()
    
    # Initialize engines
    mp_engine = MarketProfileEngine(tpo_size=30, vol_percentile=70)
    
    # Group by date and compute market profile for each session
    features_list = []
    
    for date, group in data.groupby(data.index.date):
        profile = mp_engine.compute_daily_profile(group)
        
        # Add timestamp for this session
        profile['date'] = date
        profile['timestamp'] = group.index[-1]  # Last candle of the day
        
        features_list.append(profile)
    
    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index('timestamp')
    
    # Add technical indicators (must use full data, then align)
    tech = TechnicalFeatures()
    atr = tech.calculate_atr(data['high'], data['low'], data['close'], period=14)
    rsi = tech.calculate_rsi(data['close'], period=14)
    
    returns_dict = tech.calculate_returns(data['close'], periods=[1, 3])
    
    # Resample technical indicators to daily (use last value of the day)
    daily_atr = atr.resample('D').last()
    daily_rsi = rsi.resample('D').last()
    
    daily_returns_1d = returns_dict['return_1d'].resample('D').last()
    daily_returns_3d = returns_dict['return_3d'].resample('D').last()
    
    # Align with features (use date for merging)
    features_df['atr_14'] = daily_atr.values[:len(features_df)]
    features_df['rsi_14'] = daily_rsi.values[:len(features_df)]
    features_df['one_day_return'] = daily_returns_1d.values[:len(features_df)]
    features_df['three_day_return'] = daily_returns_3d.values[:len(features_df)]
    
    # Create binary target: breaks_above_vah
    # Shift forward to get next day's high
    next_day_high = data['high'].resample('D').max().shift(-1)
    
    features_df['next_day_high'] = next_day_high.values[:len(features_df)]
    features_df['breaks_above_vah'] = (
        (features_df['next_day_high'] - features_df['vah']) / features_df['vah'] >= target_threshold
    ).astype(int)
    
    # Remove last row (no next day data)
    features_df = features_df.iloc[:-1]
    
    # Fill NaN values
    features_df = features_df.bfill().ffill()
    
    # Rename columns for consistency
    features_df = features_df.rename(columns={
        'poc': 'session_poc',
        'vah': 'session_vah',
        'val': 'session_val'
    })
    
    return features_df

