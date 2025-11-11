"""Unit tests for Market Profile feature engineering."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.market_profile import (
    MarketProfileEngine, TechnicalFeatures, engineer_features
)


@pytest.fixture
def sample_daily_data():
    """Create sample OHLCV data for a single trading day."""
    times = pd.date_range('2024-01-15 09:30', periods=24, freq='1H')
    
    data = {
        'timestamp': times,
        'open': np.random.uniform(100, 110, 24),
        'high': np.random.uniform(110, 115, 24),
        'low': np.random.uniform(95, 100, 24),
        'close': np.random.uniform(100, 110, 24),
        'volume': np.random.uniform(1000, 10000, 24)
    }
    
    df = pd.DataFrame(data).set_index('timestamp')
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_multi_day_data():
    """Create sample OHLCV data for multiple days."""
    times = pd.date_range('2024-01-01', periods=240, freq='1H')
    
    data = {
        'timestamp': times,
        'open': np.random.uniform(100, 110, 240),
        'high': np.random.uniform(110, 115, 240),
        'low': np.random.uniform(95, 100, 240),
        'close': np.random.uniform(100, 110, 240),
        'volume': np.random.uniform(1000, 10000, 240)
    }
    
    df = pd.DataFrame(data).set_index('timestamp')
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


class TestMarketProfileEngine:
    """Test Market Profile Engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = MarketProfileEngine(tpo_size=30, vol_percentile=70)
        assert engine.tpo_size == 30
        assert engine.vol_percentile == 70
    
    def test_compute_daily_profile(self, sample_daily_data):
        """Test daily profile computation."""
        engine = MarketProfileEngine()
        profile = engine.compute_daily_profile(sample_daily_data)
        
        # Check required fields
        assert 'poc' in profile
        assert 'vah' in profile
        assert 'val' in profile
        assert 'va_range_width' in profile
        assert 'balance_flag' in profile
        assert 'volume_imbalance' in profile
        assert 'session_volume' in profile
        
        # Check value ranges
        assert profile['poc'] > 0
        assert profile['vah'] > profile['val']
        assert profile['va_range_width'] > 0
        assert profile['balance_flag'] in [0, 1]
        assert 0 <= profile['volume_imbalance'] <= 1
        assert profile['session_volume'] > 0
    
    def test_poc_calculation(self, sample_daily_data):
        """Test POC is within price range."""
        engine = MarketProfileEngine()
        profile = engine.compute_daily_profile(sample_daily_data)
        
        assert sample_daily_data['low'].min() <= profile['poc'] <= sample_daily_data['high'].max()
    
    def test_va_range(self, sample_daily_data):
        """Test Value Area range is within day range."""
        engine = MarketProfileEngine()
        profile = engine.compute_daily_profile(sample_daily_data)
        
        day_low = sample_daily_data['low'].min()
        day_high = sample_daily_data['high'].max()
        
        assert day_low <= profile['val'] <= day_high
        assert day_low <= profile['vah'] <= day_high
        assert profile['val'] < profile['vah']
    
    def test_volume_imbalance(self, sample_daily_data):
        """Test volume imbalance calculation."""
        engine = MarketProfileEngine()
        profile = engine.compute_daily_profile(sample_daily_data)
        
        assert 0 <= profile['volume_imbalance'] <= 1
    
    def test_empty_data(self):
        """Test handling of empty data."""
        engine = MarketProfileEngine()
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        profile = engine.compute_daily_profile(empty_df)
        
        assert profile['poc'] == 0.0
        assert profile['vah'] == 0.0
        assert profile['val'] == 0.0


class TestTechnicalFeatures:
    """Test Technical Features calculation."""
    
    def test_atr_calculation(self, sample_multi_day_data):
        """Test ATR calculation."""
        atr = TechnicalFeatures.calculate_atr(
            sample_multi_day_data['high'],
            sample_multi_day_data['low'],
            sample_multi_day_data['close'],
            period=14
        )
        
        # Check length and NaN handling
        assert len(atr) == len(sample_multi_day_data)
        assert atr.iloc[14:].notna().all()  # Should have values from period onward
    
    def test_rsi_calculation(self, sample_multi_day_data):
        """Test RSI calculation."""
        rsi = TechnicalFeatures.calculate_rsi(
            sample_multi_day_data['close'],
            period=14
        )
        
        # Check length and value range
        assert len(rsi) == len(sample_multi_day_data)
        # RSI should be between 0-100 (after initial NaNs)
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_returns_calculation(self, sample_multi_day_data):
        """Test returns calculation."""
        returns = TechnicalFeatures.calculate_returns(
            sample_multi_day_data['close'],
            periods=[1, 3]
        )
        
        assert 'return_1d' in returns
        assert 'return_3d' in returns
        assert len(returns['return_1d']) == len(sample_multi_day_data)
        assert len(returns['return_3d']) == len(sample_multi_day_data)


class TestFeatureEngineering:
    """Test complete feature engineering pipeline."""
    
    def test_engineer_features_output(self, sample_multi_day_data):
        """Test feature engineering output."""
        features = engineer_features(sample_multi_day_data, target_threshold=0.005)
        
        # Check required columns
        required_cols = [
            'session_poc', 'session_vah', 'session_val',
            'va_range_width', 'balance_flag', 'volume_imbalance',
            'one_day_return', 'three_day_return',
            'atr_14', 'rsi_14', 'session_volume',
            'breaks_above_vah'
        ]
        
        for col in required_cols:
            assert col in features.columns, f"Missing column: {col}"
    
    def test_feature_engineering_target(self, sample_multi_day_data):
        """Test target label is binary."""
        features = engineer_features(sample_multi_day_data)
        
        # Check target values are binary
        assert set(features['breaks_above_vah'].unique()).issubset({0, 1})
    
    def test_feature_engineering_no_nans(self, sample_multi_day_data):
        """Test no NaN values in final features."""
        features = engineer_features(sample_multi_day_data)
        
        # Should have no NaNs after fillna
        assert features.isnull().sum().sum() == 0


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self, sample_multi_day_data):
        """Test full feature engineering pipeline."""
        # This tests the entire flow
        features = engineer_features(sample_multi_day_data)
        
        # Check we got features for multiple days
        assert len(features) > 1
        
        # Check all values are numeric
        for col in features.columns:
            assert pd.api.types.is_numeric_dtype(features[col])
        
        # Check no infinite values
        assert np.isfinite(features.values).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

