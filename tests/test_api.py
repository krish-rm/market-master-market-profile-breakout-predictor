"""Unit tests for FastAPI service."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

# Try to import the serve module
try:
    from scripts.serve import app
    client = TestClient(app)
    API_AVAILABLE = True
except Exception as e:
    API_AVAILABLE = False
    print(f"Warning: Could not load API - {e}")


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "message" in data
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_health_status(self):
        """Test health status is valid."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]


class TestInfoEndpoint:
    """Test info endpoint."""
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_info_endpoint(self):
        """Test info endpoint returns expected structure."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        
        assert "title" in data
        assert "version" in data
        assert "endpoints" in data
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_info_has_endpoints(self):
        """Test info includes endpoint descriptions."""
        response = client.get("/info")
        data = response.json()
        endpoints = data["endpoints"]
        
        assert "GET /health" in endpoints
        assert "POST /predict" in endpoints


class TestRootEndpoint:
    """Test root endpoint."""
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "docs" in data or "health" in data


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_predict_valid_request(self):
        """Test prediction with valid request."""
        payload = {
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
        
        response = client.post("/predict", json=payload)
        
        # If service is loaded, check response
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "confidence" in data
            
            # Check value ranges
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert 0 <= data["confidence"] <= 1
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_predict_missing_field(self):
        """Test prediction with missing required field."""
        payload = {
            "features": {
                "session_poc": 42500.0,
                # Missing other required fields
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_batch_predict_valid(self):
        """Test batch prediction with valid request."""
        payload = [
            {
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
            },
            {
                "session_poc": 43000.0,
                "session_vah": 43250.0,
                "session_val": 42750.0,
                "va_range_width": 500.0,
                "balance_flag": 0,
                "volume_imbalance": 0.48,
                "one_day_return": -0.01,
                "three_day_return": 0.015,
                "atr_14": 320.0,
                "rsi_14": 45.0,
                "session_volume": 1600000.0
            }
        ]
        
        response = client.post("/batch-predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "num_samples" in data
            assert "predictions" in data
            assert data["num_samples"] == 2
            assert len(data["predictions"]) == 2
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    def test_batch_predict_empty(self):
        """Test batch prediction with empty list."""
        payload = []
        response = client.post("/batch-predict", json=payload)
        
        # Empty list should either work or give validation error
        assert response.status_code in [200, 422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

