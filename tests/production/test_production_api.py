"""
Production API tests for SeaSeeAI
"""

import pytest
import requests
import time
import json
from datetime import datetime, timedelta
import numpy as np

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestProductionAPI:
    """Production API test suite"""
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] == True
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'SeaSeeAI' in data['message']
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = requests.get(f"{BASE_URL}/model/info", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data['model_type'] == 'TrAISformer'
        assert data['loaded'] == True
        assert data['parameters'] > 0
    
    def _generate_test_observations(self, count: int, offset: float = 0.0):
        """Generate test AIS observations"""
        observations = []
        base_lat = 37.7749 + offset * 0.01
        base_lon = -122.4194 + offset * 0.01
        
        for i in range(count):
            observations.append({
                'mmsi': 123456789,
                'timestamp': (datetime.now() - timedelta(hours=count-i)).isoformat(),
                'latitude': base_lat + i * 0.001,
                'longitude': base_lon + i * 0.001,
                'sog': 10.0 + i * 0.1,
                'cog': 45.0 + i * 1.0
            })
        
        return observations

    def test_single_prediction(self):
        """Test single prediction endpoint"""
        # Generate test data
        test_data = self._generate_test_observations(20)
        
        prediction_request = {
            'observations': test_data,
            'prediction_horizon': 5
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_request,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert 'predictions' in data
        assert 'confidence' in data
        assert 'processing_time' in data
        assert 'model_version' in data
        
        # Validate predictions
        assert len(data['predictions']) == 5
        for pred in data['predictions']:
            assert 'latitude' in pred
            assert 'longitude' in pred
            assert 'sog' in pred
            assert 'cog' in pred
            assert 'step' in pred
            assert 'timestamp' in pred
            
            # Validate value ranges
            assert -90 <= pred['latitude'] <= 90
            assert -180 <= pred['longitude'] <= 180
            assert 0 <= pred['sog'] <= 50
            assert 0 <= pred['cog'] <= 360
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        batch_requests = []
        
        # Generate multiple test requests
        for i in range(3):
            test_data = self._generate_test_observations(20, offset=i*10)
            batch_requests.append({
                'observations': test_data,
                'prediction_horizon': 3
            })
        
        batch_request = {
            'requests': batch_requests
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_request,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'results' in data
        assert 'total_processing_time' in data
        assert len(data['results']) == 3
        
        for result in data['results']:
            assert len(result['predictions']) == 3
    
    def test_prediction_performance(self):
        """Test prediction performance"""
        test_data = self._generate_test_observations(20)
        
        prediction_request = {
            'observations': test_data,
            'prediction_horizon': 5
        }
        
        # Measure response time
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_request,
            timeout=TEST_TIMEOUT
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        data = response.json()
        
        # Performance requirements (adjust based on your needs)
        assert response_time < 1.0  # Should respond within 1 second
        assert data['processing_time'] < 0.5  # Model inference should be fast
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test with insufficient observations
        test_data = self._generate_test_observations(5)  # Too few
        
        prediction_request = {
            'observations': test_data,
            'prediction_horizon': 5
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_request,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 400
        
        # Test with invalid data
        invalid_request = {
            'observations': [],
            'prediction_horizon': 5
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_request,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 400
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        test_data = self._generate_test_observations(20)
        
        prediction_request = {
            'observations': test_data,
            'prediction_horizon': 3
        }
        
        def make_request():
            response = requests.post(
                f"{BASE_URL}/predict",
                json=prediction_request,
                timeout=TEST_TIMEOUT
            )
            return response.status_code
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(result == 200 for result in results)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
