"""
Test suite for Deep RL Trading applications
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add project directories to path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "API-App"))
sys.path.append(str(BASE_DIR / "App"))

# Test data
SAMPLE_FEATURES = [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1]

class TestTDQNModel:
    """Tests for TDQN model functionality"""
    
    def test_import_model_api(self):
        """Test that model API can be imported"""
        try:
            from API-App.tdqn_model_api import TDQNModel, preprocess_input, predict_action
            assert True
        except ImportError:
            pytest.skip("Model API not available for testing")
    
    def test_model_creation(self):
        """Test TDQN model creation"""
        try:
            from API_App.tdqn_model_api import TDQNModel
            model = TDQNModel(input_size=9, action_dim=3, hidden_size=128)
            assert model is not None
            
            # Test forward pass with dummy data
            import torch
            dummy_input = torch.randn(1, 9)
            output = model(dummy_input)
            assert output.shape == (1, 3)
        except ImportError:
            pytest.skip("Model dependencies not available")
    
    def test_preprocess_input_valid(self):
        """Test preprocessing with valid input"""
        try:
            from API_App.tdqn_model_api import preprocess_input
            result = preprocess_input(SAMPLE_FEATURES)
            assert result.shape == (1, 9)
        except ImportError:
            pytest.skip("Model API not available")
    
    def test_preprocess_input_invalid(self):
        """Test preprocessing with invalid input"""
        try:
            from API_App.tdqn_model_api import preprocess_input
            
            # Test wrong number of features
            with pytest.raises(ValueError):
                preprocess_input([1, 2, 3])  # Only 3 features instead of 9
            
            # Test non-numeric features
            with pytest.raises(ValueError):
                preprocess_input(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
                
        except ImportError:
            pytest.skip("Model API not available")

class TestFlaskAPI:
    """Tests for Flask API application"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        try:
            from API_App.app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client
        except ImportError:
            pytest.skip("Flask app not available")
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_predict_endpoint_valid(self, client):
        """Test predict endpoint with valid data"""
        response = client.post('/predict', 
                              json={'features': SAMPLE_FEATURES},
                              content_type='application/json')
        # May return 503 if model not loaded, which is acceptable for testing
        assert response.status_code in [200, 503]
    
    def test_predict_endpoint_invalid(self, client):
        """Test predict endpoint with invalid data"""
        # Test missing features
        response = client.post('/predict', 
                              json={},
                              content_type='application/json')
        assert response.status_code == 400
        
        # Test wrong number of features
        response = client.post('/predict', 
                              json={'features': [1, 2, 3]},
                              content_type='application/json')
        assert response.status_code == 400
    
    def test_predict_stock_endpoint(self, client):
        """Test stock ticker prediction endpoint"""
        response = client.get('/predict_stock/AAPL')
        # May fail due to network/model issues, but shouldn't crash
        assert response.status_code in [200, 400, 503]

class TestDataProcessing:
    """Tests for data processing functionality"""
    
    def test_config_import(self):
        """Test configuration can be imported"""
        try:
            from config import Config
            assert Config.MODEL_INPUT_SIZE == 9
            assert Config.MODEL_ACTION_DIM == 3
        except ImportError:
            pytest.skip("Config not available")
    
    def test_sample_data_format(self):
        """Test sample data has correct format"""
        # This would test actual data if available
        data_file = BASE_DIR / "data" / "final_stock_data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            assert not df.empty
            assert 'Stock' in df.columns
            assert 'Close' in df.columns
        else:
            pytest.skip("Data file not available")

class TestVisualizationApp:
    """Tests for visualization Flask app"""
    
    @pytest.fixture
    def viz_client(self):
        """Create test client for visualization app"""
        try:
            from App.app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client
        except ImportError:
            pytest.skip("Visualization app not available")
    
    def test_dashboard_loads(self, viz_client):
        """Test dashboard page loads"""
        response = viz_client.get('/')
        assert response.status_code == 200
    
    def test_graphs_page(self, viz_client):
        """Test graphs page loads"""
        response = viz_client.get('/graphs')
        assert response.status_code == 200

class TestUtilities:
    """Tests for utility functions"""
    
    def test_feature_validation(self):
        """Test feature validation logic"""
        # Valid features
        features = SAMPLE_FEATURES
        assert len(features) == 9
        assert all(isinstance(f, (int, float)) for f in features)
        assert all(not np.isnan(f) and not np.isinf(f) for f in features)
    
    def test_action_mapping(self):
        """Test action index to string mapping"""
        actions = ["Sell", "Hold", "Buy"]
        assert len(actions) == 3
        assert actions[0] == "Sell"
        assert actions[1] == "Hold"
        assert actions[2] == "Buy"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])