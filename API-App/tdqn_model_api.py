import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the path to model file dynamically
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "tdqn_trading_model.pth"

# Define the TDQN Model
class TDQNModel(nn.Module):
    def __init__(self, input_size=9, action_dim=3, hidden_size=128):
        super(TDQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
def load_model(model_path=None):
    """Load the TDQN model from file"""
    if model_path is None:
        model_path = MODEL_PATH
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        logger.info(f"Model checkpoint loaded from: {model_path}")
        
        model = TDQNModel(input_size=9, action_dim=3, hidden_size=128)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        if not pretrained_dict:
            logger.warning("No matching parameters found in checkpoint")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Convert stock data into model input
def preprocess_input(stock_features):
    """Convert stock features to tensor format"""
    try:
        if not isinstance(stock_features, (list, np.ndarray)):
            raise ValueError("Stock features must be a list or numpy array")
        
        if len(stock_features) != 9:
            raise ValueError(f"Expected 9 features, got {len(stock_features)}")
        
        # Ensure all features are numeric
        features = [float(f) for f in stock_features]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

# Predict stock action
def predict_action(model, stock_features):
    """Predict trading action from stock features"""
    try:
        input_tensor = preprocess_input(stock_features)
        
        with torch.no_grad():
            action_values = model(input_tensor)
        
        action_idx = torch.argmax(action_values).item()
        actions = ["Sell", "Hold", "Buy"]
        
        if action_idx < 0 or action_idx >= len(actions):
            logger.warning(f"Invalid action index: {action_idx}")
            return "Hold"  # Default to Hold for safety
        
        predicted_action = actions[action_idx]
        logger.info(f"Predicted action: {predicted_action}")
        return predicted_action
    
    except Exception as e:
        logger.error(f"Error predicting action: {e}")
        return "Hold"  # Default to Hold on error