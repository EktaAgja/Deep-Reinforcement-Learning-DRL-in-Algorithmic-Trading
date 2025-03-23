import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
def load_model(model_path="/home/ankit/Desktop/Ingenium/tdqn_trading_model.pth"):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    model = TDQNModel(input_size=9, action_dim=3, hidden_size=128)  # Ensure input_size=9
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)
    model.eval()
    return model

# Convert stock data into model input
def preprocess_input(stock_features):
    return torch.tensor(stock_features, dtype=torch.float32).unsqueeze(0)

# Predict stock action
def predict_action(model, stock_features):
    input_tensor = preprocess_input(stock_features)
    with torch.no_grad():
        action_values = model(input_tensor)
    action = torch.argmax(action_values).item()
    return ["Sell", "Hold", "Buy"][action]