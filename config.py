"""
Configuration management for Deep RL Trading applications
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    """Base configuration class"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Data paths
    DATA_DIR = BASE_DIR / 'data'
    STOCK_DATA_FILE = DATA_DIR / 'final_stock_data.csv'
    PROCESSED_DATA_FILE = DATA_DIR / 'processed_stock_data.csv'
    
    # Model paths
    MODEL_DIR = BASE_DIR
    TDQN_MODEL_PATH = MODEL_DIR / 'tdqn_trading_model.pth'
    
    # Static files
    STATIC_DIR = BASE_DIR / 'App' / 'static'
    PLOTS_DIR = STATIC_DIR / 'plots'
    
    # API settings
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '1000/hour')
    
    # Trading settings
    DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
    STOCK_DATA_PERIOD = '5d'  # For yfinance
    
    # Model settings
    MODEL_INPUT_SIZE = 9
    MODEL_ACTION_DIM = 3
    MODEL_HIDDEN_SIZE = 128
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-required'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}