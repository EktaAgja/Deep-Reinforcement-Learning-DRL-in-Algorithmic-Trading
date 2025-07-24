from flask import Flask, request, render_template, jsonify
import numpy as np
import yfinance as yf
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from API-App.tdqn_model_api import load_model, predict_action
except ImportError:
    from tdqn_model_api import load_model, predict_action

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Global model variable
model = None

def initialize_model():
    """Initialize the model safely"""
    global model
    try:
        model = load_model()
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

# Initialize model on startup
if not initialize_model():
    logger.warning("Starting without model - predictions will not work")

# Home route - UI Page
@app.route("/")
def home():
    return render_template("index.html")

def get_stock_features(ticker):
    """Fetch and process stock features from Yahoo Finance"""
    try:
        logger.info(f"Fetching stock data for: {ticker}")  

        # Validate ticker
        if not ticker or not isinstance(ticker, str):
            logger.error("Invalid ticker provided")
            return None

        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="5d")

        logger.info(f"Stock data retrieved: {len(hist)} records")  

        if hist.empty or len(hist) < 2:  
            logger.warning("Stock data is empty or insufficient data")
            return None  

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]

        # Calculate features with proper error handling
        try:
            price_change = (latest["Close"] - prev["Close"]) / prev["Close"] if prev["Close"] != 0 else 0
            price_diff = latest["Close"] - latest["Open"]
            high_low_diff = latest["High"] - latest["Low"]
            volume_ratio = latest["Volume"] / prev["Volume"] if prev["Volume"] != 0 else 1

            features = [
                float(latest["Close"]), 
                float(latest["Volume"]), 
                float(latest["Open"]),
                float(latest["High"]), 
                float(latest["Low"]),
                float(price_change),
                float(price_diff), 
                float(high_low_diff),
                float(volume_ratio)
            ]

            # Validate features
            if any(not isinstance(f, (int, float)) or np.isnan(f) or np.isinf(f) for f in features):
                logger.error("Invalid features calculated")
                return None

            logger.info(f"Extracted Features: {features}")
            return features  

        except Exception as calc_error:
            logger.error(f"Error calculating features: {calc_error}")
            return None

    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return None  

@app.route("/predict", methods=["POST"])
def predict():
    """Predict trading action for given stock features"""
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 503

        data = request.json  
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        stock_features = data.get("features")

        if not stock_features:
            return jsonify({"error": "No features provided"}), 400

        if not isinstance(stock_features, list) or len(stock_features) != 9:
            return jsonify({"error": "Invalid input. Expected 9 stock features as a list."}), 400

        # Validate features are numeric
        try:
            stock_features = [float(f) for f in stock_features]
        except (ValueError, TypeError):
            return jsonify({"error": "All features must be numeric"}), 400

        action = predict_action(model, stock_features)
        
        return jsonify({
            "action": action,
            "confidence": "high",  # Could be enhanced with actual confidence scoring
            "timestamp": pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Server error in predict endpoint: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/predict_stock/<ticker>", methods=["GET"])
def predict_stock_ticker(ticker):
    """Predict trading action for a specific stock ticker"""
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 503

        features = get_stock_features(ticker)
        if features is None:
            return jsonify({"error": f"Could not fetch data for ticker: {ticker}"}), 400

        action = predict_action(model, features)
        
        return jsonify({
            "ticker": ticker.upper(),
            "action": action,
            "features": features,
            "timestamp": pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Server error in predict_stock endpoint: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid import error if not needed
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting API server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
