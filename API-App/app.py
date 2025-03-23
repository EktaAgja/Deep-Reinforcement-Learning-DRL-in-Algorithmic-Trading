from flask import Flask, request, render_template, jsonify
import numpy as np
import yfinance as yf  # To fetch stock data
from tdqn_model import load_model, predict_action

app = Flask(__name__)

# Load trained model
model = load_model()

# Home route - UI Page
@app.route("/")
def home():
    return render_template("index.html")

def get_stock_features(ticker):
    try:
        print(f"Fetching stock data for: {ticker}")  

        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        print(f"Stock data retrieved: {hist}")  

        if hist.empty or len(hist) < 2:  
            print("Stock data is empty or not enough data!")
            return None  

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]

        features = [
            float(latest["Close"]), float(latest["Volume"]), float(latest["Open"]),
            float(latest["High"]), float(latest["Low"]),
            (latest["Close"] - prev["Close"]) / prev["Close"] if prev["Close"] != 0 else 0,
            latest["Close"] - latest["Open"], latest["High"] - latest["Low"],
            latest["Volume"] / prev["Volume"] if prev["Volume"] != 0 else 1
        ]

        print(f"Extracted Features: {features}")
        return features  

    except Exception as e:
        print(f"Error fetching stock data: {e}")  
        return None  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  
        stock_features = data.get("features")

        if not stock_features or len(stock_features) != 9:
            return jsonify({"error": "Invalid input. Expected 9 stock features."}), 400

        action = predict_action(model, stock_features)
        return jsonify({"action": action})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
