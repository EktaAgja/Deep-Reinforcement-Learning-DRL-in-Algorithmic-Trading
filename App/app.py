from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Data loading with error handling
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "final_stock_data.csv"

def load_data():
    """Load stock data with proper error handling"""
    try:
        if not DATA_FILE.exists():
            logger.error(f"Data file not found: {DATA_FILE}")
            return None
        
        data = pd.read_csv(DATA_FILE)
        
        # Convert 'Date' column to datetime format with error handling
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        data = data.dropna(subset=['Date'])
        
        if data.empty:
            logger.error("No valid data after date parsing")
            return None
        
        data['Year'] = data['Date'].dt.year  # Extract Year for grouping
        logger.info(f"Data loaded successfully: {len(data)} records")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Load the CSV data
data = load_data()
unique_stocks = []

if data is not None:
    unique_stocks = sorted(data['Stock'].dropna().unique())
    logger.info(f"Unique stocks found: {len(unique_stocks)}")
else:
    logger.warning("No data available - app will run with limited functionality")

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

@app.route('/training_graphs')
def training_graphs():
    return render_template('training_graphs.html')

# Function to generate stock graph
def generate_stock_graph(stock):
    """Generate stock volume graph with error handling"""
    try:
        if data is None:
            return "<h3 style='color:red; text-align:center;'>Data not available.</h3>"
        
        filtered_data = data[data['Stock'] == stock]

        if filtered_data.empty:
            return "<h3 style='color:red; text-align:center;'>No data found for selected stock.</h3>"

        # Group by year and sum the volume
        volume_data = filtered_data.groupby('Year')['Volume'].sum().reset_index()

        # Create Plotly line chart
        fig = px.line(volume_data, x='Year', y='Volume', 
                      title=f"Yearly Trading Volume for {stock}",
                      labels={'Volume': 'Total Volume', 'Year': 'Year'},
                      markers=True)

        fig.update_traces(line=dict(color='blue'))  # Set line color
        fig.update_layout(template='plotly_white', hovermode='x')

        return pio.to_html(fig, full_html=False)
    
    except Exception as e:
        logger.error(f"Error generating stock graph: {e}")
        return f"<h3 style='color:red; text-align:center;'>Error generating graph: {e}</h3>"

def generate_stock_closing_graph(stock):
    """Generate stock closing price graph with error handling"""
    try:
        if data is None:
            return "<h3 style='color:red; text-align:center;'>Data not available.</h3>"
        
        filtered_data = data[data['Stock'] == stock]

        if filtered_data.empty:
            return "<h3 style='color:red; text-align:center;'>No data found for selected stock.</h3>"

        # Group by year and average the closing price (not sum)
        closing_data = filtered_data.groupby('Year')['Final_Close'].mean().reset_index()
        
        # Create Plotly line chart
        fig = px.line(closing_data, x='Year', y='Final_Close', 
                      title=f"Yearly Average Closing Price for {stock}",
                      labels={'Final_Close': 'Average Closing Price', 'Year': 'Year'},
                      markers=True)

        fig.update_traces(line=dict(color='green'))  # Set line color
        fig.update_layout(template='plotly_white', hovermode='x')

        return pio.to_html(fig, full_html=False)
    
    except Exception as e:
        logger.error(f"Error generating closing price graph: {e}")
        return f"<h3 style='color:red; text-align:center;'>Error generating graph: {e}</h3>"

@app.route('/graph_detail2', methods=['GET', 'POST'])
def graph_detail2():
    """Generate volume graph for selected stock"""
    try:
        if not unique_stocks:
            return render_template('error.html', 
                                 error="No stock data available")
        
        selected_stock = request.form.get("stock", unique_stocks[0])
        
        if selected_stock not in unique_stocks:
            selected_stock = unique_stocks[0]
            
        chart_html = generate_stock_graph(selected_stock)

        return render_template('graph_detail.html', 
                               stocks=unique_stocks, 
                               selected_stock=selected_stock, 
                               chart_html=chart_html)
    except Exception as e:
        logger.error(f"Error in graph_detail2: {e}")
        return render_template('error.html', error=str(e))

@app.route('/graph_detail1', methods=['GET', 'POST'])
def graph_detail1():
    """Generate closing price graph for selected stock"""
    try:
        if not unique_stocks:
            return render_template('error.html', 
                                 error="No stock data available")
        
        selected_stock = request.form.get("stock", unique_stocks[0])
        
        if selected_stock not in unique_stocks:
            selected_stock = unique_stocks[0]
            
        chart_html = generate_stock_closing_graph(selected_stock)

        return render_template('graph_detail.html', 
                               stocks=unique_stocks, 
                               selected_stock=selected_stock, 
                               chart_html=chart_html)
    except Exception as e:
        logger.error(f"Error in graph_detail1: {e}")
        return render_template('error.html', error=str(e))

def generate_roi_plot(stock_data):
    yearly_metrics = stock_data.groupby(["Stock", "Year"]).agg(
        Initial_Close=("Close", "first"),
        Final_Close=("Close", "last")
    )
    yearly_metrics["ROI"] = ((yearly_metrics["Final_Close"] - yearly_metrics["Initial_Close"]) / yearly_metrics["Initial_Close"]) * 100
    yearly_metrics.reset_index(inplace=True)

    plt.figure(figsize=(12, 6))
    plt.style.use("ggplot")

    stocks = yearly_metrics["Stock"].unique()
    years = np.sort(yearly_metrics["Year"].unique())
    width = 0.15
    x_positions = np.arange(len(years))
    colors = plt.cm.get_cmap("tab10", len(stocks))

    for idx, stock in enumerate(stocks):
        data = yearly_metrics[yearly_metrics["Stock"] == stock].set_index("Year").reindex(years).fillna(0).reset_index()
        plt.bar(x_positions + (idx * width), data["ROI"], width=width, label=stock, color=colors(idx))
    plt.xticks(x_positions + (width * len(stocks) / 2), years, rotation=45)
    plt.title("Yearly ROI by Stock")
    plt.ylabel("ROI (%)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(axis="y")
    plt.tight_layout()

    os.makedirs("static/plots", exist_ok=True)
    roi_plot_path = os.path.join("static", "plots", "roi_plot.png")
    plt.savefig(roi_plot_path)
    plt.close()
    return roi_plot_path

def generate_sharpe_ratio_plot(stock_data):
    stock_data["Daily_Return"] = stock_data.groupby("Stock")["Close"].pct_change()
    yearly_metrics = stock_data.groupby(["Stock", "Year"]).agg(
        Mean_Return=("Daily_Return", "mean"),
        Std_Return=("Daily_Return", "std")
    )
    yearly_metrics["Sharpe_Ratio"] = yearly_metrics["Mean_Return"] / yearly_metrics["Std_Return"]
    yearly_metrics.reset_index(inplace=True)

    plt.figure(figsize=(12, 6))
    plt.style.use("ggplot")

    os.makedirs("static/plots", exist_ok=True)
    sharpe_plot_path = os.path.join("static", "plots", "sharpe_ratio_plot.png")
    plt.savefig(sharpe_plot_path)
    plt.close()
    return sharpe_plot_path

@app.route('/gen_roi', methods=['GET', 'POST'])
def graph_detail3():
    stock_data = pd.read_csv("final_stock_data.csv", parse_dates=["Date"])
    stock_data["Year"] = stock_data["Date"].dt.year
    chart_html = generate_roi_plot(stock_data)  # Generate graph

    return render_template('graph_img_detail.html', 
                           chart_html=chart_html)

@app.route('/gen_shape', methods=['GET', 'POST'])
def graph_detail4():
    stock_data = pd.read_csv("final_stock_data.csv", parse_dates=["Date"])
    stock_data["Year"] = stock_data["Date"].dt.year
    chart_html = generate_sharpe_ratio_plot(stock_data)  # Generate graph

    return render_template('graph_img_detail.html', 
                           chart_html=chart_html)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting visualization server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
