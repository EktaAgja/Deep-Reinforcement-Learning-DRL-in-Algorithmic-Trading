from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64

app = Flask(__name__)

# Load the CSV data
file_path = "final_stock_data.csv"  # Update with your actual file path
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format with error handling
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
data = data.dropna(subset=['Date'])

data['Year'] = data['Date'].dt.year  # Extract Year for grouping

# Get unique stocks
unique_stocks = sorted(data['Stock'].dropna().unique())

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

def generate_stock_closing_graph(stock):
    filtered_data = data[data['Stock'] == stock]

    if filtered_data.empty:
        return "<h3 style='color:red; text-align:center;'>No data found for selected stock.</h3>"

    # Group by year and sum the volume
    volume_data = filtered_data.groupby('Year')['Final_Close'].sum().reset_index()
    
    # Create Plotly line chart
    fig = px.line(volume_data, x='Year', y='Final_Close', 
                  title=f"Yearly Trading Closing Price for {stock}",
                  labels={'Final_Close': 'Average Closing Price', 'Year': 'Year'},
                  markers=True)

    fig.update_traces(line=dict(color='blue'))  # Set line color
    fig.update_layout(template='plotly_white', hovermode='x')

    return pio.to_html(fig, full_html=False)

@app.route('/graph_detail2', methods=['GET', 'POST'])
def graph_detail2():
    selected_stock = request.form.get("stock", unique_stocks[0])  # Default to first stock
    chart_html = generate_stock_graph(selected_stock)  # Generate graph

    return render_template('graph_detail.html', 
                           stocks=unique_stocks, 
                           selected_stock=selected_stock, 
                           chart_html=chart_html)

@app.route('/graph_detail1', methods=['GET', 'POST'])
def graph_detail1():
    selected_stock = request.form.get("stock", unique_stocks[0])  # Default to first stock
    chart_html = generate_stock_closing_graph(selected_stock)  # Generate graph

    return render_template('graph_detail.html', 
                           stocks=unique_stocks, 
                           selected_stock=selected_stock, 
                           chart_html=chart_html)

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
    app.run(debug=True)
