# 🚀 Investigating the Effectiveness of Deep Reinforcement Learning (DRL) in Algorithmic Trading  
📌 **Hackathon: INGENIOUS 6.0**  
🔍 **Team Name: 404 Brain Not Found**  

---

## 📖 Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Solution Overview](#solution-overview)
- [Tech Stack](#tech-stack)
- [Market Data Sources](#market-data-sources)
- [Model Training & Implementation](#model-training--implementation)
- [Performance Evaluation](#performance-evaluation)
- [Challenges & Limitations](#challenges--limitations)
- [Future Work](#future-work)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📌 Introduction
Algorithmic trading has advanced with **Deep Reinforcement Learning (DRL)**, enabling adaptive trading strategies.  
This project **evaluates DRL effectiveness** by implementing a **Trading Deep Q-Network (TDQN)** for stock trading.  

### 🔥 Objective
- Implement a **TDQN-based trading model**.
- Train and test it using **real or simulated market data**.
- Evaluate the model’s **effectiveness, adaptability, and limitations**.
- Explore if DRL can create **robust trading strategies**.

---

## 🌟 Key Features
✔ **DRL-Based Trading Model** – Implements **TDQN** and tests adaptability.  
✔ **Real-Time Market Data** – Uses **Yahoo Finance API** for stock data.  
✔ **Performance Metrics** – Evaluates **Sharpe Ratio, ROI**, and risk-adjusted returns.  
✔ **Market Adaptability** – Analyzes trading behavior in dynamic market conditions.  
✔ **Flask-Based Trading Bot** – Deploys a web app for real-time trading insights.  
✔ **Data Visualization** – Uses **Matplotlib, Plotly, and Seaborn** for insights.  

---

## 🛠 Solution Overview
🔹 **Stock Market Adaptability**  
Tested **TDQN** on real stock data to analyze adaptability to market fluctuations.  

🔹 **Reinforcement Learning for Trading**  
- **Agent**: Learns from **historical/live data** to improve trading decisions.  
- **Environment**: Stock market fluctuations (OHLC, volume, economic factors).  
- **Actions**: **Buy**, **Sell**, or **Hold** based on market trends.  
- **Rewards**: **Profit/loss-based feedback** optimizes model learning.  

🔹 **Model Implementation**  
- Custom **Gym Environment** for training.  
- **Normalization & Feature Engineering**: RSI, MACD, SMA, MinMaxScaler.  
- **Performance Metrics**: Evaluated using **Sharpe Ratio** & **ROI**.  
- **Flask Deployment** for **real-time predictions**.

---

## 💻 Tech Stack

### **Programming & Libraries**
- Python  
- TensorFlow & PyTorch  
- Stable-Baselines3  
- Gymnasium (Custom Gym Environments)  

### **Data Processing & Feature Engineering**
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- Scikit-learn, Optuna  
- Log Transformation, MinMaxScaler, XGBoost  

### **Reinforcement Learning Algorithms**
- **Deep Q-Network (DQN)**  
- **Proximal Policy Optimization (PPO)**  

### **Visualization & Web Deployment**
- Flask  
- Plotly & Matplotlib  

---

## 📊 Market Data Sources
📡 **Yahoo Finance API** – Fetched **15+ years of stock data** for real-time adaptability.  

✅ **Dataset Includes:**  
- **OHLC (Open, High, Low, Close) Prices**  
- **Trading Volume**  
- **Technical Indicators (RSI, MACD, SMA/EMA)**  

---

## 🏗 Model Training & Implementation
- **Custom Gym Environment** – Trains TDQN in a simulated trading setup.  
- **Sharpe Ratio & ROI Optimization** – Enhances model performance.  
- **Action Decision Making** – **Buy**, **Sell**, or **Hold** based on stock signals.  
- **Flask Deployment** – Converts AI predictions into a **real-time trading dashboard**.  

📌 **Visualization of Training Progress**:  
![Training Graph](https://github.com/username/repository/blob/main/screenshots/training_graph.png)

---

## 📈 Performance Evaluation
The model was evaluated using **multiple trading strategies**:  
✔ **Sharpe B&H (Buy & Hold)** – Traditional long-term investment.  
✔ **Sharpe MR (Mean Reversion)** – Short-term price fluctuations.  
✔ **Sharpe TDQN** – DRL-based trading model with adaptive decision-making.  

🔹 **Performance Metrics Used**:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Huber Loss**
- **Profit/Loss Analysis**

---

## ⚠ Challenges & Limitations
🚨 **Market Volatility & Overfitting Risks**  
- Rapid fluctuations impact **generalization of the model**.  

🚨 **Reward Engineering Complexity**  
- Defining an **optimal reward function** remains a key challenge.  

🚨 **Execution Latency**  
- Real-time trading requires **fast decision-making**.  

🚨 **Data Quality & Feature Selection**  
- The model's accuracy depends on **clean, high-quality financial data**.  

🚨 **Hyperparameter Sensitivity**  
- Requires **fine-tuning** for optimal learning rates, discount factors, and exploration-exploitation balance.

---

## 🚀 Future Work
- **Enhanced Stability** – Using **PPO & A2C** for better convergence.  
- **Real-Time Trading Execution** – Faster decision-making integration.  
- **Multi-Agent Reinforcement Learning** – Smarter trading decisions.  
- **Sentiment Analysis** – Using news data to predict market trends.  
- **Real-Time Dashboards** – Live monitoring of stock performance.  

---

## 🛠 Setup & Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/username/repository.git
cd repository
pip install -r requirements.txt
python app.py
