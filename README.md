# INGENIOUS 6.0 HACKATHON
## Investigating the Effectiveness of Deep Reinforcement Learning (DRL) in Algorithmic Trading
### Define : 
Algorithmic trading has evolved significantly with the advent of advanced machine learning techniques, particularly Deep Reinforcement Learning (DRL). Participants are tasked with exploring the effectiveness of DRL in developing adaptive trading strategies using the provided research paper on DRL-based algorithmic trading. While participants are not expected to design the core algorithm, they must: Implement the given DRL-based trading model. Test the model using real or simulated market data. Analyze its performance and derive insights into its effectiveness, adaptability, and limitations in dynamic market conditions. The goal is to demonstrate if and how DRL can be leveraged to create robust and adaptive trading strategies.

### Key Features :
- Model Implementation: Implement the provided DRL-based algorithmic trading model.
- Data Handling: Use real or simulated market data to test the model.
- Performance Evaluation: Analyze effectiveness using key financial metrics (e.g., Sharpe ratio, ROI).
- Market Adaptability: Assessment must be done on how the model performs under changing market conditions.
- Visualization & Insights: Present findings through graphs, performance comparisons, and risk analysis.
- Legal Considerations: Ensure the trading model follows ethical financial practices and regulations.
  
### Solution :
- We tested Trading Deep Q-Network (TDQN) on real stock 
data to see if AI can adapt to unpredictable markets. 
- By optimizing with Sharpe Ratio, ROI, and normalization, 
we trained TDQN in a custom Gym environment to make 
buy, sell, or hold decisions based on technical indicators. 
- The trained model was then deployed as a Flask-based 
trading bot and analysis report for real-time predictions. 
- While TDQN showed promise, future improvements 
include using PPO for stability, Multi-Agent RL for 
smarter trading, and real-time dashboards for live stock
tracking. This is just the beginning of AI-driven trading!

### Tech Stack :
#### Programming & Libraries:
-  Python
-  TensorFlow
-  PyTorch
-  Stable-Baselines3
-  Gymnasium
-  Custom Gym Environments
#### Data Processing & Feature Engineering: 
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly.express
- Scikit-learn
- Log Transformation
- MinMaxScaler
- XGBoost
- Regression
- Optuna
#### Reinforcement Learning Algorithms:
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)  
#### Charts & Visualization: 
- Flask
- Matplotlib
- Plotly.expres

 ### Market Data Sources :
We fetched live stock market data 
directly from Yahoo Finance for 
last 15 years, ensuring real-time 
adaptability. 
The dataset includes OHLC 
(Open, High, Low, Close) prices, 
volume, and technical 
indicators
 - Yahoo Finance API → Stock price history (OHLC, volume, adjusted close)



