# Deep Reinforcement Learning for Algorithmic Trading - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/EktaAgja/Deep-Reinforcement-Learning-DRL-in-Algorithmic-Trading.git
cd Deep-Reinforcement-Learning-DRL-in-Algorithmic-Trading
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Applications

#### 1. Data Visualization Dashboard
```bash
cd App
python app.py
```
Access at: http://localhost:5000

Features:
- Interactive stock data visualization
- ROI and Sharpe ratio analysis
- Multi-stock comparison charts

#### 2. Trading Prediction API
```bash
cd API-App
python app.py
```
Access at: http://localhost:5000

API Endpoints:
- `POST /predict` - Predict action from features
- `GET /predict_stock/<ticker>` - Predict action for stock ticker
- `GET /health` - Health check

## 📁 Project Structure

```
├── App/                    # Web dashboard application
│   ├── app.py             # Flask web application
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JS, images
├── API-App/              # Prediction API
│   ├── app.py            # Flask API application
│   └── tdqn_model_api.py # Model interface
├── data/                 # Stock market data
├── train_model.ipynb     # Model training notebook
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_DEBUG=false           # Set to true for development
export SECRET_KEY=your-secret-key  # For production
export PORT=5000                   # Server port
```

### Configuration Files
- `config.py` - Application configuration
- `environment.yml` - Conda environment (alternative to requirements.txt)

## 🧪 API Usage Examples

### Predict with Features
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1]
  }'
```

### Predict for Stock Ticker
```bash
curl http://localhost:5000/predict_stock/AAPL
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 📊 Model Information

### TDQN (Trading Deep Q-Network)
- **Input**: 9 stock features (price, volume, technical indicators)
- **Output**: 3 actions (Buy, Hold, Sell)
- **Architecture**: 3-layer neural network (128 hidden units)

### Features Used
1. Closing Price
2. Volume
3. Opening Price
4. High Price
5. Low Price
6. Price Change Ratio
7. Price Difference (Close - Open)
8. High-Low Difference
9. Volume Ratio

## 🛡️ Security Considerations

### Production Deployment
1. Set `FLASK_DEBUG=false`
2. Use strong `SECRET_KEY`
3. Implement rate limiting
4. Use HTTPS
5. Validate all inputs
6. Monitor logs

### Data Privacy
- Stock data is publicly available
- No personal information stored
- Model predictions are stateless

## 🧰 Development Tools

### Code Quality
```bash
# Install development tools
pip install black flake8 pytest

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

### Docker Support (Future Enhancement)
```dockerfile
# Dockerfile (to be created)
FROM python:3.8-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "App/app.py"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `tdqn_trading_model.pth` exists in the root directory
   - Check file permissions

2. **Data file missing**
   - Verify `data/final_stock_data.csv` exists
   - Run the data loading notebook if needed

3. **Import errors**
   - Check Python path configuration
   - Verify all dependencies are installed

4. **API not responding**
   - Check if the model loaded successfully
   - Review server logs for errors

### Logs
- Application logs are written to console
- Set logging level with `logging.basicConfig(level=logging.DEBUG)`

## 📈 Performance Optimization

### Model Performance
- Model uses CPU by default (good for most use cases)
- For GPU acceleration, ensure PyTorch CUDA version
- Consider model quantization for production

### Web Performance
- Use production WSGI server (gunicorn, uWSGI)
- Implement caching for frequent requests
- Use CDN for static assets

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- Follow PEP 8
- Use type hints where possible
- Add docstrings for functions
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♀️ Support

For issues and questions:
1. Check existing issues on GitHub
2. Create new issue with detailed description
3. Include error logs and system information

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Multiple RL algorithms comparison
- [ ] Portfolio optimization
- [ ] Backtesting framework
- [ ] Mobile application
- [ ] Advanced technical indicators
- [ ] Multi-timeframe analysis
- [ ] Risk management tools