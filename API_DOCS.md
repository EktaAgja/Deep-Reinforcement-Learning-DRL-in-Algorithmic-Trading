# 📡 Trading Prediction API Documentation

## Base URL
```
http://localhost:5001
```

## Authentication
Currently no authentication required. For production deployment, implement API key authentication.

## Endpoints

### 1. Health Check
Check if the API service is running and model is loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. Predict Trading Action (Raw Features)
Predict trading action from 9 stock features.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "features": [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1]
}
```

**Feature Description:**
1. `features[0]` - Closing Price
2. `features[1]` - Volume
3. `features[2]` - Opening Price
4. `features[3]` - High Price
5. `features[4]` - Low Price
6. `features[5]` - Price Change Ratio
7. `features[6]` - Price Difference (Close - Open)
8. `features[7]` - High-Low Difference
9. `features[8]` - Volume Ratio

**Response:**
```json
{
  "action": "Buy",
  "confidence": "high",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Possible Actions:**
- `"Buy"` - Recommended to buy the stock
- `"Hold"` - Recommended to hold the stock
- `"Sell"` - Recommended to sell the stock

### 3. Predict Trading Action (Stock Ticker)
Predict trading action for a specific stock ticker (fetches data automatically).

**Endpoint:** `GET /predict_stock/{ticker}`

**Parameters:**
- `ticker` (string) - Stock ticker symbol (e.g., AAPL, GOOGL, AMZN)

**Example:** `GET /predict_stock/AAPL`

**Response:**
```json
{
  "ticker": "AAPL",
  "action": "Buy",
  "features": [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid input. Expected 9 stock features as a list."
}
```

### 503 Service Unavailable
```json
{
  "error": "Model not available"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "details": "Specific error message"
}
```

## Usage Examples

### cURL Examples

#### Health Check
```bash
curl -X GET http://localhost:5001/health
```

#### Predict with Features
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1]
  }'
```

#### Predict for Stock Ticker
```bash
curl -X GET http://localhost:5001/predict_stock/AAPL
```

### Python Examples

#### Using requests library
```python
import requests
import json

# Health check
response = requests.get('http://localhost:5001/health')
print(response.json())

# Predict with features
features = [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1]
response = requests.post('http://localhost:5001/predict', 
                        json={'features': features})
print(response.json())

# Predict for stock ticker
response = requests.get('http://localhost:5001/predict_stock/AAPL')
print(response.json())
```

#### Using aiohttp (async)
```python
import aiohttp
import asyncio

async def predict_stock(ticker):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:5001/predict_stock/{ticker}') as response:
            return await response.json()

# Usage
result = asyncio.run(predict_stock('AAPL'))
print(result)
```

### JavaScript Examples

#### Using fetch API
```javascript
// Health check
fetch('http://localhost:5001/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Predict with features
const features = [150.5, 1000000, 149.8, 151.2, 149.0, 0.02, 0.7, 2.2, 1.1];
fetch('http://localhost:5001/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ features: features })
})
.then(response => response.json())
.then(data => console.log(data));

// Predict for stock ticker
fetch('http://localhost:5001/predict_stock/AAPL')
  .then(response => response.json())
  .then(data => console.log(data));
```

## Rate Limits
Currently no rate limits enforced. For production deployment, consider implementing:
- 100 requests per minute per IP
- 1000 requests per hour for authenticated users

## Data Privacy
- No personal data is stored or logged
- Stock ticker requests are logged for monitoring purposes
- Predictions are stateless and not stored

## Model Information
- **Algorithm:** Trading Deep Q-Network (TDQN)
- **Framework:** PyTorch
- **Input:** 9 stock features
- **Output:** 3 trading actions (Buy, Hold, Sell)
- **Update Frequency:** Model is static (retrain required for updates)

## Performance Notes
- Average response time: < 100ms
- Concurrent requests supported: 10-50 (depends on system resources)
- Model inference: CPU-based (GPU acceleration available)

## Troubleshooting

### Common Issues

1. **503 Service Unavailable**
   - Model file missing or corrupted
   - Check if `tdqn_trading_model.pth` exists
   - Restart the service

2. **400 Bad Request for ticker prediction**
   - Invalid ticker symbol
   - Network issues fetching stock data
   - Try again with valid ticker (e.g., AAPL, GOOGL)

3. **Timeout errors**
   - Network connectivity issues
   - High server load
   - Increase request timeout

### Debugging
Enable debug mode by setting environment variable:
```bash
export FLASK_DEBUG=true
python app.py
```

## Production Considerations

### Security
- Implement API key authentication
- Add rate limiting
- Use HTTPS in production
- Validate all inputs
- Implement CORS properly

### Performance
- Use production WSGI server (gunicorn)
- Implement caching for frequent requests
- Use load balancer for high availability
- Monitor resource usage

### Monitoring
- Implement logging and metrics
- Set up health checks
- Monitor prediction accuracy
- Track API usage patterns