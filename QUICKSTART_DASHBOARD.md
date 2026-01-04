# Dashboard Quick Start Guide

## 🚀 Launch the Dashboard

### Option 1: Streamlit UI (Recommended)
```bash
streamlit run analytics_dashboard.py
```
The dashboard will automatically open in your browser at **http://localhost:8501**

### Option 2: FastAPI Backend
```bash
uvicorn api.dashboard_api:app --reload --port 8000
```
- API available at: **http://localhost:8000**
- Interactive docs: **http://localhost:8000/docs**

### Option 3: Run Both (Full Stack)
```bash
# Terminal 1: Start API
uvicorn api.dashboard_api:app --reload --port 8000

# Terminal 2: Start Dashboard
streamlit run analytics_dashboard.py
```

---

## 📊 Dashboard Features

### Performance Overview
- **Win Rate** - Current success rate across all settled bets
- **Total Profit** - Cumulative profit/loss with bet count
- **ROI** - Return on investment percentage
- **Average CLV** - Closing line value (positive = beating the market)

### Analytics Tabs

#### 1. Performance Analytics
- Cumulative profit curve over time
- Rolling win rate trend
- Performance breakdown by edge buckets
- Date range and bet type filters

#### 2. CLV Analysis
- CLV distribution histogram
- CLV vs profit scatter plot
- CLV statistics by outcome
- Positive CLV rate tracking

#### 3. Model Performance
- Calibration plot (predicted vs actual)
- Model accuracy validation
- Probability bucket analysis

#### 4. Time-based Analytics
- Win rate heatmap by day/hour
- Recent bets table (last 20)
- Temporal pattern identification

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Clear Streamlit cache
streamlit cache clear

# Check if port is available
lsof -i :8501

# Run with debug logging
streamlit run analytics_dashboard.py --logger.level=debug
```

### API connection issues
```bash
# Check if port 8000 is available
lsof -i :8000

# Try different port
uvicorn api.dashboard_api:app --port 8001

# Test API health
curl http://localhost:8000/health
```

### No data showing
```bash
# Verify database exists
ls -lh data/bets/bets.db

# Check bet count
python -c "from src.bet_tracker import get_bet_history; print(len(get_bet_history()))"
```

---

## 📡 API Endpoints

### GET /api/summary
Overall betting statistics
```bash
curl http://localhost:8000/api/summary
curl http://localhost:8000/api/summary?days=30
```

### GET /api/performance
Detailed performance metrics (Sharpe ratio, max drawdown, etc.)
```bash
curl http://localhost:8000/api/performance
curl http://localhost:8000/api/performance?days=7
```

### GET /api/bets
All bets with filtering and pagination
```bash
curl http://localhost:8000/api/bets?limit=50&offset=0
curl http://localhost:8000/api/bets?bet_type=spread&outcome=win
```

### GET /api/clv
CLV analysis
```bash
curl http://localhost:8000/api/clv
curl http://localhost:8000/api/clv?days=30
```

### GET /api/live
Real-time current day statistics
```bash
curl http://localhost:8000/api/live
```

### GET /api/stats/daily
Daily aggregated statistics
```bash
curl http://localhost:8000/api/stats/daily?days=30
```

---

## 💡 Tips

1. **Auto-refresh**: Dashboard refreshes every 60 seconds automatically
2. **Export charts**: Click camera icon on charts to download as PNG
3. **Filter data**: Use sidebar date picker and bet type selector
4. **API docs**: Visit `/docs` for interactive Swagger documentation
5. **Mobile**: Dashboard is responsive and works on mobile devices

---

## 📖 Full Documentation

For detailed information, see **DASHBOARD_README.md**
