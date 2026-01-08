# Advanced Analytics Dashboard

A comprehensive, real-time analytics dashboard for the NBA betting model with interactive visualizations, performance tracking, and CLV analysis.

## Features

### 📊 Analytics Dashboard (Streamlit)
- **Real-time Performance Tracking** - Live win rate, ROI, and profit metrics
- **Interactive Charts** - Profit curves, win rate trends, CLV distributions
- **Model Analytics** - Calibration plots, edge-based performance analysis
- **Time-based Insights** - Heatmaps showing performance by day/hour
- **CLV Analysis** - Comprehensive closing line value tracking
- **Recent Bets Table** - Latest betting activity with outcomes

### 🚀 REST API (FastAPI)
- **Real-time Data Endpoints** - JSON API for programmatic access
- **Performance Metrics** - Detailed statistics and analytics
- **Live Updates** - Current day performance and pending bets
- **Historical Data** - Daily aggregated statistics
- **Filtering & Pagination** - Flexible data queries

## Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install additional dependencies
pip install streamlit plotly fastapi uvicorn pydantic
```

### Quick Start

#### Option 1: Streamlit Dashboard (Recommended)
```bash
# Run the analytics dashboard
streamlit run analytics_dashboard.py

# Dashboard will open automatically in your browser at:
# http://localhost:8501
```

#### Option 2: FastAPI Backend
```bash
# Run the API server
uvicorn api.dashboard_api:app --reload --port 8000

# API will be available at:
# http://localhost:8000

# Interactive API docs at:
# http://localhost:8000/docs
```

#### Option 3: Both (Full Stack)
```bash
# Terminal 1: Start API
uvicorn api.dashboard_api:app --reload --port 8000

# Terminal 2: Start Dashboard
streamlit run analytics_dashboard.py
```

## Dashboard Features

### Performance Overview
- **Win Rate** - Overall and recent performance
- **Total Profit** - Cumulative profit/loss with bet count
- **ROI** - Return on investment percentage
- **Average CLV** - Closing line value tracking

### Performance Analytics Tab
- **Cumulative Profit Curve** - Visual profit progression over time
- **Win Rate Trend** - Rolling average win rate
- **Performance by Edge** - Win rate breakdown by model confidence
- **Filters** - Date range, bet type, time period selection

### CLV Analysis Tab
- **CLV Distribution** - Histogram showing closing line value spread
- **CLV vs Profit** - Scatter plot showing relationship
- **CLV Statistics** - Positive CLV rate, median, best/worst values
- **Outcome Correlation** - CLV performance by win/loss/push

### Model Performance Tab
- **Calibration Plot** - Model prediction accuracy visualization
- **Probability Analysis** - Predicted vs actual win rates
- **Model Quality Metrics** - Statistical validation

### Time-based Analytics Tab
- **Performance Heatmap** - Win rate by day of week and hour
- **Recent Bets Table** - Latest 20 bets with full details
- **Temporal Patterns** - Identify optimal betting times

## API Endpoints

### Summary Endpoints
- `GET /api/summary` - Overall betting summary statistics
- `GET /api/summary?days=30` - Summary for last 30 days

### Performance Endpoints
- `GET /api/performance` - Detailed performance metrics
- `GET /api/performance?days=7` - Weekly performance

### Bet Endpoints
- `GET /api/bets` - All bets with pagination
- `GET /api/bets?limit=50&offset=0` - Paginated results
- `GET /api/bets?bet_type=spread&outcome=win` - Filtered bets
- `GET /api/bets/recent?count=10` - Most recent bets

### CLV Endpoints
- `GET /api/clv` - Comprehensive CLV analysis
- `GET /api/clv?days=30` - Monthly CLV metrics

### Live Updates
- `GET /api/live` - Real-time current day statistics
- `GET /api/stats/daily?days=30` - Daily aggregated stats

### Health Check
- `GET /health` - API health status

## API Response Examples

### Summary Response
```json
{
  "total_bets": 150,
  "settled_bets": 145,
  "pending_bets": 5,
  "wins": 84,
  "losses": 58,
  "pushes": 3,
  "win_rate": 59.15,
  "total_profit": 1250.50,
  "total_wagered": 15000.00,
  "roi": 8.34,
  "avg_clv": 0.35
}
```

### Performance Metrics Response
```json
{
  "win_rate": 59.15,
  "roi": 8.34,
  "total_profit": 1250.50,
  "sharpe_ratio": 1.85,
  "max_drawdown": 450.00,
  "avg_bet_size": 103.45,
  "profit_factor": 1.42
}
```

### Live Update Response
```json
{
  "timestamp": "2026-01-04T18:30:00",
  "active_bets": 12,
  "todays_profit": 235.50,
  "todays_win_rate": 66.67,
  "pending_games": 5
}
```

## Configuration

### Dashboard Settings
Edit `analytics_dashboard.py` to customize:
- **Cache Duration** - Data refresh interval (default: 60 seconds)
- **Chart Colors** - Custom color schemes
- **Default Filters** - Initial date range and bet types
- **Display Options** - Metrics, columns, visualizations

### API Settings
Edit `api/dashboard_api.py` to customize:
- **CORS Origins** - Allowed frontend domains
- **Pagination Limits** - Default page sizes
- **Cache TTL** - API response caching
- **Port** - API server port (default: 8000)

## Dashboard Filters

### Time Period
- Last 7 Days
- Last 30 Days
- Last 90 Days
- All Time
- Custom Date Range

### Bet Type
- Spread
- Totals
- Moneyline

### Outcome (API only)
- Win
- Loss
- Push
- Pending

## Performance Metrics Explained

### Win Rate
- Formula: `Wins / (Wins + Losses) * 100`
- Excludes pushes from calculation
- Break-even for -110 odds: 52.38%

### ROI (Return on Investment)
- Formula: `Total Profit / Total Wagered * 100`
- Includes all bet outcomes (wins, losses, pushes)
- Positive ROI = profitable

### CLV (Closing Line Value)
- Measures if you beat the closing line
- Formula: `(Bet Odds - Closing Odds) / Closing Odds * 100`
- Positive CLV = good (beat the market)
- Negative CLV = poor (worse than closing)

### Sharpe Ratio
- Risk-adjusted return metric
- Higher is better (>1.0 is good, >2.0 is excellent)
- Annualized based on daily returns

### Profit Factor
- Formula: `Total Wins $ / Total Losses $`
- >1.0 = profitable
- >1.5 = strong performance

### Max Drawdown
- Largest peak-to-trough decline
- Measures worst-case loss scenario
- Lower is better

## Troubleshooting

### Dashboard Won't Start
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Clear Streamlit cache
streamlit cache clear

# Run with verbose logging
streamlit run analytics_dashboard.py --logger.level=debug
```

### API Connection Issues
```bash
# Check if port 8000 is available
lsof -i :8000

# Try different port
uvicorn api.dashboard_api:app --port 8001

# Check API health
curl http://localhost:8000/health
```

### Data Not Loading
```bash
# Verify database exists
ls -lh data/bets/bets.db

# Check bet data
python -c "from src.bet_tracker import get_all_bets; print(len(get_all_bets()))"

# Ensure proper permissions
chmod 644 data/bets/bets.db
```

### Performance Issues
- Reduce date range filter
- Limit number of bets displayed
- Clear browser cache
- Restart Streamlit server

## Production Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables
4. Deploy automatically

### Docker Deployment
```bash
# Build container
docker build -t nba-analytics-dashboard .

# Run dashboard
docker run -p 8501:8501 nba-analytics-dashboard

# Run API
docker run -p 8000:8000 nba-analytics-api
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run with nohup (background)
nohup streamlit run analytics_dashboard.py &

# Or use systemd service (Linux)
sudo systemctl start nba-dashboard
```

## Advanced Features

### Real-time Auto-Refresh
Dashboard automatically refreshes data every 60 seconds. Configure in sidebar:
- Refresh interval
- Manual refresh button
- Auto-scroll to recent bets

### Export Data
All charts support:
- Download as PNG
- Save interactive HTML
- Export to CSV (via API)

### Mobile Responsive
Dashboard is fully responsive and works on:
- Desktop browsers
- Tablets
- Mobile phones

### API Integration
Use the API to:
- Build custom dashboards
- Integrate with other tools
- Export to Excel/CSV
- Real-time monitoring scripts

## Data Privacy

- All data stored locally in SQLite database
- No external data transmission
- API runs on localhost by default
- Secure CORS configuration for production

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review API docs: `http://localhost:8000/docs`
3. Check Streamlit logs: `~/.streamlit/logs/`

## Future Enhancements

Planned features:
- [ ] WebSocket support for real-time updates
- [ ] Email/SMS alerts for bet outcomes
- [ ] Machine learning model comparison
- [ ] Advanced filtering (team, confidence level)
- [ ] Bankroll management visualizations
- [ ] Historical backtest comparisons
- [ ] Multi-user authentication
- [ ] Mobile app (React Native)

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API
- [Plotly](https://plotly.com/) - Interactive charts
- [Pandas](https://pandas.pydata.org/) - Data analysis

---

**Version**: 1.0.0
**Last Updated**: January 4, 2026
**Status**: ✅ Production Ready
