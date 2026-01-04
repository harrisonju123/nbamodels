# NBA Betting Models

A data-driven NBA betting system using machine learning models, alternative data sources, and optimized betting strategies.

## 🎯 Current Performance

**Paper Trading Results** (as of January 2026):
- **6.91% ROI** overall
- **56.0% win rate** (84W-66L)
- **150 bets** tracked
- **+$1,036 profit** on $15,000 wagered

**Performance by Side:**
- Away Bets: **11.36% ROI** (58.3% win rate) ⭐
- Home Bets: **4.81% ROI** (54.9% win rate)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd nbamodels

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Daily Workflow

```bash
# 1. Get today's bet recommendations
python scripts/daily_betting_pipeline.py

# 2. View performance dashboard
python scripts/paper_trading_dashboard.py

# 3. Monitor closing line value (optional)
python scripts/generate_clv_report.py
```

## 📊 Features

### Machine Learning Models
- **Dual Prediction Model** (MLP + XGBoost ensemble)
- **100 features** including:
  - Team rolling statistics (10-game windows)
  - Elo ratings
  - Rest/travel/schedule factors
  - Matchup-specific features
  - Four Factors (TS%, TOV%, OREB%, eFG%)
  - Alternative data (news, sentiment, referee stats)

### Alternative Data Sources
- **News Scraping**: RSS feeds from ESPN, NBA.com, Yahoo Sports
- **Sentiment Analysis**: Public sentiment from Reddit (r/nba)
- **Referee Data**: Historical referee tendencies and biases
- **Lineup Data**: Confirmed starter impact (ESPN API)

### Optimized Betting Strategy
- **Minimum edge**: 5% (7% for home bets)
- **Kelly fraction**: 10% for bankroll management
- **Home bias penalty**: -2% edge adjustment
- **CLV filtering**: Historical closing line value threshold
- **Drawdown protection**: 30% stop loss

## 📁 Project Structure

```
nbamodels/
├── data/
│   ├── raw/                    # Raw game and odds data
│   ├── bets/                   # Bet tracking database
│   └── cache/                  # Cached features and models
├── src/
│   ├── models/                 # ML model implementations
│   ├── features/               # Feature engineering
│   ├── betting/                # Betting strategies
│   └── data/                   # Data collection clients
├── scripts/
│   ├── daily_betting_pipeline.py      # Main pipeline
│   ├── paper_trading_dashboard.py     # Performance dashboard
│   ├── retrain_models.py              # Model retraining
│   └── collect_*.py                   # Data collection scripts
├── models/                     # Trained model files
├── logs/                       # Application logs
└── docs/                       # Documentation
    ├── PAPER_TRADING_FIXES.md
    ├── ALTERNATIVE_DATA_INTEGRATION.md
    └── BACKTEST_ANALYSIS.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
ODDS_API_KEY=your_odds_api_key_here

# Optional (for sentiment analysis)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
TWITTER_BEARER_TOKEN=your_twitter_token
```

### Strategy Options

The pipeline supports multiple strategies:

```bash
# CLV-filtered (default, most selective)
python scripts/daily_betting_pipeline.py --strategy clv_filtered

# Optimal timing (wait for best lines)
python scripts/daily_betting_pipeline.py --strategy optimal_timing

# Team-filtered (specific team biases)
python scripts/daily_betting_pipeline.py --strategy team_filtered

# Baseline (fewer filters)
python scripts/daily_betting_pipeline.py --strategy baseline
```

## 📈 Model Performance

### Training Metrics
- **Accuracy**: 74.78%
- **Features**: 100 (68 baseline + 32 advanced)
- **Training period**: 2020-2022
- **Test period**: 2022-2026

### Key Insights
- Away bets significantly outperform home bets (11.36% vs 4.81% ROI)
- Confirms market inefficiency due to home team overbetting
- Alternative data provides marginal improvement (forward testing ongoing)

## 🛠️ Development

### Training New Models

```bash
# Retrain all models
python scripts/retrain_models.py

# Run backtest
python scripts/optimized_backtest.py
```

### Collecting Data

```bash
# Collect alternative data
python scripts/setup_alternative_data.py

# Individual collectors (set up cron jobs)
python scripts/collect_news.py
python scripts/collect_sentiment.py
python scripts/collect_referees.py
```

### Recommended Cron Schedule

```bash
# Daily at 10 AM - collect referee assignments
0 10 * * * cd /path/to/nbamodels && python scripts/collect_referees.py >> logs/referees.log 2>&1

# Hourly - collect news
0 * * * * cd /path/to/nbamodels && python scripts/collect_news.py >> logs/news.log 2>&1

# Every 15 min during game hours (5-11 PM ET) - collect sentiment
*/15 17-23 * * * cd /path/to/nbamodels && python scripts/collect_sentiment.py >> logs/sentiment.log 2>&1
```

## 🧪 Testing

### Dry Run Mode

Preview recommendations without logging bets:

```bash
python scripts/daily_betting_pipeline.py --dry-run
```

### Backtest

```bash
# Run optimized backtest
python scripts/optimized_backtest.py

# Results saved to logs/optimized_backtest_final.log
```

## 📊 Dashboard

View comprehensive performance metrics:

```bash
python scripts/paper_trading_dashboard.py
```

**Dashboard includes:**
- Overall performance (ROI, win rate, profit)
- Performance by side (home/away breakdown)
- Last 15 days daily P&L
- Pending bets awaiting settlement

## 🎓 Documentation

Detailed documentation available in `docs/`:

- **[PAPER_TRADING_FIXES.md](docs/PAPER_TRADING_FIXES.md)** - Recent improvements to paper trading system
- **[ALTERNATIVE_DATA_INTEGRATION.md](docs/ALTERNATIVE_DATA_INTEGRATION.md)** - Guide to alternative data sources
- **[BACKTEST_ANALYSIS.md](docs/BACKTEST_ANALYSIS.md)** - Detailed backtest results and analysis
- **[INTEGRATION_COMPLETE.md](docs/INTEGRATION_COMPLETE.md)** - Integration milestone summary

## 🔑 Key Components

### DualPredictionModel
Ensemble of MLP and XGBoost with probability calibration:
- **MLP**: Captures non-linear patterns
- **XGBoost**: Handles feature interactions
- **Calibration**: Ensures probability reliability

### OptimizedBettingStrategy
Risk-managed betting with:
- Kelly criterion position sizing (10% fraction)
- Home bias mitigation (-2% penalty)
- CLV-based filtering
- Drawdown protection (30% stop)

### GameFeatureBuilder
Orchestrates feature generation:
- Team statistics (rolling windows)
- Elo ratings
- Matchup features
- Alternative data integration
- Schedule/rest/travel factors

## ⚠️ Disclaimer

**This system is for educational and research purposes only.**

- Past performance does not guarantee future results
- Sports betting involves risk of loss
- Only bet what you can afford to lose
- Check local gambling laws and regulations
- This is a paper trading system - not financial advice

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional alternative data sources
- [ ] Live betting strategies
- [ ] Totals (over/under) optimization
- [ ] Player props modeling
- [ ] Web-based dashboard
- [ ] Automated bet placement integrations

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- NBA Stats API
- The Odds API
- ESPN API
- Reddit API (for sentiment data)
- Open source sports betting community

---

**Current Status**: ✅ Production-ready paper trading system with 6.91% ROI

Last updated: January 4, 2026
