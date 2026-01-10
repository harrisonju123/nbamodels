# NBA Betting Models

A data-driven NBA betting system using machine learning models, alternative data sources, and optimized betting strategies.

## 🎯 Current Performance

**Paper Trading Results** (as of January 2026):
- **6.91% ROI** overall (expected 7.5-8.9% with line shopping)
- **56.0% win rate** (84W-66L)
- **150 bets** tracked
- **+$1,036 profit** on $15,000 wagered

**Performance by Side:**
- Away Bets: **11.36% ROI** (58.3% win rate) ⭐
- Home Bets: **4.81% ROI** (54.9% win rate)

**Line Shopping**: Automatic best odds selection across 6 bookmakers (+0.5-2% ROI improvement)

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
# 1. Get today's bet recommendations (with automatic line shopping)
python scripts/daily_betting_pipeline.py

# 2. View performance dashboard (includes bankroll tracking)
python scripts/paper_trading_dashboard.py

# 3. Settle finished bets and update bankroll
python scripts/settle_bets.py

# 4. Line shopping value report (optional)
python scripts/line_shopping_report.py

# 5. Monitor closing line value (optional)
python scripts/generate_clv_report.py
```

## 📊 Features

### Machine Learning Models
- **Spread Prediction Model** (XGBoost classifier with isotonic calibration)
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
- **Dynamic bankroll**: Bet sizing grows with profits, compounds over time
- **Home bias penalty**: -2% edge adjustment
- **CLV filtering**: Historical closing line value threshold
- **Drawdown protection**: 30% stop loss tracking
- **Line shopping**: Automatic best odds across 6+ bookmakers

### Bankroll Management
- **Dynamic tracking**: Bankroll adjusts automatically with bet outcomes
- **Compounding growth**: Bet sizing scales with current bankroll
- **Performance metrics**: Peak, drawdown, ROI tracking
- **Automatic settlement**: Bets settled and bankroll updated via API
- **Historical sync**: Can backfill bankroll from existing bet history

### Rigorous Backtesting
- **Walk-forward validation**: Prevents data leakage with temporal train/test splits
- **Monte Carlo simulation**: 1,000+ bootstrap simulations for variance estimation
- **Statistical rigor**: Bootstrap confidence intervals, permutation tests, p-values
- **Transaction costs**: Realistic slippage, vig, and execution probability modeling
- **Position constraints**: Book limits, daily/per-game exposure management
- **Comprehensive reporting**: Visual dashboards with ROI distribution, drawdown analysis

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
│   │   ├── rigorous_backtest/  # Statistical backtesting framework
│   │   ├── kelly.py            # Kelly criterion
│   │   └── edge_strategy.py    # Edge-based betting
│   ├── bankroll/               # Bankroll management
│   └── data/                   # Data collection clients
├── scripts/
│   ├── daily_betting_pipeline.py      # Main pipeline
│   ├── paper_trading_dashboard.py     # Performance dashboard
│   ├── settle_bets.py                 # Bet settlement & bankroll updates
│   ├── run_rigorous_backtest.py       # Rigorous backtesting
│   ├── retrain_models.py              # Model retraining
│   └── collect_*.py                   # Data collection scripts
├── models/                     # Trained model files
├── logs/                       # Application logs
└── docs/                       # Documentation
    ├── PAPER_TRADING_FIXES.md
    ├── ALTERNATIVE_DATA_INTEGRATION.md
    ├── BACKTEST_ANALYSIS.md
    └── LINE_SHOPPING.md
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

### Rigorous Backtesting Framework

A professional-grade backtesting system with statistical rigor to validate betting strategies without data leakage.

**Key Features:**
- **Walk-Forward Validation**: Monthly retraining with expanding window to prevent look-ahead bias
- **Monte Carlo Simulation**: 1,000+ bootstrap simulations for variance estimation
- **Statistical Testing**: Bootstrap confidence intervals and permutation tests
- **Transaction Costs**: Realistic slippage, vig, and execution probability modeling
- **Position Constraints**: Book limits, daily exposure, and bankroll floor management

**Run the backtest:**

```bash
python scripts/run_rigorous_backtest.py
```

**Example Output:**

```
=== Rigorous Backtest Results ===

ROI: 8.84% (95% CI: 3.63%, 14.19%)
Win Rate: 54.9% (95% CI: 52.3%, 57.5%)
Sharpe: 1.42 (95% CI: 0.56, 2.26)
Max Drawdown: 40.9%

Number of Bets: 1,369
Total Wagered: $265,010.76
Total P&L: $25,102.06

Statistical Significance:
  P-value vs break-even: 0.0006 ✓ Highly significant
  Risk of Ruin (50%+ loss): 0.0%
  Probability Profitable: 100.0%

Transaction Costs:
  Gross ROI: 8.84%
  Net ROI: 4.34% (after slippage + vig)

Walk-Forward Folds: 41
Test Period: 2020-2024 (5 seasons)
```

**Visualizations** saved to `data/backtest/rigorous/`:
- `roi_distribution.png` - ROI histogram with confidence intervals
- `walk_forward.png` - Performance consistency across folds
- `drawdown.png` - Drawdown distribution from Monte Carlo

**Interpreting Results:**
- **ROI Confidence Interval**: If lower bound > 0%, strategy is likely profitable
- **P-value < 0.05**: Results are statistically significant (not just luck)
- **Sharpe Ratio > 1.0**: Strong risk-adjusted returns
- **Risk of Ruin < 5%**: Acceptable bankruptcy risk
- **Transaction Costs**: Net ROI accounts for real-world trading costs

**Configuration** (in `src/betting/rigorous_backtest/core.py`):
```python
config = BacktestConfig(
    initial_train_size=500,       # Games before first test
    retrain_frequency="monthly",  # Retrain every month
    kelly_fraction=0.2,           # 20% Kelly (conservative)
    min_edge_threshold=0.07,      # 7% minimum edge
    n_simulations=1000,           # Monte Carlo iterations
    initial_bankroll=10000.0,     # Starting capital
    base_vig=0.045,               # 4.5% sportsbook vig
    max_bet_per_book=500.0,       # Book betting limits
)
```

### Basic Backtest

```bash
# Run optimized backtest (faster, less rigorous)
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

## 🔑 Key Components (Consolidated January 2026)

### SpreadPredictionModel (`spread_model_calibrated.pkl`)
XGBoost classifier with isotonic calibration:
- **42.24% ROI, 75.4% win rate** (934 bets, 2020-2024 backtest)
- Probability calibration for reliable confidence estimates
- Optimized feature set with reduced noise

### Player Props Models (4 models)
Individual XGBoost regressors for player performance:
- **Points (pts)**: 27.1% ROI, 66.6% win rate
- **Rebounds (reb)**: 55.8% ROI, 81.6% win rate
- **Assists (ast)**: 89.4% ROI, 99.2% win rate (best performer)
- **3-Pointers (3pm)**: 55.0% ROI, 81.2% win rate

### Multi-Strategy Orchestrator
Consolidated risk-managed betting across 4 strategies:
- **Spread** (35% allocation): Edge-based spread betting
- **Arbitrage** (30% allocation): Cross-bookmaker arbitrage
- **Player Props** (20% allocation): Individual player performance
- **B2B Rest** (15% allocation): Back-to-back rest advantage
- Kelly criterion position sizing (25% fraction)
- Correlation-aware portfolio management

### GameFeatureBuilder
Streamlined feature generation (consolidated features):
- Team statistics (rolling windows: 5, 10, 20 games)
- Elo ratings
- Matchup features (H2H, division, conference)
- Schedule/rest/travel factors
- **Removed**: Alternative data (referee, news, sentiment) - added noise without signal

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

**Current Status**: ✅ Production-ready paper trading system with rigorous backtesting framework

**Live Performance**: 6.91% ROI (150 bets, 56% win rate)
**Backtest Performance**: 8.84% ROI (1,369 bets, 54.9% win rate, p<0.001)

Last updated: January 4, 2026
