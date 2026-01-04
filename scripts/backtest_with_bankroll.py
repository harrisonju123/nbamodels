"""
Backtest with Dynamic Bankroll Management

Tests the complete system with compounding bankroll growth:
- Dynamic bet sizing based on current bankroll
- Proper Kelly criterion implementation
- Compounding growth simulation
- Comparison of static vs dynamic bankroll approaches
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Optional matplotlib for charts
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - charts will be skipped")

from src.models.dual_model import DualPredictionModel
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder
from src.betting.optimized_strategy import OptimizedBettingStrategy, OptimizedStrategyConfig

# Constants
STANDARD_SPREAD_ODDS_DECIMAL = 1.909  # -110 American odds in decimal
STANDARD_SPREAD_ODDS_AMERICAN = -110
PUSH_TOLERANCE = 0.1  # Tolerance for detecting pushes on spread results

# Skip Four Factors (has data leakage in historical data)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

logger.info("=" * 80)
logger.info("BACKTEST WITH DYNAMIC BANKROLL MANAGEMENT")
logger.info("=" * 80)

# Load data
logger.info("Loading historical data...")
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')

TEAM_MAP = {
    'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
    'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'nop': 'NOP',
    'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI',
    'phx': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
    'utah': 'UTA', 'uta': 'UTA', 'wsh': 'WAS', 'was': 'WAS', 'nj': 'BKN',
    'sea': 'OKC', 'njn': 'BKN', 'noh': 'NOP',
}

odds['home_team'] = odds['home'].map(TEAM_MAP)
odds['away_team'] = odds['away'].map(TEAM_MAP)
odds['date'] = pd.to_datetime(odds['date'])

# Merge odds
games['date'] = pd.to_datetime(games['date'])
games_with_odds = games.merge(
    odds[['date', 'home_team', 'away_team', 'spread', 'total']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).rename(columns={'spread': 'spread_home'})

logger.info(f"Total games: {len(games)}")
logger.info(f"Games with spreads: {games_with_odds['spread_home'].notna().sum()}")

# Build features
logger.info("\nBuilding features...")
builder = GameFeatureBuilder()
features = builder.build_game_features(games_with_odds.copy())

# Preserve odds columns
features['spread_home'] = games_with_odds['spread_home']
features['total'] = games_with_odds['total']

# Get feature columns
feature_cols = builder.get_feature_columns(features)
logger.info(f"Feature columns: {len(feature_cols)}")

# Add targets
features['point_diff'] = games_with_odds['home_score'] - games_with_odds['away_score']
features['home_win'] = (features['point_diff'] > 0).astype(int)

# Validate spread_home column exists
if 'spread_home' not in features.columns:
    logger.error("Missing 'spread_home' column required for spread coverage calculation")
    raise ValueError("Missing 'spread_home' column in features DataFrame")

# Check for missing spread data
missing_spreads = features['spread_home'].isna().sum()
if missing_spreads > 0:
    logger.warning(f"{missing_spreads} games missing spread data - will be filtered out")

# CRITICAL FIX: Train on spread coverage, not game wins!
# Home covers spread if: point_diff + spread_home > 0
features['home_covers'] = (features['point_diff'] + features['spread_home'] > 0).astype(int)

# Filter to games with scores and spreads
features = features[
    features['point_diff'].notna() &
    features['spread_home'].notna()
].copy()

# Validate we have data after filtering
if features.empty:
    logger.error("No games remaining after filtering for scores and spreads")
    raise ValueError("No valid game data available for backtest")

# Train/test split
train_end = pd.to_datetime('2022-09-30')
test_start = pd.to_datetime('2022-10-01')

train_data = features[features['date'] <= train_end].copy()
test_data = features[features['date'] >= test_start].copy()

logger.info(f"\nTrain period: {train_data['date'].min()} to {train_data['date'].max()}")
logger.info(f"Test period: {test_data['date'].min()} to {test_data['date'].max()}")
logger.info(f"Train games: {len(train_data)}")
logger.info(f"Test games: {len(test_data)}")

# Train model
logger.info("\nTraining model...")
model = DualPredictionModel()

X_train = train_data[feature_cols]
y_train = train_data['home_covers']  # ← FIXED: Train on spread coverage!

X_test = test_data[feature_cols]
y_test = test_data['home_covers']  # ← FIXED

model.fit(X_train, y_train)
logger.info("✓ Model trained")

# Generate predictions
probs = model.predict_proba(X_test)
if isinstance(probs, dict):
    test_data['model_prob'] = probs['ensemble']
elif len(probs.shape) == 1:
    test_data['model_prob'] = probs
else:
    test_data['model_prob'] = probs[:, 1]


def run_backtest(data: pd.DataFrame, strategy_name: str, use_dynamic_bankroll: bool = True):
    """
    Run backtest with optional dynamic bankroll.

    Args:
        data: Test data with predictions
        strategy_name: Name for logging
        use_dynamic_bankroll: If True, bet sizes grow with bankroll. If False, static $1000.

    Returns:
        Dict with results and bet history
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running: {strategy_name}")
    logger.info(f"Bankroll Mode: {'DYNAMIC (Compounding)' if use_dynamic_bankroll else 'STATIC ($1000)'}")
    logger.info(f"{'=' * 80}")

    # Validate input data
    if data.empty:
        logger.error("No test data provided - cannot run backtest")
        return {
            'strategy': strategy_name,
            'use_dynamic_bankroll': use_dynamic_bankroll,
            'error': 'No test data',
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'roi': 0.0,
            'bets_df': pd.DataFrame(),
        }

    # Validate required columns
    required_cols = ['date', 'model_prob', 'spread_home', 'point_diff', 'home_covers']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return {
            'strategy': strategy_name,
            'use_dynamic_bankroll': use_dynamic_bankroll,
            'error': f'Missing columns: {missing_cols}',
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'roi': 0.0,
            'bets_df': pd.DataFrame(),
        }

    # Strategy configuration (current production settings)
    config = OptimizedStrategyConfig(
        min_edge=0.05,  # 5%
        min_edge_home=0.07,  # 7% for home
        kelly_fraction=0.10,  # 10% Kelly
        min_disagreement=0.20,  # 20%
        home_bias_penalty=0.02,  # -2% penalty
        prefer_away=True,
        max_bet_size=0.05,  # 5% of bankroll
        min_bet_size=0.01,  # 1% of bankroll
        max_drawdown_stop=0.30,  # 30% stop
    )

    strategy = OptimizedBettingStrategy(config)

    # Initialize bankroll
    initial_bankroll = 1000.0
    current_bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    static_cumulative_profit = 0.0  # For static mode performance optimization

    bets = []
    bankroll_history = [{'date': data['date'].min(), 'bankroll': initial_bankroll}]

    for idx, row in data.iterrows():
        # Get spread for this game
        spread = row['spread_home']

        # Market probability for spread coverage
        # CRITICAL: Spreads are set by bookmakers to create a 50/50 market (balanced action).
        # After removing vig (~4.5% from -110 odds on both sides), true probability
        # should be ~50% for each side. This is the fundamental assumption of spread betting.
        # The spread VALUE adjusts to make both sides equally attractive to bettors.
        market_prob_spread_coverage = 0.50  # Both home and away sides are 50% probability

        # Check both sides
        for side in ['home', 'away']:
            if side == 'home':
                model_prob = row['model_prob']  # Prob home covers spread
                side_name = 'HOME'
            else:
                model_prob = 1 - row['model_prob']  # Prob away covers spread
                side_name = 'AWAY'

            edge = model_prob - market_prob_spread_coverage

            # Get features for filtering
            features_dict = {
                'home_news_volume_24h': row.get('home_news_volume_24h', 0),
                'away_news_volume_24h': row.get('away_news_volume_24h', 0),
                'news_volume_diff': row.get('news_volume_diff', 0),
                'sentiment_enabled': row.get('sentiment_enabled', False),
            }

            # Check if should bet
            should_bet, reason = strategy.should_bet(
                model_prob=model_prob,
                market_prob=market_prob_spread_coverage,  # Same for both sides
                side=side,
                confidence=model_prob,
                features=features_dict
            )

            if should_bet:
                # Calculate bet size
                odds = STANDARD_SPREAD_ODDS_DECIMAL

                # Use current bankroll for dynamic, or fixed $1000 for static
                bankroll_for_sizing = current_bankroll if use_dynamic_bankroll else 1000.0

                bet_size = strategy.calculate_bet_size(
                    edge=edge,
                    odds=odds,
                    bankroll=bankroll_for_sizing,
                    confidence=model_prob,
                    side=side
                )

                # Determine outcome
                actual_diff = row['point_diff']  # home_score - away_score

                # Spread coverage logic:
                # spread is from home team perspective
                # Home covers if: actual_diff + spread > 0
                # Away covers if: actual_diff + spread < 0
                # Push if: actual_diff + spread == 0 (exactly)

                spread_result = actual_diff + spread

                # Check for push first (within 0.1 point tolerance)
                if abs(spread_result) < PUSH_TOLERANCE:
                    profit = 0.0
                    outcome = 'push'
                    won = False
                elif side == 'home':
                    # Home covers if spread_result > 0
                    won = spread_result > 0
                    if won:
                        profit = bet_size * (odds - 1)
                        outcome = 'win'
                    else:
                        profit = -bet_size
                        outcome = 'loss'
                else:
                    # Away covers if spread_result < 0
                    won = spread_result < 0
                    if won:
                        profit = bet_size * (odds - 1)
                        outcome = 'win'
                    else:
                        profit = -bet_size
                        outcome = 'loss'

                # Update bankroll
                if use_dynamic_bankroll:
                    current_bankroll += profit
                    bankroll_value = current_bankroll
                else:
                    static_cumulative_profit += profit
                    bankroll_value = initial_bankroll + static_cumulative_profit

                # Update strategy state
                strategy.update_bankroll(current_bankroll if use_dynamic_bankroll else 1000.0, won)

                # Track peak for drawdown
                if use_dynamic_bankroll and current_bankroll > peak_bankroll:
                    peak_bankroll = current_bankroll
                elif not use_dynamic_bankroll and bankroll_value > peak_bankroll:
                    peak_bankroll = bankroll_value

                bets.append({
                    'date': row['date'],
                    'game_id': row.get('game_id'),
                    'home_team': row.get('home_team'),
                    'away_team': row.get('away_team'),
                    'side': side_name,
                    'edge': edge,
                    'model_prob': model_prob,
                    'market_prob': market_prob_spread_coverage,
                    'bet_size': bet_size,
                    'outcome': outcome,
                    'won': won,
                    'profit': profit,
                    'bankroll': bankroll_value,
                    'reason': reason,
                })

                bankroll_history.append({
                    'date': row['date'],
                    'bankroll': bankroll_value
                })

    # Calculate metrics
    if not bets:
        logger.warning("No bets placed - strategy filtered out all opportunities")
        return {
            'strategy': strategy_name,
            'use_dynamic_bankroll': use_dynamic_bankroll,
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'roi': 0.0,
            'warning': 'No bets passed strategy filters',
            'bets_df': pd.DataFrame(),
        }

    bets_df = pd.DataFrame(bets)

    total_bets = len(bets_df)
    wins = (bets_df['outcome'] == 'win').sum()
    losses = (bets_df['outcome'] == 'loss').sum()
    pushes = (bets_df['outcome'] == 'push').sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    total_wagered = bets_df['bet_size'].sum()
    total_profit = bets_df['profit'].sum()

    # Validate ROI calculation
    if total_wagered == 0 and total_profit != 0:
        logger.error(f"Data inconsistency: profit={total_profit:.2f} but wagered=0")
        roi = 0.0
    elif total_wagered > 0:
        roi = total_profit / total_wagered
    else:
        roi = 0.0

    final_bankroll = current_bankroll if use_dynamic_bankroll else initial_bankroll + total_profit
    max_drawdown = (peak_bankroll - bets_df['bankroll'].min()) / peak_bankroll

    # Print results
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RESULTS - {strategy_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total bets: {total_bets}")
    logger.info(f"Wins: {wins} | Losses: {losses} | Pushes: {pushes}")
    logger.info(f"Win rate: {win_rate:.1%}")
    logger.info(f"")
    logger.info(f"Starting bankroll: ${initial_bankroll:,.2f}")
    logger.info(f"Final bankroll: ${final_bankroll:,.2f}")
    logger.info(f"Total P&L: ${total_profit:+,.2f}")
    logger.info(f"Bankroll ROI: {((final_bankroll - initial_bankroll) / initial_bankroll):.1%}")
    logger.info(f"Wagered ROI: {roi:.1%}")
    logger.info(f"Peak bankroll: ${peak_bankroll:,.2f}")
    logger.info(f"Max drawdown: {max_drawdown:.1%}")
    logger.info(f"")
    logger.info(f"Total wagered: ${total_wagered:,.2f}")
    logger.info(f"Avg bet size: ${bets_df['bet_size'].mean():.2f}")
    logger.info(f"Max bet size: ${bets_df['bet_size'].max():.2f}")
    logger.info(f"Min bet size: ${bets_df['bet_size'].min():.2f}")

    # By side
    home_bets = bets_df[bets_df['side'] == 'HOME']
    away_bets = bets_df[bets_df['side'] == 'AWAY']

    logger.info(f"\nPerformance by Side:")
    if len(home_bets) > 0:
        home_wins = (home_bets['outcome'] == 'win').sum()
        home_losses = (home_bets['outcome'] == 'loss').sum()
        home_wr = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0
        home_roi = home_bets['profit'].sum() / home_bets['bet_size'].sum()
        logger.info(f"  HOME: {home_wins}W-{home_losses}L ({home_wr:.1%}) | ROI: {home_roi:.1%} | {len(home_bets)} bets")

    if len(away_bets) > 0:
        away_wins = (away_bets['outcome'] == 'win').sum()
        away_losses = (away_bets['outcome'] == 'loss').sum()
        away_wr = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0
        away_roi = away_bets['profit'].sum() / away_bets['bet_size'].sum()
        logger.info(f"  AWAY: {away_wins}W-{away_losses}L ({away_wr:.1%}) | ROI: {away_roi:.1%} | {len(away_bets)} bets")

    return {
        'strategy': strategy_name,
        'use_dynamic_bankroll': use_dynamic_bankroll,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'roi': roi,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': final_bankroll,
        'total_profit': total_profit,
        'max_drawdown': max_drawdown,
        'peak_bankroll': peak_bankroll,
        'bets_df': bets_df,
        'bankroll_history': pd.DataFrame(bankroll_history),
    }


# Run both backtests
logger.info("\n" + "=" * 80)
logger.info("RUNNING BACKTESTS")
logger.info("=" * 80)

static_results = run_backtest(test_data.copy(), "Static Bankroll ($1000)", use_dynamic_bankroll=False)
dynamic_results = run_backtest(test_data.copy(), "Dynamic Bankroll (Compounding)", use_dynamic_bankroll=True)

# Comparison
if static_results and dynamic_results:
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Static vs Dynamic Bankroll")
    logger.info("=" * 80)

    logger.info(f"\n{'Metric':<30} {'Static':<20} {'Dynamic':<20} {'Difference':<20}")
    logger.info("-" * 90)

    # Total bets
    logger.info(f"{'Total Bets':<30} {static_results['total_bets']:<20} {dynamic_results['total_bets']:<20} {dynamic_results['total_bets'] - static_results['total_bets']:<20}")

    # Win rate
    logger.info(f"{'Win Rate':<30} {static_results['win_rate']:<20.1%} {dynamic_results['win_rate']:<20.1%} {(dynamic_results['win_rate'] - static_results['win_rate']):<20.1%}")

    # Final bankroll
    logger.info(f"{'Final Bankroll':<30} ${static_results['final_bankroll']:<19.2f} ${dynamic_results['final_bankroll']:<19.2f} ${(dynamic_results['final_bankroll'] - static_results['final_bankroll']):<19.2f}")

    # Total profit
    logger.info(f"{'Total Profit':<30} ${static_results['total_profit']:<19,.2f} ${dynamic_results['total_profit']:<19,.2f} ${(dynamic_results['total_profit'] - static_results['total_profit']):<19,.2f}")

    # ROI
    logger.info(f"{'ROI':<30} {static_results['roi']:<20.1%} {dynamic_results['roi']:<20.1%} {(dynamic_results['roi'] - static_results['roi']):<20.1%}")

    # Max drawdown
    logger.info(f"{'Max Drawdown':<30} {static_results['max_drawdown']:<20.1%} {dynamic_results['max_drawdown']:<20.1%} {(dynamic_results['max_drawdown'] - static_results['max_drawdown']):<20.1%}")

    # Compounding benefit
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPOUNDING BENEFIT")
    logger.info(f"{'=' * 80}")

    profit_diff = dynamic_results['total_profit'] - static_results['total_profit']
    pct_improvement = (profit_diff / abs(static_results['total_profit'])) * 100 if static_results['total_profit'] != 0 else 0

    logger.info(f"Additional profit from compounding: ${profit_diff:+,.2f} ({pct_improvement:+.1f}%)")

    if dynamic_results['final_bankroll'] > static_results['final_bankroll']:
        logger.info(f"✓ Dynamic bankroll outperformed static by ${dynamic_results['final_bankroll'] - static_results['final_bankroll']:,.2f}")
    else:
        logger.info(f"⚠ Static bankroll outperformed dynamic (unusual - may indicate high drawdown)")

    # Create chart (if matplotlib available)
    if HAS_MATPLOTLIB:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Generating bankroll progression chart...")

        plt.figure(figsize=(14, 8))

        # Plot 1: Bankroll over time
        plt.subplot(2, 1, 1)
        static_history = static_results['bankroll_history']
        dynamic_history = dynamic_results['bankroll_history']

        plt.plot(static_history['date'], static_history['bankroll'], label='Static Bankroll', linewidth=2)
        plt.plot(dynamic_history['date'], dynamic_history['bankroll'], label='Dynamic Bankroll (Compounding)', linewidth=2)
        plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Starting Bankroll')
        plt.xlabel('Date')
        plt.ylabel('Bankroll ($)')
        plt.title('Bankroll Progression: Static vs Dynamic')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Cumulative profit
        plt.subplot(2, 1, 2)
        static_bets = static_results['bets_df'].sort_values('date')
        dynamic_bets = dynamic_results['bets_df'].sort_values('date')

        static_bets['cumulative_profit'] = static_bets['profit'].cumsum()
        dynamic_bets['cumulative_profit'] = dynamic_bets['profit'].cumsum()

        plt.plot(static_bets['date'], static_bets['cumulative_profit'], label='Static Bankroll', linewidth=2)
        plt.plot(dynamic_bets['date'], dynamic_bets['cumulative_profit'], label='Dynamic Bankroll', linewidth=2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit ($)')
        plt.title('Cumulative Profit Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('logs/bankroll_backtest_comparison.png', dpi=150, bbox_inches='tight')
        logger.info(f"✓ Chart saved to logs/bankroll_backtest_comparison.png")
    else:
        logger.info(f"\n⚠ Skipping chart generation (matplotlib not installed)")

logger.info(f"\n{'=' * 80}")
logger.info(f"✅ BACKTEST COMPLETE")
logger.info(f"{'=' * 80}")
