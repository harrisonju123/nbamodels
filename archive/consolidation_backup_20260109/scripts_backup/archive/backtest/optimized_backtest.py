"""
Optimized Backtest - Using Improved Strategy

Compares baseline strategy vs optimized strategy:

Baseline:
- Min edge: 2%
- Kelly: 25%
- No home bias adjustment

Optimized:
- Min edge: 5% (7% for home)
- Kelly: 10%
- Home bias penalty: -2%
- Stricter disagreement: 20%
- Drawdown protection
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import timedelta
from loguru import logger

from src.models.dual_model import DualPredictionModel
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder
from src.betting.optimized_strategy import OptimizedBettingStrategy, OptimizedStrategyConfig

# Skip Four Factors (has data leakage)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

logger.info("=" * 70)
logger.info("OPTIMIZED BACKTEST - Baseline vs Optimized Strategy")
logger.info("=" * 70)

# Load data
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

# Build features once
logger.info("\nBuilding features for all games...")
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

# Filter to games with scores and spreads
features = features[
    features['point_diff'].notna() &
    features['spread_home'].notna()
].copy()

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
y_train = train_data['home_win']

X_test = test_data[feature_cols]
y_test = test_data['home_win']

model.fit(X_train, y_train)
logger.info("Model trained")

# Generate predictions
probs = model.predict_proba(X_test)
# DualModel returns dict with ensemble prob
if isinstance(probs, dict):
    test_data['model_prob'] = probs['ensemble']
elif len(probs.shape) == 1:
    test_data['model_prob'] = probs
else:
    test_data['model_prob'] = probs[:, 1]


def run_strategy(data: pd.DataFrame, strategy_name: str, config: OptimizedStrategyConfig):
    """Run backtest with given strategy configuration."""

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Running {strategy_name} Strategy")
    logger.info(f"{'=' * 70}")

    strategy = OptimizedBettingStrategy(config)

    # Simulate betting
    bankroll = 1000
    initial_bankroll = bankroll
    peak_bankroll = bankroll

    bets = []

    for idx, row in data.iterrows():
        # Calculate market probability from spread
        spread = row['spread_home']
        market_prob = 1 / (1 + np.exp(-spread / 4))  # Approximate conversion

        # Check both sides
        for side in ['home', 'away']:
            if side == 'home':
                model_prob = row['model_prob']
                side_name = 'HOME'
            else:
                model_prob = 1 - row['model_prob']
                market_prob = 1 - market_prob
                side_name = 'AWAY'

            edge = model_prob - market_prob

            # Get features for alt data filtering
            features_dict = {
                'home_news_volume_24h': row.get('home_news_volume_24h', 0),
                'away_news_volume_24h': row.get('away_news_volume_24h', 0),
                'news_volume_diff': row.get('news_volume_diff', 0),
                'sentiment_enabled': row.get('sentiment_enabled', False),
            }

            # Check if should bet
            should_bet, reason = strategy.should_bet(
                model_prob=model_prob,
                market_prob=market_prob,
                side=side,
                confidence=model_prob,
                features=features_dict
            )

            if should_bet:
                # Calculate bet size
                odds = 1.909  # -110 in decimal
                bet_size = strategy.calculate_bet_size(
                    edge=edge,
                    odds=odds,
                    bankroll=bankroll,
                    confidence=model_prob,
                    side=side
                )

                # Determine outcome
                if side == 'home':
                    won = row['point_diff'] > 0
                else:
                    won = row['point_diff'] < 0

                # Calculate profit
                if won:
                    profit = bet_size * (odds - 1)
                else:
                    profit = -bet_size

                bankroll += profit

                # Update strategy state
                strategy.update_bankroll(bankroll, won)

                # Track peak for drawdown
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll

                bets.append({
                    'date': row['date'],
                    'game_id': row.get('game_id'),
                    'home_team': row.get('home_team'),
                    'away_team': row.get('away_team'),
                    'side': side_name,
                    'edge': edge,
                    'model_prob': model_prob,
                    'market_prob': market_prob,
                    'bet_size': bet_size,
                    'won': won,
                    'profit': profit,
                    'bankroll': bankroll,
                    'reason': reason,
                })

    # Calculate metrics
    if not bets:
        logger.warning("No bets placed!")
        return None

    bets_df = pd.DataFrame(bets)

    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_wagered = bets_df['bet_size'].sum()
    total_profit = bets_df['profit'].sum()
    roi = total_profit / total_wagered if total_wagered > 0 else 0

    max_drawdown = (peak_bankroll - bets_df['bankroll'].min()) / peak_bankroll

    # Results
    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULTS - {strategy_name}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total bets: {total_bets}")
    logger.info(f"Wins: {wins}")
    logger.info(f"Losses: {losses}")
    logger.info(f"Win rate: {win_rate:.1%}")
    logger.info(f"")
    logger.info(f"Starting bankroll: ${initial_bankroll:.2f}")
    logger.info(f"Final bankroll: ${bankroll:.2f}")
    logger.info(f"Total P&L: ${total_profit:.2f}")
    logger.info(f"ROI: {roi:.1%}")
    logger.info(f"Max drawdown: {max_drawdown:.1%}")

    # By side
    home_bets = bets_df[bets_df['side'] == 'HOME']
    away_bets = bets_df[bets_df['side'] == 'AWAY']

    logger.info(f"\nBy Side:")
    if len(home_bets) > 0:
        home_roi = home_bets['profit'].sum() / home_bets['bet_size'].sum()
        logger.info(f"  HOME: {home_bets['won'].sum()}W-{len(home_bets) - home_bets['won'].sum()}L ({home_bets['won'].mean():.1%}) ROI: {home_roi:.1%}")
    if len(away_bets) > 0:
        away_roi = away_bets['profit'].sum() / away_bets['bet_size'].sum()
        logger.info(f"  AWAY: {away_bets['won'].sum()}W-{len(away_bets) - away_bets['won'].sum()}L ({away_bets['won'].mean():.1%}) ROI: {away_roi:.1%}")

    return {
        'strategy': strategy_name,
        'total_bets': total_bets,
        'win_rate': win_rate,
        'roi': roi,
        'final_bankroll': bankroll,
        'max_drawdown': max_drawdown,
        'bets_df': bets_df,
    }


# Run baseline strategy
baseline_config = OptimizedStrategyConfig(
    min_edge=0.02,  # 2%
    min_edge_home=0.02,  # Same as away
    kelly_fraction=0.25,  # 25%
    min_disagreement=0.15,  # 15%
    home_bias_penalty=0.0,  # No penalty
    prefer_away=False,
)

baseline_results = run_strategy(test_data, "BASELINE", baseline_config)

# Run optimized strategy
optimized_config = OptimizedStrategyConfig(
    min_edge=0.05,  # 5%
    min_edge_home=0.07,  # 7% for home
    kelly_fraction=0.10,  # 10%
    min_disagreement=0.20,  # 20%
    home_bias_penalty=0.02,  # -2% penalty
    prefer_away=True,
)

optimized_results = run_strategy(test_data, "OPTIMIZED", optimized_config)

# Comparison
if baseline_results and optimized_results:
    logger.info(f"\n{'=' * 70}")
    logger.info("STRATEGY COMPARISON")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Change'}")
    logger.info(f"{'-' * 70}")

    metrics = [
        ('Total Bets', 'total_bets', ''),
        ('Win Rate', 'win_rate', '%'),
        ('ROI', 'roi', '%'),
        ('Final Bankroll', 'final_bankroll', '$'),
        ('Max Drawdown', 'max_drawdown', '%'),
    ]

    for label, key, fmt in metrics:
        base_val = baseline_results[key]
        opt_val = optimized_results[key]

        if fmt == '%':
            base_str = f"{base_val:.1%}"
            opt_str = f"{opt_val:.1%}"
            change = opt_val - base_val
            change_str = f"{change:+.1%}"
        elif fmt == '$':
            base_str = f"${base_val:.2f}"
            opt_str = f"${opt_val:.2f}"
            change = opt_val - base_val
            change_str = f"${change:+.2f}"
        else:
            base_str = f"{base_val}"
            opt_str = f"{opt_val}"
            change = opt_val - base_val
            change_str = f"{change:+.0f}"

        logger.info(f"{label:<20} {base_str:<15} {opt_str:<15} {change_str}")

    logger.info(f"\n{'=' * 70}")

    if optimized_results['roi'] > baseline_results['roi']:
        logger.info("✓ OPTIMIZED STRATEGY PERFORMED BETTER")
    else:
        logger.info("⚠ Baseline still outperformed (may need further tuning)")
