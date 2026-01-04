"""
Diagnose Away Bet Performance Issue

Tests multiple hypotheses to identify why away bets have 35.7% win rate:
1. prefer_away setting causing bad selection
2. home_bias_penalty applied incorrectly
3. Edge calculation inverted for away bets
4. Home bets only baseline

Date: January 4, 2026
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.models.dual_model import DualPredictionModel
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder
from src.betting.optimized_strategy import OptimizedBettingStrategy, OptimizedStrategyConfig

# Disable Four Factors
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

logger.info("=" * 80)
logger.info("DIAGNOSING AWAY BET ISSUE")
logger.info("=" * 80)

# Load and prepare data (same as backtest)
logger.info("\nLoading data...")
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')

TEAM_MAP = {
    'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
    'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'nop': 'NOP',
    'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI',
    'phx': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
    'uta': 'UTA', 'was': 'WAS',
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

# Preserve odds
features['spread_home'] = games_with_odds['spread_home']
features['total'] = games_with_odds['total']

# Get feature columns
feature_cols = builder.get_feature_columns(features)

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

train_df = features[features['date'] <= train_end].copy()
test_df = features[features['date'] >= test_start].copy()

logger.info(f"\nTrain period: {train_df['date'].min()} to {train_df['date'].max()}")
logger.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
logger.info(f"Train games: {len(train_df)}")
logger.info(f"Test games: {len(test_df)}")

# Train model
logger.info("\nTraining model...")
model = DualPredictionModel()

X_train = train_df[feature_cols]
y_train = train_df['home_win']

model.fit(X_train, y_train)
logger.info("✓ Model trained")

# Generate predictions
X_test = test_df[feature_cols]
probs = model.predict_proba(X_test)
if isinstance(probs, dict):
    test_df['home_win_prob'] = probs['ensemble']
else:
    test_df['home_win_prob'] = probs
test_df['away_win_prob'] = 1 - test_df['home_win_prob']

# Calculate market probabilities from spread
test_df['market_home_prob'] = 0.5 + (test_df['spread_home'] * 0.025)
test_df['market_home_prob'] = test_df['market_home_prob'].clip(0.1, 0.9)
test_df['market_away_prob'] = 1 - test_df['market_home_prob']

logger.info(f"\nTest set ready: {len(test_df)} games")


def run_strategy_test(name, config, test_data):
    """Run backtest with specific strategy configuration."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TEST: {name}")
    logger.info(f"{'=' * 80}")

    strategy = OptimizedBettingStrategy(config)

    home_bets = []
    away_bets = []

    for idx, row in test_data.iterrows():
        if pd.isna(row['spread_home']):
            continue

        # Test home bet
        should_bet_home, reason = strategy.should_bet(
            model_prob=row['home_win_prob'],
            market_prob=row['market_home_prob'],
            side='home'
        )

        if should_bet_home:
            spread_result = row['point_diff'] + row['spread_home']
            won = spread_result > 0

            home_bets.append({
                'won': won,
                'model_prob': row['home_win_prob'],
                'market_prob': row['market_home_prob'],
                'edge': row['home_win_prob'] - row['market_home_prob'],
                'spread': row['spread_home'],
                'spread_result': spread_result
            })

        # Test away bet
        should_bet_away, reason = strategy.should_bet(
            model_prob=row['away_win_prob'],
            market_prob=row['market_away_prob'],
            side='away'
        )

        if should_bet_away:
            spread_result = row['point_diff'] + row['spread_home']
            won = spread_result < 0

            away_bets.append({
                'won': won,
                'model_prob': row['away_win_prob'],
                'market_prob': row['market_away_prob'],
                'edge': row['away_win_prob'] - row['market_away_prob'],
                'spread': row['spread_home'],
                'spread_result': spread_result
            })

    # Report results
    total_bets = len(home_bets) + len(away_bets)

    if len(home_bets) > 0:
        home_wins = sum(b['won'] for b in home_bets)
        home_wr = home_wins / len(home_bets)
        home_avg_edge = np.mean([b['edge'] for b in home_bets])
        logger.info(f"HOME: {home_wins}/{len(home_bets)} ({home_wr:.1%} WR) | Avg Edge: {home_avg_edge:+.3f}")
    else:
        logger.info("HOME: No bets")
        home_wr = 0

    if len(away_bets) > 0:
        away_wins = sum(b['won'] for b in away_bets)
        away_wr = away_wins / len(away_bets)
        away_avg_edge = np.mean([b['edge'] for b in away_bets])
        logger.info(f"AWAY: {away_wins}/{len(away_bets)} ({away_wr:.1%} WR) | Avg Edge: {away_avg_edge:+.3f}")
    else:
        logger.info("AWAY: No bets")
        away_wr = 0

    if total_bets > 0:
        total_wins = sum(b['won'] for b in home_bets) + sum(b['won'] for b in away_bets)
        overall_wr = total_wins / total_bets
        logger.info(f"TOTAL: {total_wins}/{total_bets} ({overall_wr:.1%} WR)")
    else:
        overall_wr = 0
        logger.info("TOTAL: No bets")

    return {
        'name': name,
        'home_bets': len(home_bets),
        'home_wr': home_wr,
        'away_bets': len(away_bets),
        'away_wr': away_wr,
        'total_bets': total_bets,
        'overall_wr': overall_wr
    }


# Test configurations
results = []

# Baseline (current settings)
logger.info("\n" + "=" * 80)
logger.info("BASELINE (Current Settings)")
logger.info("=" * 80)

baseline_config = OptimizedStrategyConfig(
    min_edge=0.05,
    min_edge_home=0.07,
    kelly_fraction=0.10,
    min_disagreement=0.20,
    home_bias_penalty=0.02,
    prefer_away=True
)
results.append(run_strategy_test("Baseline (Current)", baseline_config, test_df))

# Test 1: Disable prefer_away
test1_config = OptimizedStrategyConfig(
    min_edge=0.05,
    min_edge_home=0.07,
    kelly_fraction=0.10,
    min_disagreement=0.20,
    home_bias_penalty=0.02,
    prefer_away=False  # CHANGED
)
results.append(run_strategy_test("Test 1: prefer_away=False", test1_config, test_df))

# Test 2: Remove home bias penalty
test2_config = OptimizedStrategyConfig(
    min_edge=0.05,
    min_edge_home=0.07,
    kelly_fraction=0.10,
    min_disagreement=0.20,
    home_bias_penalty=0.00,  # CHANGED
    prefer_away=True
)
results.append(run_strategy_test("Test 2: No home bias penalty", test2_config, test_df))

# Test 3: Equal thresholds for home and away
test3_config = OptimizedStrategyConfig(
    min_edge=0.05,
    min_edge_home=0.05,  # CHANGED (same as away)
    kelly_fraction=0.10,
    min_disagreement=0.20,
    home_bias_penalty=0.00,  # CHANGED
    prefer_away=False  # CHANGED
)
results.append(run_strategy_test("Test 3: Equal thresholds, no penalty", test3_config, test_df))

# Test 4: Home bets only (manual filter)
logger.info(f"\n{'=' * 80}")
logger.info("TEST: Home Bets Only (Manual Filter)")
logger.info(f"{'=' * 80}")

strategy = OptimizedBettingStrategy(test3_config)  # Use test3 config
home_only_bets = []

for idx, row in test_df.iterrows():
    if pd.isna(row['spread_home']):
        continue

    should_bet_home, reason = strategy.should_bet(
        model_prob=row['home_win_prob'],
        market_prob=row['market_home_prob'],
        side='home'
    )

    if should_bet_home:
        spread_result = row['point_diff'] + row['spread_home']
        won = spread_result > 0

        home_only_bets.append({
            'won': won,
            'model_prob': row['home_win_prob'],
            'market_prob': row['market_home_prob'],
            'edge': row['home_win_prob'] - row['market_home_prob']
        })

if len(home_only_bets) > 0:
    wins = sum(b['won'] for b in home_only_bets)
    wr = wins / len(home_only_bets)
    avg_edge = np.mean([b['edge'] for b in home_only_bets])
    logger.info(f"HOME ONLY: {wins}/{len(home_only_bets)} ({wr:.1%} WR) | Avg Edge: {avg_edge:+.3f}")

    results.append({
        'name': 'Home Bets Only',
        'home_bets': len(home_only_bets),
        'home_wr': wr,
        'away_bets': 0,
        'away_wr': 0,
        'total_bets': len(home_only_bets),
        'overall_wr': wr
    })

# Summary
logger.info("\n" + "=" * 80)
logger.info("SUMMARY OF ALL TESTS")
logger.info("=" * 80)

summary_df = pd.DataFrame(results)
logger.info(f"\n{summary_df.to_string(index=False)}")

# Find best configuration
best = summary_df.loc[summary_df['overall_wr'].idxmax()]
logger.info(f"\n✅ BEST CONFIGURATION: {best['name']}")
logger.info(f"   Win Rate: {best['overall_wr']:.1%}")
logger.info(f"   Total Bets: {best['total_bets']}")
logger.info(f"   Home: {best['home_bets']} bets ({best['home_wr']:.1%})")
logger.info(f"   Away: {best['away_bets']} bets ({best['away_wr']:.1%})")

# Save results
summary_df.to_csv('logs/away_bet_diagnosis.csv', index=False)
logger.info(f"\n✅ Results saved to logs/away_bet_diagnosis.csv")

logger.info("\n" + "=" * 80)
logger.info("DIAGNOSIS COMPLETE")
logger.info("=" * 80)
