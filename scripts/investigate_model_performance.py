"""
Investigate Why Model Has 40% Win Rate

Analyzes model predictions vs actual outcomes to understand poor performance.

Issues to investigate:
1. Why is model worse than random (40% vs 50%)?
2. Why are away bets performing terribly (35.7% vs expected 56%)?
3. Are model probabilities calibrated correctly?
4. Is there a systematic bias in predictions?
5. Are features calculated correctly?

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

# Disable Four Factors (data leakage)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

logger.info("=" * 80)
logger.info("MODEL PERFORMANCE INVESTIGATION")
logger.info("=" * 80)

# Load data
logger.info("\nLoading data...")
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')

# Prepare team mapping
TEAM_MAP = {
    'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
    'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'nop': 'NOP',
    'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI',
    'phx': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
    'uta': 'UTA', 'uta': 'UTA', 'was': 'WAS',
}

odds['home_team'] = odds['home_team'].str.lower().map(TEAM_MAP)
odds['away_team'] = odds['away_team'].str.lower().map(TEAM_MAP)

# Merge odds with games
merged = games.merge(odds, on=['home_team', 'away_team'], how='inner')
merged = merged[merged['spread_home'].notna()].copy()
merged['point_diff'] = merged['home_score'] - merged['away_score']

logger.info(f"Games with spreads: {len(merged)}")

# Build features
logger.info("\nBuilding features...")
team_builder = TeamFeatureBuilder()
team_features = team_builder.build_all_features(games)

game_builder = GameFeatureBuilder(team_features)
all_features = game_builder.build_all_features(games)

# Merge
all_features = all_features.merge(
    merged[['game_id', 'spread_home', 'point_diff']].drop_duplicates(),
    on='game_id',
    how='left'
)

# Split train/test
train_end = pd.Timestamp('2022-06-16')
test_start = pd.Timestamp('2022-10-02')

train_df = all_features[all_features['game_date'] <= train_end].copy()
test_df = all_features[all_features['game_date'] >= test_start].copy()

logger.info(f"\nTrain games: {len(train_df)}")
logger.info(f"Test games: {len(test_df)}")

# Train model
logger.info("\nTraining model...")
model = DualPredictionModel()

feature_cols = [c for c in train_df.columns if c not in [
    'game_id', 'home_team', 'away_team', 'game_date', 'season',
    'home_score', 'away_score', 'point_diff', 'spread_home', 'home_win'
]]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['home_win'].astype(int)

model.fit(X_train, y_train)

# Predict on test set
logger.info("\nGenerating predictions...")
X_test = test_df[feature_cols].fillna(0)
test_df['model_prob'] = model.predict_proba(X_test)

# Calculate market probabilities from spreads
# Use simplified conversion: spread ~= -2 * log(prob / (1-prob))
# More accurate: use historical spread-to-probability mapping
test_df['market_prob'] = 0.5 + (test_df['spread_home'] * 0.025)  # Simplified
test_df['market_prob'] = test_df['market_prob'].clip(0.1, 0.9)

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS 1: Model Calibration")
logger.info("=" * 80)

# Check if model probabilities are calibrated
bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
bin_labels = ['<40%', '40-45%', '45-50%', '50-55%', '55-60%', '>60%']

test_df['prob_bin'] = pd.cut(test_df['model_prob'], bins=bins, labels=bin_labels)

calibration = test_df.groupby('prob_bin', observed=True).agg({
    'home_win': ['count', 'mean'],
    'model_prob': 'mean'
})

logger.info("\nModel Calibration (Predicted vs Actual):")
logger.info(f"{'Prob Range':<15} {'Count':<10} {'Model Prob':<15} {'Actual Win %':<15} {'Diff':<10}")
logger.info("-" * 75)

for idx in calibration.index:
    count = int(calibration.loc[idx, ('home_win', 'count')])
    model_avg = calibration.loc[idx, ('model_prob', 'mean')]
    actual_pct = calibration.loc[idx, ('home_win', 'mean')]
    diff = actual_pct - model_avg

    logger.info(f"{idx:<15} {count:<10} {model_avg:<15.3f} {actual_pct:<15.3f} {diff:+.3f}")

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS 2: Betting Performance by Confidence")
logger.info("=" * 80)

# Simulate betting at different confidence thresholds
test_df['edge'] = test_df['model_prob'] - test_df['market_prob']
test_df['abs_edge'] = test_df['edge'].abs()

edge_bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
edge_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']

test_df['edge_bin'] = pd.cut(test_df['abs_edge'], bins=edge_bins, labels=edge_labels)

logger.info("\nPerformance by Edge Size:")
logger.info(f"{'Edge Range':<15} {'Count':<10} {'Win %':<15} {'Avg Edge':<15}")
logger.info("-" * 60)

for edge_label in edge_labels:
    subset = test_df[test_df['edge_bin'] == edge_label]
    if len(subset) == 0:
        continue

    # Determine which side to bet based on edge
    home_bets = subset[subset['edge'] > 0]
    away_bets = subset[subset['edge'] < 0]

    # Calculate outcomes
    home_wins = (home_bets['home_win'] == 1).sum() if len(home_bets) > 0 else 0
    away_wins = (away_bets['home_win'] == 0).sum() if len(away_bets) > 0 else 0

    total_wins = home_wins + away_wins
    total_bets = len(home_bets) + len(away_bets)
    win_pct = total_wins / total_bets if total_bets > 0 else 0
    avg_edge = subset['abs_edge'].mean()

    logger.info(f"{edge_label:<15} {total_bets:<10} {win_pct:<15.1%} {avg_edge:<15.3f}")

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS 3: Home vs Away Bet Performance")
logger.info("=" * 80)

# Calculate spread results
test_df['spread_result'] = test_df['point_diff'] + test_df['spread_home']

strategy = OptimizedBettingStrategy(OptimizedStrategyConfig())

# Simulate bets
home_bets_list = []
away_bets_list = []

for idx, row in test_df.iterrows():
    model_prob = row['model_prob']
    market_prob = row['market_prob']

    # Check if would bet on home
    should_bet_home, _ = strategy.should_bet(
        model_prob=model_prob,
        market_prob=market_prob,
        side='home'
    )

    # Check if would bet on away
    should_bet_away, _ = strategy.should_bet(
        model_prob=1 - model_prob,
        market_prob=1 - market_prob,
        side='away'
    )

    if should_bet_home:
        # Home covers if spread_result > 0
        won = row['spread_result'] > 0
        home_bets_list.append({
            'game_id': row['game_id'],
            'model_prob': model_prob,
            'market_prob': market_prob,
            'edge': model_prob - market_prob,
            'spread': row['spread_home'],
            'point_diff': row['point_diff'],
            'spread_result': row['spread_result'],
            'won': won
        })

    if should_bet_away:
        # Away covers if spread_result < 0
        won = row['spread_result'] < 0
        away_bets_list.append({
            'game_id': row['game_id'],
            'model_prob': 1 - model_prob,
            'market_prob': 1 - market_prob,
            'edge': (1 - model_prob) - (1 - market_prob),
            'spread': row['spread_home'],
            'point_diff': row['point_diff'],
            'spread_result': row['spread_result'],
            'won': won
        })

home_bets_df = pd.DataFrame(home_bets_list)
away_bets_df = pd.DataFrame(away_bets_list)

logger.info(f"\nHome Bets: {len(home_bets_df)}")
if len(home_bets_df) > 0:
    home_wins = home_bets_df['won'].sum()
    home_win_pct = home_wins / len(home_bets_df)
    home_avg_edge = home_bets_df['edge'].mean()
    logger.info(f"  Wins: {home_wins}/{len(home_bets_df)} ({home_win_pct:.1%})")
    logger.info(f"  Avg Edge: {home_avg_edge:.3f}")
    logger.info(f"  Avg Spread: {home_bets_df['spread'].mean():.2f}")
    logger.info(f"  Avg Spread Result: {home_bets_df['spread_result'].mean():+.2f}")

logger.info(f"\nAway Bets: {len(away_bets_df)}")
if len(away_bets_df) > 0:
    away_wins = away_bets_df['won'].sum()
    away_win_pct = away_wins / len(away_bets_df)
    away_avg_edge = away_bets_df['edge'].mean()
    logger.info(f"  Wins: {away_wins}/{len(away_bets_df)} ({away_win_pct:.1%})")
    logger.info(f"  Avg Edge: {away_avg_edge:.3f}")
    logger.info(f"  Avg Spread: {away_bets_df['spread'].mean():.2f}")
    logger.info(f"  Avg Spread Result: {away_bets_df['spread_result'].mean():+.2f}")

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS 4: Sample Losing Bets")
logger.info("=" * 80)

if len(away_bets_df) > 0:
    logger.info("\nSample AWAY bets that LOST (first 10):")
    losing_away = away_bets_df[~away_bets_df['won']].head(10)

    for idx, bet in losing_away.iterrows():
        logger.info(f"\n  Spread: {bet['spread']:+.1f}")
        logger.info(f"  Point Diff: {bet['point_diff']:+.1f}")
        logger.info(f"  Spread Result: {bet['spread_result']:+.1f}")
        logger.info(f"  Model Prob (away): {bet['model_prob']:.3f}")
        logger.info(f"  Market Prob (away): {bet['market_prob']:.3f}")
        logger.info(f"  Edge: {bet['edge']:+.3f}")
        logger.info(f"  Outcome: Away covers if spread_result < 0: {bet['spread_result']} < 0? {bet['won']}")

logger.info("\n" + "=" * 80)
logger.info("INVESTIGATION COMPLETE")
logger.info("=" * 80)

# Export detailed results
if len(home_bets_df) > 0 or len(away_bets_df) > 0:
    all_bets = pd.concat([
        home_bets_df.assign(side='home'),
        away_bets_df.assign(side='away')
    ], ignore_index=True)

    all_bets.to_csv('logs/model_investigation_bets.csv', index=False)
    logger.info("\n✅ Exported bet details to logs/model_investigation_bets.csv")
