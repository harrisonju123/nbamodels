"""
Backtest Point Spread Model with Elo Features

Compares performance of spread model with vs without Elo features.
"""

import sys
sys.path.insert(0, '.')
import os
from dotenv import load_dotenv
load_dotenv('.env')

import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from scipy import stats
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder

# Skip Four Factors (has data leakage)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("BACKTEST: POINT SPREAD MODEL WITH ELO")
print("=" * 70)

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
games['date'] = pd.to_datetime(games['date'])

# Fix spread sign: negative = home favored
odds['home_spread'] = odds.apply(
    lambda r: -r['spread'] if r['whos_favored'] == 'home' else r['spread'],
    axis=1
)

# Merge
merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'home_spread']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

print(f"Total games: {len(merged)}")
print(f"Games with spreads: {merged['home_spread'].notna().sum()}")

# Build features
print("\nBuilding features (with Elo)...")
builder = GameFeatureBuilder(use_elo=True)
all_features = builder.build_game_features(merged)

# Merge spreads
all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'point_diff']].drop_duplicates(),
    on='game_id', how='left', suffixes=('', '_merged')
)
if 'point_diff_merged' in all_features.columns:
    all_features['point_diff'] = all_features['point_diff'].fillna(all_features['point_diff_merged'])
    all_features = all_features.drop(columns=['point_diff_merged'])

# Feature columns - WITH Elo
feature_cols_with_elo = [c for c in builder.get_feature_columns(all_features)
                         if c != 'home_spread'
                         and 'OREB_PCT' not in c and 'EFG_PCT' not in c
                         and 'TS_PCT' not in c and 'TOV_PCT' not in c]

# Feature columns - WITHOUT Elo (for comparison)
feature_cols_no_elo = [c for c in feature_cols_with_elo
                       if 'elo' not in c.lower()]

print(f"Features with Elo: {len(feature_cols_with_elo)}")
print(f"Features without Elo: {len(feature_cols_no_elo)}")
print(f"Elo features: {[c for c in feature_cols_with_elo if 'elo' in c.lower()]}")

# Parameters
MIN_EDGE = 3.0  # Minimum edge in points to bet
LINE_MOVEMENT = 0.5
BET_SIZE = 10
STARTING_BANKROLL = 1000

# Test period
test_start_date = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start_date].copy()
test_data = all_features[(all_features['date'] >= test_start_date) &
                          (all_features['home_spread'].notna())].copy()

print(f"\nTrain: {len(train_data)} games")
print(f"Test: {len(test_data)} games with spreads")


def train_spread_model(X_train, y_train, X_val, y_val):
    """Train XGBoost spread model."""
    params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_backtest(feature_cols, model_name):
    """Run backtest with given feature set."""
    print(f"\n{'='*50}")
    print(f"TESTING: {model_name}")
    print(f"{'='*50}")

    # Train/val split for training
    train_size = int(0.85 * len(train_data))
    train_subset = train_data[:train_size]
    val_subset = train_data[train_size:]

    X_train = train_subset[feature_cols].fillna(0)
    y_train = train_subset['point_diff']
    X_val = val_subset[feature_cols].fillna(0)
    y_val = val_subset['point_diff']

    model = train_spread_model(X_train, y_train, X_val, y_val)

    # Calibrate std from validation
    val_preds = model.predict(X_val)
    calibrated_std = np.std(y_val - val_preds)
    print(f"Calibrated std: {calibrated_std:.2f} points")

    # Test predictions
    X_test = test_data[feature_cols].fillna(0)
    test_preds = model.predict(X_test)
    test_data_copy = test_data.copy()
    test_data_copy['pred_diff'] = test_preds

    # Calculate edge
    test_data_copy['model_edge'] = test_data_copy['pred_diff'] + test_data_copy['home_spread']

    # Betting results
    bets = []
    bankroll = STARTING_BANKROLL

    for _, row in test_data_copy.iterrows():
        edge = row['model_edge']
        spread = row['home_spread']
        actual_diff = row['point_diff']

        if abs(edge) >= MIN_EDGE:
            if edge > 0:
                # Bet home to cover
                adjusted_spread = spread - LINE_MOVEMENT
                won = actual_diff > -adjusted_spread
                bet_side = 'home'
            else:
                # Bet away to cover
                adjusted_spread = spread + LINE_MOVEMENT
                won = actual_diff < -adjusted_spread
                bet_side = 'away'

            pnl = BET_SIZE * 0.909 if won else -BET_SIZE
            bankroll += pnl

            bets.append({
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet_side': bet_side,
                'spread': spread,
                'pred_diff': row['pred_diff'],
                'edge': edge,
                'actual_diff': actual_diff,
                'won': won,
                'pnl': pnl,
            })

    if len(bets) == 0:
        print("No bets placed!")
        return None

    bets_df = pd.DataFrame(bets)
    wins = bets_df['won'].sum()
    total_pnl = bets_df['pnl'].sum()
    win_rate = wins / len(bets_df)
    roi = total_pnl / (len(bets_df) * BET_SIZE)

    print(f"\nResults:")
    print(f"  Total bets: {len(bets_df)}")
    print(f"  Wins: {wins}, Losses: {len(bets_df) - wins}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  ROI: {roi*100:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Final bankroll: ${bankroll:.2f}")

    # By edge bucket
    print(f"\n  By Edge Size:")
    for edge_min, edge_max in [(3, 5), (5, 7), (7, 10), (10, 100)]:
        bucket = bets_df[(bets_df['edge'].abs() >= edge_min) & (bets_df['edge'].abs() < edge_max)]
        if len(bucket) > 0:
            bw = bucket['won'].sum()
            bwr = bw / len(bucket) * 100
            broi = bucket['pnl'].sum() / (len(bucket) * BET_SIZE) * 100
            print(f"    {edge_min}-{edge_max} pts: {len(bucket)} bets, {bwr:.1f}% win, {broi:.1f}% ROI")

    # By year
    print(f"\n  By Season:")
    bets_df['season'] = bets_df['date'].apply(lambda d: d.year if d.month >= 10 else d.year - 1)
    for season in sorted(bets_df['season'].unique()):
        season_bets = bets_df[bets_df['season'] == season]
        sw = season_bets['won'].sum()
        swr = sw / len(season_bets) * 100
        sroi = season_bets['pnl'].sum() / (len(season_bets) * BET_SIZE) * 100
        print(f"    {season}-{season+1}: {len(season_bets)} bets, {swr:.1f}% win, {sroi:.1f}% ROI")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n  Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        elo_imp = importance[importance['feature'].str.contains('elo', case=False)]
        if len(elo_imp) > 0:
            print(f"\n  Elo Features:")
            for _, row in elo_imp.iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

    return {
        'bets': len(bets_df),
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi,
        'pnl': total_pnl,
        'bankroll': bankroll,
    }


# Run both backtests
results_no_elo = run_backtest(feature_cols_no_elo, "WITHOUT Elo")
results_with_elo = run_backtest(feature_cols_with_elo, "WITH Elo")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

if results_no_elo and results_with_elo:
    print(f"\n{'Metric':<20} {'Without Elo':<15} {'With Elo':<15} {'Diff':<10}")
    print("-" * 60)
    print(f"{'Bets':<20} {results_no_elo['bets']:<15} {results_with_elo['bets']:<15}")
    print(f"{'Win Rate':<20} {results_no_elo['win_rate']*100:<14.1f}% {results_with_elo['win_rate']*100:<14.1f}% {(results_with_elo['win_rate'] - results_no_elo['win_rate'])*100:+.1f}%")
    print(f"{'ROI':<20} {results_no_elo['roi']*100:<14.1f}% {results_with_elo['roi']*100:<14.1f}% {(results_with_elo['roi'] - results_no_elo['roi'])*100:+.1f}%")
    print(f"{'Total P&L':<20} ${results_no_elo['pnl']:<13.2f} ${results_with_elo['pnl']:<13.2f} ${results_with_elo['pnl'] - results_no_elo['pnl']:+.2f}")

print("\n" + "=" * 70)
