"""
Investigate Edge Decay - Why did 2025 performance drop?

Analysis:
1. Model prediction accuracy over time
2. Spread efficiency (model vs line accuracy)
3. Edge distribution (are large edges still occurring?)
4. Feature importance changes
5. Market adjustment
"""

import sys
sys.path.insert(0, '.')
import os
from dotenv import load_dotenv
load_dotenv('.env')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder
from src.models.point_spread import PointSpreadModel
from scipy import stats

# Skip Four Factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("EDGE DECAY INVESTIGATION")
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
odds['home_spread'] = odds.apply(
    lambda r: -r['spread'] if r['whos_favored'] == 'home' else r['spread'],
    axis=1
)

merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'home_spread']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

# Build features
print("\nBuilding features...")
builder = GameFeatureBuilder()
all_features = builder.build_game_features(merged)

feature_cols = [c for c in builder.get_feature_columns(all_features)
                if c != 'home_spread'
                and 'OREB_PCT' not in c
                and 'EFG_PCT' not in c]

all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'home_score', 'away_score']].drop_duplicates(),
    on='game_id',
    how='left',
    suffixes=('', '_m')
)

all_features['actual_diff'] = all_features['home_score'] - all_features['away_score']

# Train model on pre-2022 data
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[(all_features['date'] >= test_start) & (all_features['home_spread'].notna())].copy()

print(f"Train: {len(train_data)}")
print(f"Test: {len(test_data)}")

# Train model
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']

val_size = int(0.2 * len(train_data))
model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
          X_train[-val_size:], y_train[-val_size:],
          feature_columns=feature_cols)

# Get predictions for test data
X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['year'] = test_data['date'].dt.year

# Calculate errors
test_data['model_error'] = test_data['pred_diff'] - test_data['actual_diff']
test_data['line_error'] = -test_data['home_spread'] - test_data['actual_diff']  # Line implied diff

print("\n" + "=" * 70)
print("1. MODEL PREDICTION ACCURACY BY YEAR")
print("=" * 70)

for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    model_mae = np.abs(year_data['model_error']).mean()
    line_mae = np.abs(year_data['line_error']).mean()
    model_rmse = np.sqrt((year_data['model_error']**2).mean())
    line_rmse = np.sqrt((year_data['line_error']**2).mean())

    # Correlation with actual
    model_corr = year_data['pred_diff'].corr(year_data['actual_diff'])
    line_corr = (-year_data['home_spread']).corr(year_data['actual_diff'])

    print(f"\n{year}:")
    print(f"  Model MAE: {model_mae:.2f}, RMSE: {model_rmse:.2f}, Corr: {model_corr:.3f}")
    print(f"  Line  MAE: {line_mae:.2f}, RMSE: {line_rmse:.2f}, Corr: {line_corr:.3f}")
    print(f"  Model vs Line: {'Model better' if model_mae < line_mae else 'Line better'}")

print("\n" + "=" * 70)
print("2. EDGE DISTRIBUTION BY YEAR")
print("=" * 70)

for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]

    # Distribution of model edge
    edge_mean = year_data['model_edge'].mean()
    edge_std = year_data['model_edge'].std()

    # Large edges (5+ points)
    large_pos = (year_data['model_edge'] >= 5).sum()
    large_neg = (year_data['model_edge'] <= -5).sum()
    total = len(year_data)

    print(f"\n{year}:")
    print(f"  Edge mean: {edge_mean:.2f}, std: {edge_std:.2f}")
    print(f"  Large edges (5+): {large_pos} ({large_pos/total*100:.1f}%)")
    print(f"  Large edges (-5-): {large_neg} ({large_neg/total*100:.1f}%)")
    print(f"  Total games: {total}")

print("\n" + "=" * 70)
print("3. SPREAD LINE TIGHTNESS BY YEAR")
print("=" * 70)

print("\nAre spreads getting more accurate over time?")
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]

    # How often does the line predict winner correctly?
    line_pred_home = year_data['home_spread'] < 0  # Line thinks home wins
    actual_home_won = year_data['actual_diff'] > 0
    line_accuracy = (line_pred_home == actual_home_won).mean()

    # Spread vs actual margin correlation
    spread_corr = (-year_data['home_spread']).corr(year_data['actual_diff'])

    # Average absolute spread
    avg_spread = np.abs(year_data['home_spread']).mean()

    print(f"\n{year}:")
    print(f"  Line winner accuracy: {line_accuracy*100:.1f}%")
    print(f"  Spread-margin correlation: {spread_corr:.3f}")
    print(f"  Avg absolute spread: {avg_spread:.1f}")

print("\n" + "=" * 70)
print("4. ATS PERFORMANCE BY EDGE SIZE AND YEAR")
print("=" * 70)

# When we have large edge, how often does it cover?
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']

for edge_thresh in [3, 5, 7]:
    print(f"\n--- Edge threshold: {edge_thresh}+ ---")
    for year in sorted(test_data['year'].unique()):
        year_data = test_data[test_data['year'] == year]

        # Bet home when edge >= threshold
        home_edge = year_data[year_data['model_edge'] >= edge_thresh]
        if len(home_edge) > 0:
            home_ats = home_edge['home_covers'].mean()
            home_n = len(home_edge)
        else:
            home_ats, home_n = 0, 0

        # Bet away when edge <= -threshold
        away_edge = year_data[year_data['model_edge'] <= -edge_thresh]
        if len(away_edge) > 0:
            away_ats = (~away_edge['home_covers']).mean()
            away_n = len(away_edge)
        else:
            away_ats, away_n = 0, 0

        total_ats = (home_ats * home_n + away_ats * away_n) / max(home_n + away_n, 1)

        print(f"  {year}: Home {home_ats*100:.1f}% (n={home_n}), Away {away_ats*100:.1f}% (n={away_n}), Combined {total_ats*100:.1f}%")

print("\n" + "=" * 70)
print("5. MODEL DISAGREEMENT WITH LINE BY YEAR")
print("=" * 70)

print("\nHow often does model agree with line direction?")
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]

    # Model thinks home wins (pred_diff > 0) vs Line thinks home wins (spread < 0)
    model_home = year_data['pred_diff'] > 0
    line_home = year_data['home_spread'] < 0
    agree = (model_home == line_home).mean()

    # When they disagree, who is right?
    disagree_mask = model_home != line_home
    if disagree_mask.sum() > 0:
        disagree_data = year_data[disagree_mask]
        model_right = ((disagree_data['pred_diff'] > 0) == (disagree_data['actual_diff'] > 0)).mean()
        line_right = ((disagree_data['home_spread'] < 0) == (disagree_data['actual_diff'] > 0)).mean()
    else:
        model_right, line_right = 0, 0

    print(f"\n{year}:")
    print(f"  Model-Line agreement: {agree*100:.1f}%")
    print(f"  When disagreeing ({disagree_mask.sum()} games):")
    print(f"    Model correct: {model_right*100:.1f}%")
    print(f"    Line correct: {line_right*100:.1f}%")

print("\n" + "=" * 70)
print("6. MONTHLY PERFORMANCE (2024-2025)")
print("=" * 70)

recent = test_data[test_data['date'] >= '2024-01-01'].copy()
recent['month'] = recent['date'].dt.to_period('M')

print("\nLarge edge (5+) ATS by month:")
for month in sorted(recent['month'].unique()):
    month_data = recent[recent['month'] == month]
    large_edge = month_data[np.abs(month_data['model_edge']) >= 5]
    if len(large_edge) > 0:
        # Bet with model
        model_right = (
            ((large_edge['model_edge'] >= 5) & large_edge['home_covers']) |
            ((large_edge['model_edge'] <= -5) & ~large_edge['home_covers'])
        ).mean()
        print(f"  {month}: {model_right*100:.1f}% ATS (n={len(large_edge)})")

print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

# Calculate if model is still better than line
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    model_mae = np.abs(year_data['model_error']).mean()
    line_mae = np.abs(year_data['line_error']).mean()
    edge = line_mae - model_mae
    print(f"{year}: Model edge over line: {edge:.2f} points MAE")
