"""
Debug script to understand why EdgeStrategy backtest generates no bets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Load data
print("Loading data...")
features_path = Path("data/features/game_features.parquet")
df = pd.read_parquet(features_path)
print(f"Loaded {len(df)} games from features")

# Load historical odds
odds_path = Path("data/raw/historical_odds.csv")
odds_df = pd.read_csv(odds_path)
print(f"Loaded {len(odds_df)} historical odds records")

# Convert dates
odds_df["date"] = pd.to_datetime(odds_df["date"])

# Merge
df = df.merge(
    odds_df[["date", "home", "away", "spread", "total", "moneyline_home", "moneyline_away"]],
    left_on=["date", "home_team", "away_team"],
    right_on=["date", "home", "away"],
    how="left",
)
df = df.drop(["home", "away"], axis=1, errors="ignore")

# Filter to 2020-2024
if "season" in df.columns:
    df = df[df["season"].between(2020, 2024)]
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year.between(2020, 2024)]

print(f"\nFiltered to {len(df)} games (2020-2024)")

# Check spread data availability
print(f"\n=== Spread Data Availability ===")
print(f"Games with spread data: {df['spread'].notna().sum()} / {len(df)} ({df['spread'].notna().mean():.1%})")
print(f"\nBy season:")
if "season" in df.columns:
    season_spread = df.groupby('season')['spread'].apply(lambda x: f"{x.notna().sum()}/{len(x)} ({x.notna().mean():.1%})")
    for season, stats in season_spread.items():
        print(f"  {season}: {stats}")

# Sample a few games with spreads
print(f"\n=== Sample Games with Spread Data ===")
sample_with_spread = df[df['spread'].notna()].head(5)
for _, row in sample_with_spread.iterrows():
    print(f"{row.get('date', 'N/A')}: {row.get('away_team', 'N/A')} @ {row.get('home_team', 'N/A')}, spread={row['spread']}, point_diff={row.get('point_diff', 'N/A')}")

# Check if all required columns exist
print(f"\n=== Required Columns Check ===")
required = ['point_diff', 'spread', 'home_is_b2b', 'away_is_b2b', 'home_team', 'away_team']
for col in required:
    if col in df.columns:
        has_data = df[col].notna().sum() if col in ['point_diff', 'spread'] else len(df)
        print(f"  ✓ {col}: {has_data}/{len(df)} available")
    else:
        alt_col = col.replace('_is_', '_')  # Try alternate naming
        if alt_col in df.columns:
            print(f"  ~ {col} not found, but {alt_col} exists")
        else:
            print(f"  ✗ {col}: MISSING")

# Quick model test on a sample fold
print(f"\n=== Quick Model Test ===")
train_df = df[(df['season'] >= 2020) & (df['season'] <= 2021)]
test_df = df[df['season'] == 2022].head(50)

print(f"Train: {len(train_df)} games")
print(f"Test: {len(test_df)} games")
print(f"Test games with spread: {test_df['spread'].notna().sum()}")

if test_df['spread'].notna().sum() > 0:
    # Train simple model
    from xgboost import XGBRegressor

    exclude_cols = {
        'game_id', 'date', 'season', 'home_team', 'away_team',
        'home_win', 'away_win', 'home_score', 'away_score',
        'point_diff', 'total_points', 'spread', 'total',
        'home_is_home', 'away_is_home',
        'home_point_diff', 'away_point_diff',
        'moneyline_home', 'moneyline_away',
    }

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    target_col = 'point_diff'

    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[target_col]

    model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on test games with spreads
    test_with_spread = test_df[test_df['spread'].notna()].copy()
    X_test = test_with_spread[feature_cols].fillna(train_df[feature_cols].median())
    predictions = model.predict(X_test)
    test_with_spread['pred_diff'] = predictions
    test_with_spread['edge'] = test_with_spread['pred_diff'] - test_with_spread['spread']

    print(f"\n=== Edge Distribution (first 10 test games with spreads) ===")
    for _, row in test_with_spread.head(10).iterrows():
        print(f"  {row.get('away_team', 'N/A')} @ {row.get('home_team', 'N/A')}: pred_diff={row['pred_diff']:.1f}, spread={row['spread']:.1f}, edge={row['edge']:.1f}")

    print(f"\n=== Edge Statistics ===")
    print(f"Mean edge: {test_with_spread['edge'].abs().mean():.2f} points")
    print(f"Median edge: {test_with_spread['edge'].abs().median():.2f} points")
    print(f"Games with 2+ pts edge: {(test_with_spread['edge'].abs() >= 2).sum()} ({(test_with_spread['edge'].abs() >= 2).mean():.1%})")
    print(f"Games with 4+ pts edge: {(test_with_spread['edge'].abs() >= 4).sum()} ({(test_with_spread['edge'].abs() >= 4).mean():.1%})")
    print(f"Games with 6+ pts edge: {(test_with_spread['edge'].abs() >= 6).sum()} ({(test_with_spread['edge'].abs() >= 6).mean():.1%})")
else:
    print("No test games with spread data to analyze!")
