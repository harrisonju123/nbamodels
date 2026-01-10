"""
Edge Backtest - Combine Model Predictions with Situational Factors

Tests:
1. Spread model (predicts margin, converts to cover prob)
2. Totals model
3. Situational filters (B2B, rest, etc.)
4. Combined strategies
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
from src.models.totals import TotalsModel
from scipy import stats

# Skip Four Factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("EDGE BACKTEST - Model + Situational Factors")
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
odds['total_line'] = odds['total']

merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'home_spread', 'total_line']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

# Add rest features
def add_rest_features(df):
    df = df.sort_values('date').copy()
    team_last_game = {}
    home_rest, away_rest, home_b2b, away_b2b = [], [], [], []

    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        game_date = row['date']

        h_rest = (game_date - team_last_game.get(ht, game_date - pd.Timedelta(days=7))).days
        a_rest = (game_date - team_last_game.get(at, game_date - pd.Timedelta(days=7))).days

        home_rest.append(h_rest)
        away_rest.append(a_rest)
        home_b2b.append(1 if h_rest == 1 else 0)
        away_b2b.append(1 if a_rest == 1 else 0)

        team_last_game[ht] = game_date
        team_last_game[at] = game_date

    df['home_rest'] = home_rest
    df['away_rest'] = away_rest
    df['home_b2b'] = home_b2b
    df['away_b2b'] = away_b2b
    df['rest_diff'] = df['home_rest'] - df['away_rest']
    return df

merged = add_rest_features(merged)

# Build features
print("\nBuilding features...")
builder = GameFeatureBuilder()
all_features = builder.build_game_features(merged)

# Get feature columns
feature_cols = [c for c in builder.get_feature_columns(all_features)
                if c != 'home_spread'
                and 'OREB_PCT' not in c
                and 'EFG_PCT' not in c]

print(f"Feature columns: {len(feature_cols)}")

# Merge spreads, rest, and scores into features
all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'total_line', 'home_rest', 'away_rest', 'home_b2b', 'away_b2b', 'rest_diff', 'home_score', 'away_score']].drop_duplicates(),
    on='game_id',
    how='left',
    suffixes=('', '_m')
)

# Test/train split
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[all_features['date'] >= test_start].copy()
test_data = test_data.dropna(subset=['home_spread'])

print(f"\nTrain: {len(train_data)} games ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
print(f"Test: {len(test_data)} games ({test_data['date'].min().date()} to {test_data['date'].max().date()})")

# Train spread model
print("\nTraining point spread model...")
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']

# Split train into train/val for calibration
val_size = int(0.2 * len(train_data))
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

spread_model = PointSpreadModel()
spread_model.fit(X_tr, y_tr, X_val, y_val, feature_columns=feature_cols)

# Evaluate on test
X_test = test_data[feature_cols].fillna(0)
test_preds = spread_model.predict(X_test)
test_data['pred_diff'] = test_preds
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']  # Positive = bet home

print(f"Calibrated std: {spread_model.calibrated_std:.2f}")

# Calculate cover probability
test_data['home_cover_prob'] = spread_model.predict_spread_probs_vectorized(
    X_test, test_data['home_spread'].values
)

# Actual results
test_data['actual_diff'] = test_data['home_score'] - test_data['away_score']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']
test_data['away_covers'] = test_data['actual_diff'] < -test_data['home_spread']


def backtest_strategy(df, name, bet_mask, bet_home):
    """Run backtest on filtered games."""
    games = df[bet_mask].copy()
    n = len(games)
    if n < 30:
        return None

    if bet_home:
        wins = games['home_covers'].sum()
    else:
        wins = games['away_covers'].sum()

    win_rate = wins / n
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    z = (win_rate - 0.50) / np.sqrt(0.25 / n)
    p = 1 - stats.norm.cdf(z)

    return {
        'name': name,
        'n': n,
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi,
        'z_score': z,
        'p_value': p
    }


print("\n" + "=" * 70)
print("STRATEGY BACKTESTS")
print("=" * 70)

results = []

# Strategy 1: Pure model edge - bet when model predicts large edge
print("\n--- PURE MODEL EDGE ---")
for edge_thresh in [1, 2, 3, 4, 5]:
    # Bet home when model_edge > threshold
    mask = test_data['model_edge'] >= edge_thresh
    r = backtest_strategy(test_data, f"Bet HOME when edge >= {edge_thresh}", mask, bet_home=True)
    if r:
        results.append(r)
        print(f"  Edge >= {edge_thresh}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

    # Bet away when model_edge < -threshold
    mask = test_data['model_edge'] <= -edge_thresh
    r = backtest_strategy(test_data, f"Bet AWAY when edge <= -{edge_thresh}", mask, bet_home=False)
    if r:
        results.append(r)
        print(f"  Edge <= -{edge_thresh}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 2: Model + B2B situational
print("\n--- MODEL + SITUATIONAL (B2B) ---")
# Model likes away + home is on B2B
mask = (test_data['model_edge'] <= -1) & (test_data['home_b2b'] == 1)
r = backtest_strategy(test_data, "Model edge <= -1 + Home B2B", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Model <= -1 + Home B2B (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Model likes home + away is on B2B
mask = (test_data['model_edge'] >= 1) & (test_data['away_b2b'] == 1)
r = backtest_strategy(test_data, "Model edge >= 1 + Away B2B", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Model >= 1 + Away B2B (bet HOME): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 3: Model + rest advantage
print("\n--- MODEL + REST ADVANTAGE ---")
mask = (test_data['model_edge'] >= 1) & (test_data['rest_diff'] >= 2)
r = backtest_strategy(test_data, "Model >= 1 + rest advantage 2+", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Model >= 1 + Home rest adv 2+ (bet HOME): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

mask = (test_data['model_edge'] <= -1) & (test_data['rest_diff'] <= -2)
r = backtest_strategy(test_data, "Model <= -1 + away rest advantage 2+", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Model <= -1 + Away rest adv 2+ (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 4: High confidence model picks
print("\n--- HIGH CONFIDENCE MODEL ---")
for conf in [0.55, 0.58, 0.60, 0.65]:
    mask = test_data['home_cover_prob'] >= conf
    r = backtest_strategy(test_data, f"Home cover prob >= {conf:.0%}", mask, bet_home=True)
    if r:
        results.append(r)
        print(f"  Home prob >= {conf:.0%}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

    mask = test_data['home_cover_prob'] <= (1-conf)
    r = backtest_strategy(test_data, f"Home cover prob <= {1-conf:.0%}", mask, bet_home=False)
    if r:
        results.append(r)
        print(f"  Home prob <= {1-conf:.0%}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 5: Best pure situational
print("\n--- PURE SITUATIONAL (no model) ---")
mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2)
r = backtest_strategy(test_data, "Home B2B + Away rested", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B + Away rested 2+ (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 6: Combined - Best situational + model confirmation
print("\n--- SITUATIONAL + MODEL CONFIRMATION ---")
# Home B2B + Away rested + Model agrees (likes away)
mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2) & (test_data['model_edge'] <= 0)
r = backtest_strategy(test_data, "Home B2B + Away rested + Model <= 0", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B + Away 2+ rest + Model agrees (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Stronger filter
mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2) & (test_data['model_edge'] <= -2)
r = backtest_strategy(test_data, "Home B2B + Away rested + Model <= -2", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B + Away 2+ rest + Model <= -2 (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")


# Summary
print("\n" + "=" * 70)
print("TOP STRATEGIES (by ROI, n >= 50)")
print("=" * 70)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    top_strategies = results_df[
        (results_df['n'] >= 50) &
        (results_df['roi'] > 0)
    ].sort_values('roi', ascending=False).head(10)

    if len(top_strategies) > 0:
        for _, r in top_strategies.iterrows():
            sig = "*" if r['p_value'] < 0.10 else ""
            sig = "**" if r['p_value'] < 0.05 else sig
            print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig:2s} {r['name']}")
    else:
        print("No profitable strategies found with n >= 50")

# Show all results
print("\n--- ALL STRATEGIES (sorted by ROI) ---")
for _, r in results_df.sort_values('roi', ascending=False).iterrows():
    sig = "*" if r['p_value'] < 0.10 else ""
    print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig} {r['name']}")
