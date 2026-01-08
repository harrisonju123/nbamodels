"""
Totals Backtest - Test Over/Under betting edge

Totals may be less efficient than spreads because:
1. Pace is harder for oddsmakers to predict
2. B2B/rest affects pace
3. Weather/altitude affects totals
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
from src.models.totals import TotalsModel
from scipy import stats

# Skip Four Factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("TOTALS BACKTEST - Over/Under Betting Edge")
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
odds['total_line'] = odds['total']

merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'total_line']],
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
    return df

merged = add_rest_features(merged)

# Build features
print("\nBuilding features...")
builder = GameFeatureBuilder()
all_features = builder.build_game_features(merged)

# Get feature columns
feature_cols = [c for c in builder.get_feature_columns(all_features)
                if 'OREB_PCT' not in c
                and 'EFG_PCT' not in c]

print(f"Feature columns: {len(feature_cols)}")

# Merge totals and rest
all_features = all_features.merge(
    merged[['game_id', 'total_line', 'home_rest', 'away_rest', 'home_b2b', 'away_b2b', 'home_score', 'away_score']].drop_duplicates(),
    on='game_id',
    how='left',
    suffixes=('', '_m')
)

# Calculate total points
all_features['total_points'] = all_features['home_score'] + all_features['away_score']

# Test/train split
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[all_features['date'] >= test_start].copy()
test_data = test_data.dropna(subset=['total_line'])

print(f"\nTrain: {len(train_data)} games")
print(f"Test: {len(test_data)} games")

# Train totals model
print("\nTraining totals model...")
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['home_score'] + train_data['away_score']

# Split for calibration
val_size = int(0.2 * len(train_data))
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

totals_model = TotalsModel()
totals_model.fit(X_tr, y_tr, X_val, y_val, feature_columns=feature_cols)

print(f"Calibrated std: {totals_model.calibrated_std:.2f}")

# Predictions on test
X_test = test_data[feature_cols].fillna(0)
test_data['pred_total'] = totals_model.predict(X_test)
test_data['model_edge'] = test_data['pred_total'] - test_data['total_line']  # Positive = bet over

# Over/under probability
test_data['over_prob'] = totals_model.predict_over_probs_vectorized(
    X_test, test_data['total_line'].values
)

# Actual results
test_data['went_over'] = test_data['total_points'] > test_data['total_line']
test_data['went_under'] = test_data['total_points'] < test_data['total_line']


def backtest_totals(df, name, bet_mask, bet_over):
    """Run totals backtest on filtered games."""
    games = df[bet_mask].copy()
    n = len(games)
    if n < 30:
        return None

    if bet_over:
        wins = games['went_over'].sum()
    else:
        wins = games['went_under'].sum()

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
print("TOTALS STRATEGY BACKTESTS")
print("=" * 70)

results = []

# Strategy 1: Pure model edge
print("\n--- PURE MODEL EDGE ---")
for edge in [2, 4, 6, 8, 10]:
    # Bet over when model predicts higher
    mask = test_data['model_edge'] >= edge
    r = backtest_totals(test_data, f"Bet OVER when edge >= {edge}", mask, bet_over=True)
    if r:
        results.append(r)
        print(f"  Edge >= {edge}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

    # Bet under when model predicts lower
    mask = test_data['model_edge'] <= -edge
    r = backtest_totals(test_data, f"Bet UNDER when edge <= -{edge}", mask, bet_over=False)
    if r:
        results.append(r)
        print(f"  Edge <= -{edge}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 2: B2B games - tired teams = lower scoring?
print("\n--- B2B SITUATIONAL ---")
# Both teams on B2B (lower energy)
mask = (test_data['home_b2b'] == 1) & (test_data['away_b2b'] == 1)
r = backtest_totals(test_data, "Both B2B (bet UNDER)", mask, bet_over=False)
if r:
    results.append(r)
    print(f"Both B2B (bet UNDER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# One team B2B, other rested
mask = (test_data['home_b2b'] == 1) & (test_data['away_b2b'] == 0)
r = backtest_totals(test_data, "Home B2B only (bet UNDER)", mask, bet_over=False)
if r:
    results.append(r)
    print(f"Home B2B only (bet UNDER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 3: High/Low lines
print("\n--- LINE SIZE ---")
mask = test_data['total_line'] >= 230
r = backtest_totals(test_data, "High lines (230+) bet UNDER", mask, bet_over=False)
if r:
    results.append(r)
    print(f"High lines 230+ (bet UNDER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

mask = test_data['total_line'] <= 215
r = backtest_totals(test_data, "Low lines (<=215) bet OVER", mask, bet_over=True)
if r:
    results.append(r)
    print(f"Low lines <=215 (bet OVER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 4: Model + situational
print("\n--- MODEL + SITUATIONAL ---")
# Model likes under + B2B game
mask = (test_data['model_edge'] <= -4) & ((test_data['home_b2b'] == 1) | (test_data['away_b2b'] == 1))
r = backtest_totals(test_data, "Model <= -4 + any B2B (bet UNDER)", mask, bet_over=False)
if r:
    results.append(r)
    print(f"Model <= -4 + any B2B (bet UNDER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Model likes over + both rested
mask = (test_data['model_edge'] >= 4) & (test_data['home_rest'] >= 2) & (test_data['away_rest'] >= 2)
r = backtest_totals(test_data, "Model >= 4 + both rested 2+ (bet OVER)", mask, bet_over=True)
if r:
    results.append(r)
    print(f"Model >= 4 + both rested 2+ (bet OVER): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

# Strategy 5: High confidence
print("\n--- HIGH CONFIDENCE ---")
for conf in [0.55, 0.60, 0.65]:
    mask = test_data['over_prob'] >= conf
    r = backtest_totals(test_data, f"Over prob >= {conf:.0%}", mask, bet_over=True)
    if r:
        results.append(r)
        print(f"  Over prob >= {conf:.0%}: {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")

    mask = test_data['over_prob'] <= (1-conf)
    r = backtest_totals(test_data, f"Over prob <= {1-conf:.0%}", mask, bet_over=False)
    if r:
        results.append(r)
        print(f"  Under (Over prob <= {1-conf:.0%}): {r['n']:4d} games, {r['win_rate']*100:.1f}%, {r['roi']:+.1f}% ROI, p={r['p_value']:.3f}")


# Summary
print("\n" + "=" * 70)
print("TOP TOTALS STRATEGIES (by ROI, n >= 50)")
print("=" * 70)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    top = results_df[
        (results_df['n'] >= 50) &
        (results_df['roi'] > 0)
    ].sort_values('roi', ascending=False).head(10)

    if len(top) > 0:
        for _, r in top.iterrows():
            sig = "*" if r['p_value'] < 0.10 else ""
            sig = "**" if r['p_value'] < 0.05 else sig
            print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig:2s} {r['name']}")
    else:
        print("No profitable totals strategies found with n >= 50")

    print("\n--- ALL TOTALS STRATEGIES (sorted by ROI) ---")
    for _, r in results_df.sort_values('roi', ascending=False).iterrows():
        sig = "*" if r['p_value'] < 0.10 else ""
        print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig} {r['name']}")

# Baseline
print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)
overall_over = backtest_totals(test_data, "All games OVER", test_data['total_line'].notna(), bet_over=True)
overall_under = backtest_totals(test_data, "All games UNDER", test_data['total_line'].notna(), bet_over=False)
print(f"Overall OVER: {overall_over['win_rate']*100:.1f}% ({overall_over['n']} games)")
print(f"Overall UNDER: {overall_under['win_rate']*100:.1f}% ({overall_under['n']} games)")
