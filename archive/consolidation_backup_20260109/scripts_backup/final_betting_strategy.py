"""
Final Betting Strategy - Consolidated edge analysis

Based on comprehensive backtesting, these are the validated profitable strategies:

TIER 1 - Highest confidence (p < 0.01):
- Edge 5+ & No B2B: 54.1% win rate, +3.2% ROI, p=0.0085, n=860

TIER 2 - Strong signals (p < 0.05):
- Edge 5+ & Rest Aligns: 57.8% win rate, +10.3% ROI, p=0.014, n=199
- Edge 5+ & Small Spread: 56.8% win rate, +8.4% ROI, p=0.026, n=206
- Edge 5+ (any): 52.6% win rate, +0.5% ROI, p=0.04, n=1695

Key findings from market analysis:
- Pure market biases no longer work (efficient markets)
- Our model is ~0.8 points worse than Vegas at raw prediction
- BUT when model strongly disagrees (5+ pts), it adds value
- Combining model edge with situational filters amplifies the edge
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
print("FINAL BETTING STRATEGY VALIDATION")
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

# Walk-forward validation with periodic retraining
def walk_forward_backtest(df, feature_cols, initial_train_years=4, retrain_months=3):
    """Walk-forward backtest with periodic retraining."""

    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['home_spread'].notna()].copy()

    # Start testing from 2022
    test_start = pd.Timestamp('2022-10-01')
    df_train_initial = df[df['date'] < test_start]
    df_test = df[df['date'] >= test_start].copy()

    if len(df_train_initial) < 500:
        print("Not enough training data")
        return None

    # Initialize model
    model = PointSpreadModel()
    last_train_date = test_start

    results = []

    for idx, row in df_test.iterrows():
        game_date = row['date']

        # Retrain every N months
        if (game_date - last_train_date).days > retrain_months * 30:
            train_data = df[df['date'] < game_date].copy()
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['point_diff']

            val_size = int(0.2 * len(train_data))
            model.fit(X_train[:-val_size], y_train[:-val_size],
                     X_train[-val_size:], y_train[-val_size:],
                     feature_columns=feature_cols)
            last_train_date = game_date

        # Make prediction
        X = pd.DataFrame([row[feature_cols].fillna(0)])
        pred_diff = model.predict(X)[0]

        results.append({
            'game_id': row['game_id'],
            'date': game_date,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'pred_diff': pred_diff,
            'actual_diff': row['actual_diff'],
            'home_spread': row['home_spread'],
            'home_b2b': row.get('home_b2b', False),
            'away_b2b': row.get('away_b2b', False),
            'rest_advantage': row.get('rest_advantage', 0),
        })

    # Initial training
    X_train = df_train_initial[feature_cols].fillna(0)
    y_train = df_train_initial['point_diff']
    val_size = int(0.2 * len(df_train_initial))
    model.fit(X_train[:-val_size], y_train[:-val_size],
             X_train[-val_size:], y_train[-val_size:],
             feature_columns=feature_cols)

    # Process all test games
    for idx, row in df_test.iterrows():
        game_date = row['date']

        # Retrain periodically
        if (game_date - last_train_date).days > retrain_months * 30:
            train_data = df[df['date'] < game_date].copy()
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['point_diff']
            val_size = int(0.2 * len(train_data))
            model.fit(X_train[:-val_size], y_train[:-val_size],
                     X_train[-val_size:], y_train[-val_size:],
                     feature_columns=feature_cols)
            last_train_date = game_date

        X = pd.DataFrame([row[feature_cols].fillna(0)])
        pred_diff = model.predict(X)[0]

        results.append({
            'game_id': row['game_id'],
            'date': game_date,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'pred_diff': pred_diff,
            'actual_diff': row['actual_diff'],
            'home_spread': row['home_spread'],
            'home_b2b': row.get('home_b2b', False),
            'away_b2b': row.get('away_b2b', False),
            'rest_advantage': row.get('rest_advantage', 0),
        })

    return pd.DataFrame(results)

# Quick backtest using single model trained on pre-2022 data
print("\nRunning backtest...")

test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[(all_features['date'] >= test_start) & (all_features['home_spread'].notna())].copy()

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']
val_size = int(0.2 * len(train_data))

model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
         X_train[-val_size:], y_train[-val_size:],
         feature_columns=feature_cols)

X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']
test_data['year'] = test_data['date'].dt.year

# Define strategies
def eval_strategy(name, df, home_mask, away_mask):
    """Evaluate a betting strategy."""
    # Align masks to df index
    home_mask = home_mask.reindex(df.index, fill_value=False)
    away_mask = away_mask.reindex(df.index, fill_value=False)

    home_bets = df[home_mask]
    away_bets = df[away_mask]

    home_wins = home_bets['home_covers'].sum() if len(home_bets) > 0 else 0
    away_wins = (~away_bets['home_covers']).sum() if len(away_bets) > 0 else 0

    total_bets = len(home_bets) + len(away_bets)
    total_wins = home_wins + away_wins

    if total_bets < 30:
        return None

    win_rate = total_wins / total_bets
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    z = (win_rate - 0.50) / np.sqrt(0.25 / total_bets)
    p = 1 - stats.norm.cdf(z)

    return {
        'name': name,
        'n': total_bets,
        'wins': total_wins,
        'win_rate': win_rate,
        'roi': roi,
        'p_value': p
    }

# Calculate strategy masks
df = test_data

# Core edge conditions
edge_5_home = df['model_edge'] >= 5
edge_5_away = df['model_edge'] <= -5

# Situational filters
no_home_b2b = ~df['home_b2b'].fillna(False)
no_away_b2b = ~df['away_b2b'].fillna(False)
rest_aligns_home = df['rest_advantage'].fillna(0) >= 0
rest_aligns_away = df['rest_advantage'].fillna(0) <= 0
small_spread = (df['home_spread'] >= -3) & (df['home_spread'] < 3)

print("\n" + "=" * 70)
print("STRATEGY PERFORMANCE (2022-2025)")
print("=" * 70)

strategies = [
    # Tier 1 - Base edge strategies
    ("Edge 5+ (baseline)", edge_5_home, edge_5_away),

    # Tier 2 - Combined strategies
    ("Edge 5+ & No B2B",
     edge_5_home & no_home_b2b,
     edge_5_away & no_away_b2b),

    ("Edge 5+ & Rest Aligns",
     edge_5_home & rest_aligns_home,
     edge_5_away & rest_aligns_away),

    ("Edge 5+ & Small Spread",
     edge_5_home & small_spread,
     edge_5_away & small_spread),

    # Tier 3 - Triple combinations
    ("Edge 5+ & No B2B & Rest Aligns",
     edge_5_home & no_home_b2b & rest_aligns_home,
     edge_5_away & no_away_b2b & rest_aligns_away),

    ("Edge 5+ & No B2B & Small Spread",
     edge_5_home & no_home_b2b & small_spread,
     edge_5_away & no_away_b2b & small_spread),

    ("Edge 5+ & Rest Aligns & Small Spread",
     edge_5_home & rest_aligns_home & small_spread,
     edge_5_away & rest_aligns_away & small_spread),
]

results = []
for name, home_mask, away_mask in strategies:
    r = eval_strategy(name, df, home_mask, away_mask)
    if r:
        results.append(r)
        sig = "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"\n{name}:")
        print(f"  Win rate: {r['win_rate']*100:.1f}% ({r['wins']}/{r['n']})")
        print(f"  ROI: {r['roi']:+.1f}%")
        print(f"  p-value: {r['p_value']:.4f} {sig}")

# Year-by-year breakdown for top strategies
print("\n" + "=" * 70)
print("YEAR-BY-YEAR BREAKDOWN - TOP STRATEGIES")
print("=" * 70)

top_strategies = [
    ("Edge 5+ & No B2B", edge_5_home & no_home_b2b, edge_5_away & no_away_b2b),
    ("Edge 5+ & Rest Aligns", edge_5_home & rest_aligns_home, edge_5_away & rest_aligns_away),
]

for name, home_mask, away_mask in top_strategies:
    print(f"\n--- {name} ---")
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        r = eval_strategy(f"{name} ({year})", year_df,
                         home_mask.reindex(year_df.index, fill_value=False),
                         away_mask.reindex(year_df.index, fill_value=False))
        if r:
            print(f"  {year}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

# Recommended strategy summary
print("\n" + "=" * 70)
print("RECOMMENDED BETTING RULES")
print("=" * 70)

print("""
PRIMARY STRATEGY: Edge 5+ & No B2B
- Bet HOME when: model_edge >= 5 AND home team NOT on back-to-back
- Bet AWAY when: model_edge <= -5 AND away team NOT on back-to-back
- Expected win rate: ~54%, ROI: ~3%
- Statistical significance: p < 0.01

ENHANCED STRATEGY: Add Rest Alignment Filter
- Same as above, but also require rest advantage aligns with bet direction
- Expected win rate: ~58%, ROI: ~10%
- Trade-off: Fewer bets (~200 vs ~860)

PRACTICAL IMPLEMENTATION:
1. Run model daily to get predictions for upcoming games
2. Calculate model_edge = pred_diff + home_spread
3. Apply filters:
   - |model_edge| >= 5
   - No back-to-back for team being bet
   - (Optional) Rest advantage aligns
4. Place bets at closing lines
5. Track performance monthly

BANKROLL MANAGEMENT:
- Flat bet 1-2% of bankroll per game
- ~100-200 bets per season with primary strategy
- Expected value: +$30-60 per $1000 bet
""")

# Show current season performance
print("\n" + "=" * 70)
print("2024-25 SEASON PERFORMANCE (Current)")
print("=" * 70)

current = df[df['date'] >= '2024-10-01'].copy()
for name, home_mask, away_mask in strategies[:4]:
    r = eval_strategy(name, current,
                     home_mask.reindex(current.index, fill_value=False),
                     away_mask.reindex(current.index, fill_value=False))
    if r:
        print(f"{name}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")
