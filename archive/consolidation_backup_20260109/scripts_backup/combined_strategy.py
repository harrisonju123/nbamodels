"""
Combined Strategy - Best Edges with Realistic Conditions

Combines:
1. Large model edge (5+ points)
2. Model + rest advantage
3. Situational (Home B2B + Away rested)

With realistic conditions:
- Line movement (0.5 points)
- Bet availability (85%)
- Walk-forward training
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
print("COMBINED STRATEGY BACKTEST")
print("Edges: Model 5+, Model+Rest, Situational B2B")
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

feature_cols = [c for c in builder.get_feature_columns(all_features)
                if c != 'home_spread'
                and 'OREB_PCT' not in c
                and 'EFG_PCT' not in c]

# Merge with game data
all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'home_rest', 'away_rest', 'home_b2b', 'away_b2b',
            'rest_diff', 'home_score', 'away_score']].drop_duplicates(),
    on='game_id',
    how='left',
    suffixes=('', '_m')
)

# Parameters
LINE_MOVEMENT = 0.5
BET_AVAILABILITY = 0.85
BET_SIZE = 10
STARTING_BANKROLL = 1000
RETRAIN_INTERVAL = 300

# Test/train split
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[(all_features['date'] >= test_start) & (all_features['home_spread'].notna())].copy()

print(f"Train: {len(train_data)} games")
print(f"Test: {len(test_data)} games")
print(f"\nLine movement: {LINE_MOVEMENT}")
print(f"Bet availability: {BET_AVAILABILITY*100}%")

# Train initial model
print("\nTraining spread model...")
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']

val_size = int(0.2 * len(train_data))
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

model = PointSpreadModel()
model.fit(X_tr, y_tr, X_val, y_val, feature_columns=feature_cols)
print(f"Model calibrated std: {model.calibrated_std:.2f}")

# Tracking
bankroll = STARTING_BANKROLL
peak_bankroll = STARTING_BANKROLL
max_drawdown = 0
bets = []
np.random.seed(42)

# Walk-forward backtest
games_processed = 0
last_retrain = 0

print("\nRunning walk-forward backtest...")

for idx, row in test_data.iterrows():
    games_processed += 1

    # Retrain periodically
    if (games_processed - last_retrain) >= RETRAIN_INTERVAL:
        retrain_mask = all_features['date'] < row['date']
        retrain_data = all_features[retrain_mask]

        if len(retrain_data) >= 1500:
            X_rt = retrain_data[feature_cols].fillna(0)
            y_rt = retrain_data['point_diff']

            val_size = int(0.2 * len(retrain_data))
            X_tr, X_val = X_rt[:-val_size], X_rt[-val_size:]
            y_tr, y_val = y_rt[:-val_size], y_rt[-val_size:]

            model = PointSpreadModel()
            model.fit(X_tr, y_tr, X_val, y_val, feature_columns=feature_cols)
            last_retrain = games_processed

    # Get prediction
    X_game = pd.DataFrame([row[feature_cols].fillna(0)])
    pred_diff = model.predict(X_game)[0]
    spread = row['home_spread']
    model_edge = pred_diff + spread  # Positive = bet home

    # Calculate actual result
    actual_diff = row['home_score'] - row['away_score']

    # Check betting criteria
    should_bet = False
    bet_side = None

    # Strategy 1: Large model edge (5+ points)
    if abs(model_edge) >= 5:
        should_bet = True
        bet_side = 'home' if model_edge >= 5 else 'away'

    # Strategy 2: Model + rest advantage (model >= 1 and rest diff >= 2)
    elif model_edge >= 1 and row['rest_diff'] >= 2:
        should_bet = True
        bet_side = 'home'
    elif model_edge <= -1 and row['rest_diff'] <= -2:
        should_bet = True
        bet_side = 'away'

    # Strategy 3: Situational B2B (pure situational when model edge 0-5)
    elif row['home_b2b'] == 1 and row['away_rest'] >= 2 and abs(model_edge) < 5:
        should_bet = True
        bet_side = 'away'

    if not should_bet:
        continue

    # Simulate bet availability
    if np.random.random() > BET_AVAILABILITY:
        continue

    # Apply line movement
    if bet_side == 'home':
        adjusted_spread = spread - LINE_MOVEMENT
        won = actual_diff > -adjusted_spread
    else:
        adjusted_spread = spread + LINE_MOVEMENT
        won = actual_diff < -adjusted_spread

    # Calculate P&L
    pnl = BET_SIZE * 0.909 if won else -BET_SIZE

    bankroll += pnl
    peak_bankroll = max(peak_bankroll, bankroll)
    drawdown = (peak_bankroll - bankroll) / peak_bankroll
    max_drawdown = max(max_drawdown, drawdown)

    bets.append({
        'date': row['date'],
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'bet_side': bet_side,
        'strategy': 'large_edge' if abs(model_edge) >= 5 else
                   ('rest_adv' if (model_edge >= 1 and row['rest_diff'] >= 2) or (model_edge <= -1 and row['rest_diff'] <= -2) else 'b2b'),
        'model_edge': model_edge,
        'spread': spread,
        'adjusted_spread': adjusted_spread,
        'actual_diff': actual_diff,
        'won': won,
        'pnl': pnl,
        'bankroll': bankroll
    })

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

bets_df = pd.DataFrame(bets)
if len(bets_df) == 0:
    print("No bets placed!")
else:
    wins = bets_df['won'].sum()
    losses = len(bets_df) - wins
    total_pnl = bets_df['pnl'].sum()

    print(f"Total bets: {len(bets_df)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {wins/len(bets_df)*100:.1f}%")
    print()
    print(f"Starting bankroll: ${STARTING_BANKROLL}")
    print(f"Final bankroll: ${bankroll:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"ROI: {total_pnl/(len(bets_df)*BET_SIZE)*100:.1f}%")
    print(f"Max drawdown: {max_drawdown*100:.1f}%")

    # Break-even analysis
    breakeven = 0.524
    print(f"\nBreak-even: {breakeven*100:.1f}%")
    print(f"Edge over break-even: {(wins/len(bets_df) - breakeven)*100:.1f}%")

    # Stats
    z = (wins/len(bets_df) - 0.50) / np.sqrt(0.25 / len(bets_df))
    p = 1 - stats.norm.cdf(z)
    print(f"Z-score vs 50%: {z:.2f}")
    print(f"P-value: {p:.4f}")

    # By strategy
    print("\n" + "-" * 40)
    print("By Strategy:")
    print("-" * 40)
    for strat in bets_df['strategy'].unique():
        strat_bets = bets_df[bets_df['strategy'] == strat]
        sw = strat_bets['won'].sum()
        sl = len(strat_bets) - sw
        swr = sw / len(strat_bets) * 100
        sroi = strat_bets['pnl'].sum() / (len(strat_bets) * BET_SIZE) * 100
        print(f"  {strat}: {sw}W-{sl}L ({swr:.1f}%) ROI: {sroi:.1f}%")

    # By year
    print("\n" + "-" * 40)
    print("By Year:")
    print("-" * 40)
    bets_df['year'] = bets_df['date'].dt.year
    for year in sorted(bets_df['year'].unique()):
        year_bets = bets_df[bets_df['year'] == year]
        yw = year_bets['won'].sum()
        yl = len(year_bets) - yw
        ywr = yw / len(year_bets) * 100
        yroi = year_bets['pnl'].sum() / (len(year_bets) * BET_SIZE) * 100
        print(f"  {year}: {yw}W-{yl}L ({ywr:.1f}%) ROI: {yroi:.1f}%")

    # By bet side
    print("\n" + "-" * 40)
    print("By Bet Side:")
    print("-" * 40)
    for side in ['home', 'away']:
        side_bets = bets_df[bets_df['bet_side'] == side]
        if len(side_bets) > 0:
            sw = side_bets['won'].sum()
            sl = len(side_bets) - sw
            swr = sw / len(side_bets) * 100
            sroi = side_bets['pnl'].sum() / (len(side_bets) * BET_SIZE) * 100
            print(f"  {side.upper()}: {sw}W-{sl}L ({swr:.1f}%) ROI: {sroi:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("EDGE ASSESSMENT")
    print("=" * 70)
    if total_pnl > 0:
        print(f"PROFITABLE: +${total_pnl:.2f} ({total_pnl/(len(bets_df)*BET_SIZE)*100:.1f}% ROI)")
        if p < 0.05:
            print("Statistically significant at 95% confidence!")
        elif p < 0.10:
            print("Statistically significant at 90% confidence.")
        else:
            print("Not statistically significant - could be variance.")
    else:
        print(f"NOT PROFITABLE: -${abs(total_pnl):.2f}")
        print("Combined strategy does not produce edge with realistic conditions.")
