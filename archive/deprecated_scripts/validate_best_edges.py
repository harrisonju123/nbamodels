"""
Validate Best Edges - Year-by-year validation of promising strategies

Key findings from find_edge_v2:
1. Model edge 3+ & injuries <= 1: 54.3%, +3.7% ROI, p=0.048
2. Model edge 3+ & injuries <= 2: 53.4%, +2.0% ROI, p=0.047
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

TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("VALIDATING BEST EDGE STRATEGIES")
print("=" * 70)

# Load data
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')
box_scores = pd.read_parquet('data/cache/player_box_scores.parquet')

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
    lambda r: -r['spread'] if r['whos_favored'] == 'home' else r['spread'], axis=1
)

merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'home_spread']],
    on=['date', 'home_team', 'away_team'], how='left'
).sort_values('date').reset_index(drop=True)

# Parse minutes
def parse_minutes(min_str):
    if pd.isna(min_str) or min_str == '':
        return 0.0
    try:
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except:
        return 0.0

box_scores['min'] = box_scores['min'].apply(parse_minutes)

# Calculate star players
player_stats = box_scores.groupby(['player_name', 'team_abbreviation']).agg({
    'pts': 'mean', 'min': 'mean', 'game_id': 'count'
}).reset_index()
player_stats.columns = ['player_name', 'team', 'avg_pts', 'avg_min', 'games_played']
player_stats = player_stats[player_stats['games_played'] >= 20]
player_stats['impact_score'] = player_stats['avg_pts'] + player_stats['avg_min'] * 0.3

stars = player_stats[player_stats['impact_score'] >= 30].copy()
star_lookup = stars.groupby('team')['player_name'].apply(list).to_dict()

# Get injury counts per game
games_played = box_scores.groupby(['game_id', 'team_abbreviation'])['player_name'].apply(set).reset_index()
games_played.columns = ['game_id', 'team', 'players_played']

def count_missing_stars(team, players_played):
    if team not in star_lookup:
        return 0
    return len(set(star_lookup[team]) - players_played)

injury_data = []
for _, row in games_played.iterrows():
    n_missing = count_missing_stars(row['team'], row['players_played'])
    injury_data.append({'game_id': row['game_id'], 'team': row['team'], 'n_missing': n_missing})

injury_df = pd.DataFrame(injury_data)
injury_lookup = injury_df.set_index(['game_id', 'team'])['n_missing'].to_dict()

box_game_ids = set(box_scores['game_id'].unique())
matched = merged[merged['game_id'].isin(box_game_ids)].copy()

# Add injury counts
home_missing = []
away_missing = []
for _, row in matched.iterrows():
    h_miss = injury_lookup.get((row['game_id'], row['home_team']), 0)
    a_miss = injury_lookup.get((row['game_id'], row['away_team']), 0)
    home_missing.append(h_miss)
    away_missing.append(a_miss)

matched['home_missing'] = home_missing
matched['away_missing'] = away_missing
matched['total_injuries'] = matched['home_missing'] + matched['away_missing']

# Build features
print("\nBuilding features...")
builder = GameFeatureBuilder()
all_features = builder.build_game_features(merged)

feature_cols = [c for c in builder.get_feature_columns(all_features)
                if c != 'home_spread' and 'OREB_PCT' not in c and 'EFG_PCT' not in c]

all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'home_score', 'away_score']].drop_duplicates(),
    on='game_id', how='left', suffixes=('', '_m')
)
all_features['actual_diff'] = all_features['home_score'] - all_features['away_score']

# Add injury data
injury_info = matched[['game_id', 'total_injuries', 'home_missing', 'away_missing']].drop_duplicates()
all_features = all_features.merge(injury_info, on='game_id', how='left')
all_features['total_injuries'] = all_features['total_injuries'].fillna(0)

# Train/test split
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[
    (all_features['date'] >= test_start) &
    (all_features['home_spread'].notna()) &
    (all_features['game_id'].isin(box_game_ids))
].copy()

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Train model
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']
val_size = int(0.2 * len(train_data))

model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
          X_train[-val_size:], y_train[-val_size:],
          feature_columns=feature_cols)

# Predict
X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']
test_data['year'] = test_data['date'].dt.year

print("\n" + "=" * 70)
print("STRATEGY 1: Model edge 3+ & injuries <= 1 (bet HOME)")
print("=" * 70)

print("\nYear-by-year breakdown:")
total_bets = 0
total_wins = 0
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    mask = (year_data['model_edge'] >= 3) & (year_data['total_injuries'] <= 1)
    games = year_data[mask]
    if len(games) >= 10:
        wins = games['home_covers'].sum()
        n = len(games)
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        total_bets += n
        total_wins += wins
        print(f"  {year}: {wr:.1f}% ATS (n={n}, wins={wins}, ROI={roi:+.1f}%)")
    else:
        print(f"  {year}: Too few games (n={len(games)})")

if total_bets > 0:
    overall_wr = total_wins / total_bets * 100
    overall_roi = (total_wins/total_bets * 0.909 - (1 - total_wins/total_bets)) * 100
    print(f"\n  OVERALL: {overall_wr:.1f}% (n={total_bets}, wins={total_wins}, ROI={overall_roi:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 2: Model edge 3+ & injuries <= 2 (bet HOME)")
print("=" * 70)

print("\nYear-by-year breakdown:")
total_bets = 0
total_wins = 0
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    mask = (year_data['model_edge'] >= 3) & (year_data['total_injuries'] <= 2)
    games = year_data[mask]
    if len(games) >= 10:
        wins = games['home_covers'].sum()
        n = len(games)
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        total_bets += n
        total_wins += wins
        print(f"  {year}: {wr:.1f}% ATS (n={n}, wins={wins}, ROI={roi:+.1f}%)")

if total_bets > 0:
    overall_wr = total_wins / total_bets * 100
    overall_roi = (total_wins/total_bets * 0.909 - (1 - total_wins/total_bets)) * 100
    print(f"\n  OVERALL: {overall_wr:.1f}% (n={total_bets}, wins={total_wins}, ROI={overall_roi:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 3: Model edge >= 5 (no injury filter)")
print("=" * 70)

print("\nYear-by-year breakdown:")
total_bets = 0
total_wins = 0
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    mask = year_data['model_edge'] >= 5
    games = year_data[mask]
    if len(games) >= 10:
        wins = games['home_covers'].sum()
        n = len(games)
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        total_bets += n
        total_wins += wins
        print(f"  {year}: {wr:.1f}% ATS (n={n}, wins={wins}, ROI={roi:+.1f}%)")

if total_bets > 0:
    overall_wr = total_wins / total_bets * 100
    overall_roi = (total_wins/total_bets * 0.909 - (1 - total_wins/total_bets)) * 100
    print(f"\n  OVERALL: {overall_wr:.1f}% (n={total_bets}, wins={total_wins}, ROI={overall_roi:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 4: Model edge <= -5 (bet AWAY)")
print("=" * 70)

print("\nYear-by-year breakdown:")
total_bets = 0
total_wins = 0
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    mask = year_data['model_edge'] <= -5
    games = year_data[mask]
    if len(games) >= 10:
        wins = (~games['home_covers']).sum()
        n = len(games)
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        total_bets += n
        total_wins += wins
        print(f"  {year}: {wr:.1f}% ATS (n={n}, wins={wins}, ROI={roi:+.1f}%)")

if total_bets > 0:
    overall_wr = total_wins / total_bets * 100
    overall_roi = (total_wins/total_bets * 0.909 - (1 - total_wins/total_bets)) * 100
    print(f"\n  OVERALL: {overall_wr:.1f}% (n={total_bets}, wins={total_wins}, ROI={overall_roi:+.1f}%)")

print("\n" + "=" * 70)
print("COMBINED STRATEGY: Both directions with 5+ edge")
print("=" * 70)

print("\nYear-by-year breakdown:")
total_bets = 0
total_wins = 0
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]

    # Bet home when model edge >= 5
    home_mask = year_data['model_edge'] >= 5
    home_games = year_data[home_mask]
    home_wins = home_games['home_covers'].sum() if len(home_games) > 0 else 0

    # Bet away when model edge <= -5
    away_mask = year_data['model_edge'] <= -5
    away_games = year_data[away_mask]
    away_wins = (~away_games['home_covers']).sum() if len(away_games) > 0 else 0

    n = len(home_games) + len(away_games)
    wins = home_wins + away_wins
    if n >= 20:
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        total_bets += n
        total_wins += wins
        print(f"  {year}: {wr:.1f}% ATS (n={n}, home={len(home_games)}, away={len(away_games)}, ROI={roi:+.1f}%)")

if total_bets > 0:
    overall_wr = total_wins / total_bets * 100
    overall_roi = (total_wins/total_bets * 0.909 - (1 - total_wins/total_bets)) * 100
    z = (overall_wr/100 - 0.50) / np.sqrt(0.25 / total_bets)
    p = 1 - stats.norm.cdf(z) if z > 0 else stats.norm.cdf(z)
    print(f"\n  OVERALL: {overall_wr:.1f}% (n={total_bets}, wins={total_wins}, ROI={overall_roi:+.1f}%, z={z:.2f}, p={abs(p):.4f})")

print("\n" + "=" * 70)
print("REALISTIC SIMULATION (with line movement & availability)")
print("=" * 70)

# Parameters
LINE_MOVEMENT = 0.5
BET_AVAILABILITY = 0.85
np.random.seed(42)

print(f"\nLine movement: {LINE_MOVEMENT} pts")
print(f"Bet availability: {BET_AVAILABILITY*100}%")

bets = []
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]

    for _, row in year_data.iterrows():
        if abs(row['model_edge']) < 5:
            continue

        # Availability check
        if np.random.random() > BET_AVAILABILITY:
            continue

        bet_home = row['model_edge'] >= 5
        spread = row['home_spread']
        actual_diff = row['actual_diff']

        # Apply line movement against us
        if bet_home:
            adj_spread = spread - LINE_MOVEMENT
            won = actual_diff > -adj_spread
        else:
            adj_spread = spread + LINE_MOVEMENT
            won = actual_diff < -adj_spread

        bets.append({
            'year': year,
            'bet_home': bet_home,
            'won': won,
            'edge': row['model_edge']
        })

bets_df = pd.DataFrame(bets)
if len(bets_df) > 0:
    print(f"\nTotal bets after filters: {len(bets_df)}")
    print("\nYear-by-year with realistic conditions:")
    for year in sorted(bets_df['year'].unique()):
        year_bets = bets_df[bets_df['year'] == year]
        wins = year_bets['won'].sum()
        n = len(year_bets)
        wr = wins / n * 100
        roi = (wins/n * 0.909 - (1 - wins/n)) * 100
        print(f"  {year}: {wr:.1f}% (n={n}, ROI={roi:+.1f}%)")

    total_wins = bets_df['won'].sum()
    total_n = len(bets_df)
    overall_wr = total_wins / total_n * 100
    overall_roi = (total_wins/total_n * 0.909 - (1 - total_wins/total_n)) * 100
    print(f"\n  REALISTIC OVERALL: {overall_wr:.1f}% (n={total_n}, ROI={overall_roi:+.1f}%)")

    if overall_roi > 0:
        print("\n  ** EDGE EXISTS UNDER REALISTIC CONDITIONS **")
    else:
        print("\n  No edge after realistic conditions applied")
