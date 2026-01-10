"""
Injury Hypothesis Analysis - Does injury news explain model vs line gap?

Hypothesis: The closing line incorporates late injury news that our model doesn't have.
If true:
1. Games with star player absences should show larger model-line disagreement
2. Model should perform BETTER on games with no major injuries
3. The "model edge" should disappear when star players are out

Using player box scores to identify who actually played (proxy for injuries).
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
print("INJURY HYPOTHESIS ANALYSIS")
print("Does late injury news explain why the line beats our model?")
print("=" * 70)

# Load data
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')
box_scores = pd.read_parquet('data/cache/player_box_scores.parquet')

print(f"\nPlayer box scores: {len(box_scores)} rows")
print(f"Columns: {list(box_scores.columns)}")

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

# Explore box scores structure
print("\n" + "=" * 70)
print("BOX SCORES DATA EXPLORATION")
print("=" * 70)

print(f"\nSample data:")
print(box_scores.head())

# Convert min to numeric (it's stored as string like "34:12")
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

print(f"\nData types:")
print(box_scores.dtypes)

# Check for game_id or date columns
if 'game_id' in box_scores.columns:
    print(f"\nUnique games: {box_scores['game_id'].nunique()}")

# Identify star players based on average stats
print("\n" + "=" * 70)
print("IDENTIFYING STAR PLAYERS")
print("=" * 70)

# Calculate player averages
player_stats = box_scores.groupby(['player_name', 'team_abbreviation']).agg({
    'pts': 'mean',
    'min': 'mean',
    'game_id': 'count'
}).reset_index()
player_stats.columns = ['player_name', 'team', 'avg_pts', 'avg_min', 'games_played']

# Filter to players with enough games
player_stats = player_stats[player_stats['games_played'] >= 20]

# Star players: avg 20+ ppg or 30+ minutes
stars = player_stats[
    (player_stats['avg_pts'] >= 20) |
    (player_stats['avg_min'] >= 32)
].copy()
stars['impact_score'] = stars['avg_pts'] + stars['avg_min'] * 0.3

print(f"\nIdentified {len(stars)} star players (20+ PPG or 32+ MPG)")
print("\nTop 20 by impact score:")
print(stars.nlargest(20, 'impact_score')[['player_name', 'team', 'avg_pts', 'avg_min', 'games_played', 'impact_score']])

# Create star player lookup by team
star_lookup = stars.groupby('team')['player_name'].apply(list).to_dict()

print("\nStar players by team:")
for team, players in sorted(star_lookup.items()):
    print(f"  {team}: {', '.join(players[:3])}{'...' if len(players) > 3 else ''}")

# For each game, check which stars played
print("\n" + "=" * 70)
print("CHECKING STAR PLAYER ABSENCES BY GAME")
print("=" * 70)

# Get players who played in each game
games_played = box_scores.groupby(['game_id', 'team_abbreviation'])['player_name'].apply(set).reset_index()
games_played.columns = ['game_id', 'team', 'players_played']

# Function to count missing stars
def count_missing_stars(team, players_played):
    if team not in star_lookup:
        return 0, []
    team_stars = set(star_lookup[team])
    missing = team_stars - players_played
    return len(missing), list(missing)

print(f"\nUnique games in box_scores: {box_scores['game_id'].nunique()}")

# Build missing star info per team per game
team_game_missing = []
for _, row in games_played.iterrows():
    n_missing, missing_names = count_missing_stars(row['team'], row['players_played'])
    team_game_missing.append({
        'game_id': row['game_id'],
        'team': row['team'],
        'n_missing_stars': n_missing,
        'missing_stars': missing_names
    })

missing_df = pd.DataFrame(team_game_missing)
print(f"\nTeam-game rows with any missing stars: {(missing_df['n_missing_stars'] > 0).sum()}")

if (missing_df['n_missing_stars'] > 0).sum() > 0:
    print("\nSample of games with missing stars:")
    print(missing_df[missing_df['n_missing_stars'] > 0].head(20))

# game_id can be directly matched (100% overlap verified)
box_game_ids = set(box_scores['game_id'].unique())
matched = merged[merged['game_id'].isin(box_game_ids)].copy()

print(f"\nMatched {len(matched)} games to box scores out of {len(merged)}")

# Now for matched games, calculate missing star impact
def get_team_missing_info(game_id, team):
    subset = missing_df[(missing_df['game_id'] == game_id) & (missing_df['team'] == team)]
    if len(subset) > 0:
        return subset.iloc[0]['n_missing_stars'], subset.iloc[0]['missing_stars']
    return 0, []

home_missing = []
away_missing = []
home_missing_names = []
away_missing_names = []

for _, row in matched.iterrows():
    h_miss, h_names = get_team_missing_info(row['game_id'], row['home_team'])
    a_miss, a_names = get_team_missing_info(row['game_id'], row['away_team'])
    home_missing.append(h_miss)
    away_missing.append(a_miss)
    home_missing_names.append(h_names)
    away_missing_names.append(a_names)

matched['home_missing_stars'] = home_missing
matched['away_missing_stars'] = away_missing
matched['home_missing_names'] = home_missing_names
matched['away_missing_names'] = away_missing_names
matched['total_missing_stars'] = matched['home_missing_stars'] + matched['away_missing_stars']
matched['injury_asymmetry'] = matched['home_missing_stars'] - matched['away_missing_stars']

print("\n" + "=" * 70)
print("INJURY ANALYSIS")
print("=" * 70)

print(f"\nGames with at least one missing star: {(matched['total_missing_stars'] > 0).sum()}")
print(f"Games with multiple missing stars: {(matched['total_missing_stars'] > 1).sum()}")

# Distribution
print("\nMissing stars distribution:")
print(matched['total_missing_stars'].value_counts().sort_index().head(10))

# Build features on ALL games (need historical for training)
print("\n" + "=" * 70)
print("BUILDING MODEL TO TEST HYPOTHESIS")
print("=" * 70)

# Build features on full merged dataset (not just matched)
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

# Train/test split - use test games that have box score injury data
test_start = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start].copy()
test_data = all_features[
    (all_features['date'] >= test_start) &
    (all_features['home_spread'].notna()) &
    (all_features['game_id'].isin(box_game_ids))
].copy()

print(f"Train: {len(train_data)}, Test: {len(test_data)} (games with box score data)")

# Train model on pre-2022 data
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']

val_size = int(0.2 * len(train_data))
model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
          X_train[-val_size:], y_train[-val_size:],
          feature_columns=feature_cols)

# Add injury data from matched dataframe
injury_data = matched[['game_id', 'home_missing_stars', 'away_missing_stars',
                        'total_missing_stars', 'injury_asymmetry']].drop_duplicates()
test_data = test_data.merge(injury_data, on='game_id', how='left')

# Predict on test
X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['model_error'] = test_data['pred_diff'] - test_data['actual_diff']
test_data['line_error'] = -test_data['home_spread'] - test_data['actual_diff']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']

# Now test the hypothesis
print("\n" + "=" * 70)
print("HYPOTHESIS TEST 1: Model accuracy by injury presence")
print("=" * 70)

# Games with NO missing stars
no_injuries = test_data[test_data['total_missing_stars'] == 0]
has_injuries = test_data[test_data['total_missing_stars'] > 0]

print(f"\nGames with no missing stars: {len(no_injuries)}")
print(f"Games with missing stars: {len(has_injuries)}")

if len(no_injuries) > 30 and len(has_injuries) > 30:
    no_inj_model_mae = np.abs(no_injuries['model_error']).mean()
    no_inj_line_mae = np.abs(no_injuries['line_error']).mean()
    has_inj_model_mae = np.abs(has_injuries['model_error']).mean()
    has_inj_line_mae = np.abs(has_injuries['line_error']).mean()

    print(f"\nNO MISSING STARS:")
    print(f"  Model MAE: {no_inj_model_mae:.2f}")
    print(f"  Line MAE:  {no_inj_line_mae:.2f}")
    print(f"  Model vs Line: {no_inj_line_mae - no_inj_model_mae:+.2f} (positive = model better)")

    print(f"\nHAS MISSING STARS:")
    print(f"  Model MAE: {has_inj_model_mae:.2f}")
    print(f"  Line MAE:  {has_inj_line_mae:.2f}")
    print(f"  Model vs Line: {has_inj_line_mae - has_inj_model_mae:+.2f} (positive = model better)")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 2: ATS performance by injury presence")
print("=" * 70)

def analyze_ats(df, name):
    if len(df) < 30:
        return None
    # Bet with model (home if model_edge > 0, away otherwise)
    correct = ((df['model_edge'] > 0) & df['home_covers']) | ((df['model_edge'] < 0) & ~df['home_covers'])
    win_rate = correct.mean()
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    z = (win_rate - 0.50) / np.sqrt(0.25 / len(df))
    p = 1 - stats.norm.cdf(z) if z > 0 else stats.norm.cdf(z)

    print(f"{name}:")
    print(f"  Games: {len(df)}, Win rate: {win_rate*100:.1f}%, ROI: {roi:+.1f}%, p={abs(p):.3f}")
    return win_rate

# By injury count
for n in [0, 1, 2]:
    if n == 0:
        df = test_data[test_data['total_missing_stars'] == 0]
    elif n == 1:
        df = test_data[test_data['total_missing_stars'] == 1]
    else:
        df = test_data[test_data['total_missing_stars'] >= 2]

    if len(df) >= 30:
        analyze_ats(df, f"Missing stars = {n}{'+'  if n >= 2 else ''}")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 3: Model-Line disagreement by injury")
print("=" * 70)

# When model and line disagree strongly (5+ point edge), how do injuries factor in?
large_edge = test_data[np.abs(test_data['model_edge']) >= 5]
print(f"\nGames with large model edge (5+ pts): {len(large_edge)}")

if len(large_edge) > 30:
    large_no_inj = large_edge[large_edge['total_missing_stars'] == 0]
    large_has_inj = large_edge[large_edge['total_missing_stars'] > 0]

    print(f"  With no injuries: {len(large_no_inj)}")
    print(f"  With injuries: {len(large_has_inj)}")

    if len(large_no_inj) >= 20:
        analyze_ats(large_no_inj, "Large edge + NO injuries")
    if len(large_has_inj) >= 20:
        analyze_ats(large_has_inj, "Large edge + HAS injuries")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 4: Injury asymmetry impact")
print("=" * 70)

# Does injury asymmetry (one team has injuries, other doesn't) explain model failure?
asymmetric = test_data[test_data['injury_asymmetry'] != 0]
print(f"\nGames with asymmetric injuries: {len(asymmetric)}")

# Home team has more injuries (injury_asymmetry > 0)
home_hurt = test_data[test_data['injury_asymmetry'] > 0]
away_hurt = test_data[test_data['injury_asymmetry'] < 0]

print(f"  Home team more injured: {len(home_hurt)}")
print(f"  Away team more injured: {len(away_hurt)}")

# Does the line adjust for this better than our model?
if len(home_hurt) >= 30:
    print(f"\nWhen HOME has more injuries:")
    # Model should be too high on home (doesn't know about injury)
    # If line adjusts, line should be lower on home
    model_bias = home_hurt['model_edge'].mean()  # Model's perceived edge for home
    actual = home_hurt['actual_diff'].mean()
    line_implied = -home_hurt['home_spread'].mean()

    print(f"  Model predicted home edge: {home_hurt['pred_diff'].mean():.2f}")
    print(f"  Line implied home edge: {line_implied:.2f}")
    print(f"  Actual home diff: {actual:.2f}")
    print(f"  Model error: {home_hurt['pred_diff'].mean() - actual:.2f}")
    print(f"  Line error: {line_implied - actual:.2f}")

if len(away_hurt) >= 30:
    print(f"\nWhen AWAY has more injuries:")
    actual = away_hurt['actual_diff'].mean()
    line_implied = -away_hurt['home_spread'].mean()

    print(f"  Model predicted home edge: {away_hurt['pred_diff'].mean():.2f}")
    print(f"  Line implied home edge: {line_implied:.2f}")
    print(f"  Actual home diff: {actual:.2f}")
    print(f"  Model error: {away_hurt['pred_diff'].mean() - actual:.2f}")
    print(f"  Line error: {line_implied - actual:.2f}")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 5: Year-over-year injury data quality")
print("=" * 70)

# Check if we have injury data for recent years
test_data['year'] = test_data['date'].dt.year
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    games_with_data = (year_data['total_missing_stars'] >= 0).sum()
    pct_with_injuries = (year_data['total_missing_stars'] > 0).mean() * 100
    print(f"  {year}: {len(year_data)} games, {pct_with_injuries:.1f}% have missing stars")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Summarize findings
no_inj_wr = None
has_inj_wr = None

if len(no_injuries) >= 30:
    no_inj_correct = ((no_injuries['model_edge'] > 0) & no_injuries['home_covers']) | ((no_injuries['model_edge'] < 0) & ~no_injuries['home_covers'])
    no_inj_wr = no_inj_correct.mean()

if len(has_injuries) >= 30:
    has_inj_correct = ((has_injuries['model_edge'] > 0) & has_injuries['home_covers']) | ((has_injuries['model_edge'] < 0) & ~has_injuries['home_covers'])
    has_inj_wr = has_inj_correct.mean()

if no_inj_wr and has_inj_wr:
    print(f"\nModel ATS win rate:")
    print(f"  No missing stars: {no_inj_wr*100:.1f}%")
    print(f"  Has missing stars: {has_inj_wr*100:.1f}%")
    print(f"  Difference: {(no_inj_wr - has_inj_wr)*100:+.1f}%")

    if no_inj_wr > has_inj_wr + 0.02:
        print("\n** HYPOTHESIS SUPPORTED **")
        print("Model performs BETTER when there are no missing stars.")
        print("This suggests injuries (not in our model) affect line accuracy.")
    elif has_inj_wr > no_inj_wr + 0.02:
        print("\n** HYPOTHESIS REJECTED **")
        print("Model actually performs BETTER when there ARE injuries.")
        print("Injuries are not explaining the model-line gap.")
    else:
        print("\n** INCONCLUSIVE **")
        print("No significant difference in model performance by injury status.")
