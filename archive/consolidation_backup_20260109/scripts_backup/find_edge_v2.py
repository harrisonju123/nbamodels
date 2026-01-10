"""
Find Edge V2 - Incorporate injury insights into strategy

Based on findings:
1. Model performs best when no injuries (52.3% ATS)
2. Vegas line adjusts for injuries better than our model
3. Our model under-estimates injury impact by ~1.5 points

New approaches:
1. Add injury features to model
2. Filter to low-injury games
3. Bet against overreaction to star injuries
4. Combine best edges with injury info
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
print("EDGE FINDING V2 - Incorporating Injury Insights")
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

# Process odds
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

# Parse minutes from box scores
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

# Calculate player impact scores
print("\nCalculating player impact scores...")
player_stats = box_scores.groupby(['player_name', 'team_abbreviation']).agg({
    'pts': 'mean',
    'min': 'mean',
    'game_id': 'count'
}).reset_index()
player_stats.columns = ['player_name', 'team', 'avg_pts', 'avg_min', 'games_played']
player_stats = player_stats[player_stats['games_played'] >= 20]
player_stats['impact_score'] = player_stats['avg_pts'] + player_stats['avg_min'] * 0.3

# Star players (impact score > 30)
stars = player_stats[player_stats['impact_score'] >= 30].copy()
print(f"Identified {len(stars)} star players (impact >= 30)")

star_lookup = stars.groupby('team')['player_name'].apply(list).to_dict()
star_impact = stars.set_index(['team', 'player_name'])['impact_score'].to_dict()

# Get players who played in each game
games_played = box_scores.groupby(['game_id', 'team_abbreviation'])['player_name'].apply(set).reset_index()
games_played.columns = ['game_id', 'team', 'players_played']

# Calculate missing star impact per team per game
def calc_missing_impact(team, players_played):
    if team not in star_lookup:
        return 0.0, 0, []
    team_stars = set(star_lookup[team])
    missing = team_stars - players_played
    if not missing:
        return 0.0, 0, []
    total_impact = sum(star_impact.get((team, p), 0) for p in missing)
    return total_impact, len(missing), list(missing)

# Build injury info per game
print("Building injury features...")
team_game_injury = []
for _, row in games_played.iterrows():
    impact, n_missing, names = calc_missing_impact(row['team'], row['players_played'])
    team_game_injury.append({
        'game_id': row['game_id'],
        'team': row['team'],
        'missing_impact': impact,
        'n_missing': n_missing,
        'missing_names': names
    })

injury_df = pd.DataFrame(team_game_injury)

# Match to games
box_game_ids = set(box_scores['game_id'].unique())
matched = merged[merged['game_id'].isin(box_game_ids)].copy()

# Get home/away injury impact
home_injury = injury_df[injury_df.apply(lambda r: r['team'] ==
    matched[matched['game_id'] == r['game_id']]['home_team'].values[0]
    if r['game_id'] in matched['game_id'].values else False, axis=1) == False]

# Simpler approach - create lookup
injury_lookup = injury_df.set_index(['game_id', 'team']).to_dict('index')

home_impact = []
away_impact = []
home_n_missing = []
away_n_missing = []

for _, row in matched.iterrows():
    h_key = (row['game_id'], row['home_team'])
    a_key = (row['game_id'], row['away_team'])

    h_info = injury_lookup.get(h_key, {'missing_impact': 0, 'n_missing': 0})
    a_info = injury_lookup.get(a_key, {'missing_impact': 0, 'n_missing': 0})

    home_impact.append(h_info['missing_impact'])
    away_impact.append(a_info['missing_impact'])
    home_n_missing.append(h_info['n_missing'])
    away_n_missing.append(a_info['n_missing'])

matched['home_injury_impact'] = home_impact
matched['away_injury_impact'] = away_impact
matched['home_n_missing'] = home_n_missing
matched['away_n_missing'] = away_n_missing
matched['injury_impact_diff'] = matched['away_injury_impact'] - matched['home_injury_impact']
matched['total_injuries'] = matched['home_n_missing'] + matched['away_n_missing']

print(f"\nMatched {len(matched)} games with injury data")

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

# Add injury features for matched games
injury_features = matched[['game_id', 'home_injury_impact', 'away_injury_impact',
                           'injury_impact_diff', 'total_injuries',
                           'home_n_missing', 'away_n_missing']].drop_duplicates()
all_features = all_features.merge(injury_features, on='game_id', how='left')

# Fill NaN for games without injury data
all_features['injury_impact_diff'] = all_features['injury_impact_diff'].fillna(0)
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

# Train base model (no injury features)
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']

val_size = int(0.2 * len(train_data))
model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
          X_train[-val_size:], y_train[-val_size:],
          feature_columns=feature_cols)

# Predict on test
X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']

# Add rest features
def add_rest_features(df):
    df = df.sort_values('date').copy()
    team_last_game = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        game_date = row['date']

        h_rest = (game_date - team_last_game.get(ht, game_date - pd.Timedelta(days=7))).days
        a_rest = (game_date - team_last_game.get(at, game_date - pd.Timedelta(days=7))).days

        home_rest.append(h_rest)
        away_rest.append(a_rest)

        team_last_game[ht] = game_date
        team_last_game[at] = game_date

    df['home_rest'] = home_rest
    df['away_rest'] = away_rest
    df['rest_diff'] = df['home_rest'] - df['away_rest']
    df['home_b2b'] = (df['home_rest'] == 1).astype(int)
    df['away_b2b'] = (df['away_rest'] == 1).astype(int)
    return df

test_data = add_rest_features(test_data)


def analyze_strategy(df, name, bet_mask, bet_home):
    """Analyze a betting strategy."""
    games = df[bet_mask].copy()
    n = len(games)
    if n < 20:
        return None

    if bet_home:
        wins = games['home_covers'].sum()
    else:
        wins = (~games['home_covers']).sum()

    win_rate = wins / n
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    z = (win_rate - 0.50) / np.sqrt(0.25 / n)
    p = 1 - stats.norm.cdf(z) if z > 0 else stats.norm.cdf(z)

    return {
        'name': name,
        'n': n,
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi,
        'z_score': z,
        'p_value': abs(p)
    }


results = []

print("\n" + "=" * 70)
print("STRATEGY 1: Low Injury Games")
print("=" * 70)

# Only bet on games with 0-1 missing stars total
for max_injuries in [0, 1, 2]:
    mask = (test_data['total_injuries'] <= max_injuries) & (test_data['model_edge'] >= 3)
    r = analyze_strategy(test_data, f"Model edge 3+ & injuries <= {max_injuries} (bet HOME)", mask, bet_home=True)
    if r:
        results.append(r)
        print(f"  Edge 3+ & injuries <= {max_injuries}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

    mask = (test_data['total_injuries'] <= max_injuries) & (test_data['model_edge'] <= -3)
    r = analyze_strategy(test_data, f"Model edge -3+ & injuries <= {max_injuries} (bet AWAY)", mask, bet_home=False)
    if r:
        results.append(r)
        print(f"  Edge -3+ & injuries <= {max_injuries}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 2: Injury-Adjusted Model Edge")
print("=" * 70)

# Adjust model edge by injury impact difference
# If away has more injuries (positive impact_diff), add to home edge
INJURY_FACTOR = 0.05  # Points per impact score
test_data['adjusted_edge'] = test_data['model_edge'] + test_data['injury_impact_diff'] * INJURY_FACTOR

for edge in [3, 4, 5]:
    mask = test_data['adjusted_edge'] >= edge
    r = analyze_strategy(test_data, f"Injury-adjusted edge >= {edge} (bet HOME)", mask, bet_home=True)
    if r:
        results.append(r)
        print(f"  Adj edge >= {edge}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

    mask = test_data['adjusted_edge'] <= -edge
    r = analyze_strategy(test_data, f"Injury-adjusted edge <= -{edge} (bet AWAY)", mask, bet_home=False)
    if r:
        results.append(r)
        print(f"  Adj edge <= -{edge}: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 3: Rest + Model + Injury")
print("=" * 70)

# Combine rest advantage with model edge and injury info
mask = (test_data['model_edge'] >= 2) & (test_data['rest_diff'] >= 2) & (test_data['injury_impact_diff'] >= 0)
r = analyze_strategy(test_data, "Edge 2+ & rest 2+ & injury_diff >= 0", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Model 2+ & Rest 2+ & Injury helps home: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%, p={r['p_value']:.3f})")

mask = (test_data['model_edge'] <= -2) & (test_data['rest_diff'] <= -2) & (test_data['injury_impact_diff'] <= 0)
r = analyze_strategy(test_data, "Edge -2+ & rest -2+ & injury_diff <= 0", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Model -2+ & Rest -2+ & Injury helps away: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%, p={r['p_value']:.3f})")

print("\n" + "=" * 70)
print("STRATEGY 4: Fade Star Injury Overreaction")
print("=" * 70)

# When one team has high injury impact (big star out), line may overreact
# Bet AGAINST the adjustment (i.e., bet on team with injured star)
high_impact = 35  # One superstar out

mask = (test_data['home_injury_impact'] >= high_impact) & (test_data['away_injury_impact'] < 10)
r = analyze_strategy(test_data, f"Home has star out (impact {high_impact}+), bet HOME (fade overreaction)", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Home star out, bet HOME: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

mask = (test_data['away_injury_impact'] >= high_impact) & (test_data['home_injury_impact'] < 10)
r = analyze_strategy(test_data, f"Away has star out (impact {high_impact}+), bet AWAY (fade overreaction)", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Away star out, bet AWAY: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 5: Model Edge + B2B Situational")
print("=" * 70)

# Home B2B + Away rested + Model also likes away
mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2) & (test_data['model_edge'] <= 0)
r = analyze_strategy(test_data, "Home B2B + Away rested + Model <= 0", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B + Away rested 2+ + Model <= 0: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%, p={r['p_value']:.3f})")

mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2) & (test_data['model_edge'] <= -2)
r = analyze_strategy(test_data, "Home B2B + Away rested + Model <= -2", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B + Away rested 2+ + Model <= -2: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%, p={r['p_value']:.3f})")

print("\n" + "=" * 70)
print("STRATEGY 6: Large Spreads (Fade Favorites)")
print("=" * 70)

# Fade large home favorites
mask = test_data['home_spread'] <= -10
r = analyze_strategy(test_data, "Fade home -10+ (bet AWAY)", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Fade home -10+ (bet AWAY): {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

# Large favorites that model ALSO likes
mask = (test_data['home_spread'] <= -10) & (test_data['model_edge'] >= 0)
r = analyze_strategy(test_data, "Home -10+ but model likes home", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Home -10+ but model agrees: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

mask = (test_data['home_spread'] <= -10) & (test_data['model_edge'] <= -3)
r = analyze_strategy(test_data, "Home -10+ and model strongly disagrees", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Home -10+ and model disagrees 3+: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("STRATEGY 7: Conference/Division Games")
print("=" * 70)

# Define conferences
EAST = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
WEST = ['DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN', 'NOP', 'OKC', 'PHX', 'POR', 'SAC', 'SAS', 'UTA']

test_data['same_conf'] = test_data.apply(lambda r:
    (r['home_team'] in EAST and r['away_team'] in EAST) or
    (r['home_team'] in WEST and r['away_team'] in WEST), axis=1)

# Intra-conference games with model edge
mask = test_data['same_conf'] & (test_data['model_edge'] >= 4)
r = analyze_strategy(test_data, "Same conf + model 4+ (bet HOME)", mask, bet_home=True)
if r:
    results.append(r)
    print(f"Same conf + model 4+: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

mask = test_data['same_conf'] & (test_data['model_edge'] <= -4)
r = analyze_strategy(test_data, "Same conf + model -4+ (bet AWAY)", mask, bet_home=False)
if r:
    results.append(r)
    print(f"Same conf + model -4+: {r['win_rate']*100:.1f}% (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("SUMMARY - TOP STRATEGIES")
print("=" * 70)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    # Sort by ROI
    top = results_df[results_df['roi'] > 0].sort_values('roi', ascending=False)

    if len(top) > 0:
        print("\n--- PROFITABLE STRATEGIES (sorted by ROI) ---")
        for _, r in top.iterrows():
            sig = "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.10 else ""
            print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig:2s} {r['name']}")

    # Show statistically significant
    sig_results = results_df[results_df['p_value'] < 0.10].sort_values('roi', ascending=False)
    if len(sig_results) > 0:
        print("\n--- STATISTICALLY SIGNIFICANT (p < 0.10) ---")
        for _, r in sig_results.iterrows():
            print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {r['name']}")
    else:
        print("\nNo statistically significant strategies found at p < 0.10")

# Year-by-year for best strategy
print("\n" + "=" * 70)
print("YEAR-BY-YEAR VALIDATION")
print("=" * 70)

test_data['year'] = test_data['date'].dt.year

# Check B2B + rest + model strategy by year
print("\nHome B2B + Away rested 2+ + Model <= 0 by year:")
for year in sorted(test_data['year'].unique()):
    year_data = test_data[test_data['year'] == year]
    mask = (year_data['home_b2b'] == 1) & (year_data['away_rest'] >= 2) & (year_data['model_edge'] <= 0)
    games = year_data[mask]
    if len(games) >= 10:
        wins = (~games['home_covers']).sum()
        wr = wins / len(games) * 100
        roi = (wins/len(games) * 0.909 - (1 - wins/len(games))) * 100
        print(f"  {year}: {wr:.1f}% ATS (n={len(games)}, ROI={roi:+.1f}%)")
