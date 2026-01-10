"""
Team-Specific ATS Analysis

Find which teams are consistently mispriced by the market:
1. Teams to BACK when model has positive edge
2. Teams to FADE when model has negative edge
3. Teams where our model outperforms/underperforms
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
from src.models.spread_model import SpreadPredictionModel
from scipy import stats

# Skip Four Factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("TEAM-SPECIFIC ATS ANALYSIS")
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

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Train model
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['point_diff']
val_size = int(0.2 * len(train_data))

model = PointSpreadModel()
model.fit(X_train[:-val_size], y_train[:-val_size],
         X_train[-val_size:], y_train[-val_size:],
         feature_columns=feature_cols)

# Get predictions
X_test = test_data[feature_cols].fillna(0)
test_data['pred_diff'] = model.predict(X_test)
test_data['model_edge'] = test_data['pred_diff'] + test_data['home_spread']
test_data['home_covers'] = test_data['actual_diff'] > -test_data['home_spread']
test_data['away_covers'] = test_data['actual_diff'] < -test_data['home_spread']


def calc_ats_stats(wins, n):
    """Calculate ATS statistics."""
    if n < 10:
        return None
    win_rate = wins / n
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    z = (win_rate - 0.50) / np.sqrt(0.25 / n)
    p = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed
    return {'n': n, 'wins': wins, 'win_rate': win_rate, 'roi': roi, 'p_value': p}


print("\n" + "=" * 70)
print("1. OVERALL TEAM ATS PERFORMANCE (2022-2025)")
print("=" * 70)

all_teams = sorted(set(test_data['home_team'].unique()) | set(test_data['away_team'].unique()))

team_overall = []
for team in all_teams:
    # As home team
    home_games = test_data[test_data['home_team'] == team]
    home_covers = home_games['home_covers'].sum()
    home_n = len(home_games)

    # As away team
    away_games = test_data[test_data['away_team'] == team]
    away_covers = away_games['away_covers'].sum()
    away_n = len(away_games)

    total_covers = home_covers + away_covers
    total_n = home_n + away_n

    if total_n >= 50:
        stats_dict = calc_ats_stats(total_covers, total_n)
        if stats_dict:
            team_overall.append({
                'team': team,
                **stats_dict,
                'home_n': home_n,
                'away_n': away_n,
            })

team_df = pd.DataFrame(team_overall).sort_values('win_rate', ascending=False)

print("\n--- Best ATS Teams ---")
for _, row in team_df.head(10).iterrows():
    sig = "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")

print("\n--- Worst ATS Teams ---")
for _, row in team_df.tail(10).iterrows():
    sig = "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")


print("\n" + "=" * 70)
print("2. TEAM PERFORMANCE WHEN MODEL HAS EDGE 5+ ON THEM")
print("=" * 70)
print("(Model says bet ON this team)")

team_with_edge = []
for team in all_teams:
    # Home games where model edge >= 5 (bet home = bet on this team)
    home_edge = test_data[(test_data['home_team'] == team) & (test_data['model_edge'] >= 5)]
    home_wins = home_edge['home_covers'].sum()
    home_n = len(home_edge)

    # Away games where model edge <= -5 (bet away = bet on this team)
    away_edge = test_data[(test_data['away_team'] == team) & (test_data['model_edge'] <= -5)]
    away_wins = away_edge['away_covers'].sum()
    away_n = len(away_edge)

    total_wins = home_wins + away_wins
    total_n = home_n + away_n

    if total_n >= 20:
        stats_dict = calc_ats_stats(total_wins, total_n)
        if stats_dict:
            team_with_edge.append({
                'team': team,
                **stats_dict,
            })

edge_df = pd.DataFrame(team_with_edge).sort_values('win_rate', ascending=False)

print("\n--- Best Teams to BACK when model has edge (Edge 5+) ---")
for _, row in edge_df.head(10).iterrows():
    sig = "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")

print("\n--- Teams to AVOID even when model has edge ---")
for _, row in edge_df.tail(10).iterrows():
    sig = "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")


print("\n" + "=" * 70)
print("3. TEAM PERFORMANCE WHEN MODEL HAS EDGE AGAINST THEM")
print("=" * 70)
print("(Model says bet AGAINST this team)")

team_against = []
for team in all_teams:
    # Home games where model edge <= -5 (bet away = bet against home team)
    home_against = test_data[(test_data['home_team'] == team) & (test_data['model_edge'] <= -5)]
    # Model says bet away, so we check if away covers
    against_wins = home_against['away_covers'].sum()
    home_n = len(home_against)

    # Away games where model edge >= 5 (bet home = bet against away team)
    away_against = test_data[(test_data['away_team'] == team) & (test_data['model_edge'] >= 5)]
    # Model says bet home, so we check if home covers
    against_wins += away_against['home_covers'].sum()
    away_n = len(away_against)

    total_n = home_n + away_n

    if total_n >= 20:
        stats_dict = calc_ats_stats(against_wins, total_n)
        if stats_dict:
            team_against.append({
                'team': team,
                **stats_dict,
            })

against_df = pd.DataFrame(team_against).sort_values('win_rate', ascending=False)

print("\n--- Best Teams to FADE when model signals against ---")
for _, row in against_df.head(10).iterrows():
    sig = "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
    print(f"  FADE {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")

print("\n--- Teams that cover even when model is against them ---")
for _, row in against_df.tail(10).iterrows():
    sig = "*" if row['p_value'] < 0.10 else ""
    print(f"  DON'T FADE {row['team']}: {row['win_rate']*100:.1f}% ATS (n={row['n']}, ROI={row['roi']:+.1f}%) {sig}")


print("\n" + "=" * 70)
print("4. MODEL EDGE ACCURACY BY TEAM")
print("=" * 70)
print("(How well does model predict spreads for each team)")

team_accuracy = []
for team in all_teams:
    # Home games
    home_games = test_data[test_data['home_team'] == team]
    # Away games
    away_games = test_data[test_data['away_team'] == team]

    # Model error when this team is home
    home_errors = (home_games['pred_diff'] - home_games['actual_diff']).abs()
    # Model error when this team is away (flip sign since pred_diff is home perspective)
    away_errors = ((-away_games['pred_diff']) - (away_games['away_score'] - away_games['home_score'])).abs()

    all_errors = pd.concat([home_errors, away_errors])

    if len(all_errors) >= 50:
        team_accuracy.append({
            'team': team,
            'mae': all_errors.mean(),
            'n': len(all_errors),
        })

accuracy_df = pd.DataFrame(team_accuracy).sort_values('mae')

print("\n--- Teams model predicts BEST (lowest MAE) ---")
for _, row in accuracy_df.head(10).iterrows():
    print(f"  {row['team']}: MAE {row['mae']:.2f} pts (n={row['n']})")

print("\n--- Teams model predicts WORST (highest MAE) ---")
for _, row in accuracy_df.tail(10).iterrows():
    print(f"  {row['team']}: MAE {row['mae']:.2f} pts (n={row['n']})")


print("\n" + "=" * 70)
print("5. YEAR-OVER-YEAR CONSISTENCY")
print("=" * 70)

test_data['year'] = test_data['date'].dt.year

print("\n--- Teams consistently good ATS (above 52% each year) ---")
consistent_good = []
for team in all_teams:
    yearly_rates = []
    for year in [2022, 2023, 2024, 2025]:
        year_data = test_data[test_data['year'] == year]
        home = year_data[(year_data['home_team'] == team)]
        away = year_data[(year_data['away_team'] == team)]
        covers = home['home_covers'].sum() + away['away_covers'].sum()
        n = len(home) + len(away)
        if n >= 20:
            yearly_rates.append(covers / n)

    if len(yearly_rates) >= 3 and all(r > 0.52 for r in yearly_rates):
        avg_rate = np.mean(yearly_rates)
        consistent_good.append({'team': team, 'avg_rate': avg_rate, 'years': len(yearly_rates)})

for item in sorted(consistent_good, key=lambda x: -x['avg_rate']):
    print(f"  {item['team']}: avg {item['avg_rate']*100:.1f}% ATS ({item['years']} years)")

print("\n--- Teams consistently bad ATS (below 48% each year) ---")
consistent_bad = []
for team in all_teams:
    yearly_rates = []
    for year in [2022, 2023, 2024, 2025]:
        year_data = test_data[test_data['year'] == year]
        home = year_data[(year_data['home_team'] == team)]
        away = year_data[(year_data['away_team'] == team)]
        covers = home['home_covers'].sum() + away['away_covers'].sum()
        n = len(home) + len(away)
        if n >= 20:
            yearly_rates.append(covers / n)

    if len(yearly_rates) >= 3 and all(r < 0.48 for r in yearly_rates):
        avg_rate = np.mean(yearly_rates)
        consistent_bad.append({'team': team, 'avg_rate': avg_rate, 'years': len(yearly_rates)})

for item in sorted(consistent_bad, key=lambda x: x['avg_rate']):
    print(f"  {item['team']}: avg {item['avg_rate']*100:.1f}% ATS ({item['years']} years)")


print("\n" + "=" * 70)
print("6. COMBINED EDGE STRATEGY + TEAM FILTER")
print("=" * 70)

# Find teams that perform well with our edge strategy
print("\nTesting: Edge 5+ & No B2B + Team filters")

# Get top 5 and bottom 5 teams from edge analysis
if len(edge_df) >= 5:
    top_teams = set(edge_df.head(5)['team'])
    bottom_teams = set(edge_df.tail(5)['team'])

    # Strategy: Only bet when team is in top performers
    top_mask_home = (test_data['model_edge'] >= 5) & (~test_data['home_b2b'].fillna(False)) & (test_data['home_team'].isin(top_teams))
    top_mask_away = (test_data['model_edge'] <= -5) & (~test_data['away_b2b'].fillna(False)) & (test_data['away_team'].isin(top_teams))

    top_home_wins = test_data[top_mask_home]['home_covers'].sum()
    top_away_wins = test_data[top_mask_away]['away_covers'].sum()
    top_n = top_mask_home.sum() + top_mask_away.sum()

    if top_n >= 30:
        top_stats = calc_ats_stats(top_home_wins + top_away_wins, top_n)
        print(f"\nEdge 5+ & No B2B + TOP 5 teams only:")
        print(f"  Teams: {', '.join(sorted(top_teams))}")
        print(f"  {top_stats['win_rate']*100:.1f}% ATS (n={top_stats['n']}, ROI={top_stats['roi']:+.1f}%, p={top_stats['p_value']:.3f})")

    # Strategy: Exclude bottom performers
    excl_mask_home = (test_data['model_edge'] >= 5) & (~test_data['home_b2b'].fillna(False)) & (~test_data['home_team'].isin(bottom_teams))
    excl_mask_away = (test_data['model_edge'] <= -5) & (~test_data['away_b2b'].fillna(False)) & (~test_data['away_team'].isin(bottom_teams))

    excl_home_wins = test_data[excl_mask_home]['home_covers'].sum()
    excl_away_wins = test_data[excl_mask_away]['away_covers'].sum()
    excl_n = excl_mask_home.sum() + excl_mask_away.sum()

    if excl_n >= 30:
        excl_stats = calc_ats_stats(excl_home_wins + excl_away_wins, excl_n)
        print(f"\nEdge 5+ & No B2B + EXCLUDE bottom 5 teams:")
        print(f"  Excluded: {', '.join(sorted(bottom_teams))}")
        print(f"  {excl_stats['win_rate']*100:.1f}% ATS (n={excl_stats['n']}, ROI={excl_stats['roi']:+.1f}%, p={excl_stats['p_value']:.3f})")

# Compare to baseline
baseline_home = (test_data['model_edge'] >= 5) & (~test_data['home_b2b'].fillna(False))
baseline_away = (test_data['model_edge'] <= -5) & (~test_data['away_b2b'].fillna(False))
baseline_wins = test_data[baseline_home]['home_covers'].sum() + test_data[baseline_away]['away_covers'].sum()
baseline_n = baseline_home.sum() + baseline_away.sum()
baseline_stats = calc_ats_stats(baseline_wins, baseline_n)

print(f"\nBaseline (Edge 5+ & No B2B):")
print(f"  {baseline_stats['win_rate']*100:.1f}% ATS (n={baseline_stats['n']}, ROI={baseline_stats['roi']:+.1f}%, p={baseline_stats['p_value']:.3f})")


print("\n" + "=" * 70)
print("SUMMARY - ACTIONABLE INSIGHTS")
print("=" * 70)

# Find statistically significant findings
print("\nStatistically Significant Team Findings (p < 0.10):")

sig_back = edge_df[edge_df['p_value'] < 0.10].sort_values('win_rate', ascending=False)
if len(sig_back) > 0:
    print("\n  BACK these teams when model has edge:")
    for _, row in sig_back[sig_back['win_rate'] > 0.55].iterrows():
        print(f"    {row['team']}: {row['win_rate']*100:.1f}% (n={row['n']}, p={row['p_value']:.3f})")

sig_fade = against_df[against_df['p_value'] < 0.10].sort_values('win_rate', ascending=False)
if len(sig_fade) > 0:
    print("\n  FADE these teams when model signals against:")
    for _, row in sig_fade[sig_fade['win_rate'] > 0.55].iterrows():
        print(f"    {row['team']}: {row['win_rate']*100:.1f}% (n={row['n']}, p={row['p_value']:.3f})")

# Avoid warnings
sig_avoid = edge_df[edge_df['p_value'] < 0.10].sort_values('win_rate')
if len(sig_avoid) > 0:
    print("\n  AVOID betting on these teams even with edge:")
    for _, row in sig_avoid[sig_avoid['win_rate'] < 0.45].iterrows():
        print(f"    {row['team']}: {row['win_rate']*100:.1f}% (n={row['n']}, p={row['p_value']:.3f})")
