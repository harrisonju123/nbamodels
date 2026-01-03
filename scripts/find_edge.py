"""
Find Edge - Analyze specific situations for betting edge

Looks at:
1. Back-to-back games
2. Rest advantage differentials
3. Altitude (Denver home)
4. Early season
5. Schedule spots
6. Model confidence levels
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
from scipy import stats

# Skip Four Factors (leakage)
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("EDGE ANALYSIS - Finding Situational Advantages")
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
).dropna(subset=['home_spread']).sort_values('date').reset_index(drop=True)

# Add schedule features
print("\nCalculating schedule features...")

# Calculate rest days for each team
def add_rest_features(df):
    df = df.sort_values('date').copy()

    # Track last game date for each team
    team_last_game = {}

    home_rest = []
    away_rest = []
    home_b2b = []
    away_b2b = []

    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        game_date = row['date']

        # Home team rest
        if ht in team_last_game:
            h_rest = (game_date - team_last_game[ht]).days
        else:
            h_rest = 7  # Default

        # Away team rest
        if at in team_last_game:
            a_rest = (game_date - team_last_game[at]).days
        else:
            a_rest = 7

        home_rest.append(h_rest)
        away_rest.append(a_rest)
        home_b2b.append(1 if h_rest == 1 else 0)
        away_b2b.append(1 if a_rest == 1 else 0)

        # Update last game
        team_last_game[ht] = game_date
        team_last_game[at] = game_date

    df['home_rest'] = home_rest
    df['away_rest'] = away_rest
    df['home_b2b'] = home_b2b
    df['away_b2b'] = away_b2b
    df['rest_diff'] = df['home_rest'] - df['away_rest']

    return df

merged = add_rest_features(merged)

# Add derived columns
merged['point_diff'] = merged['home_score'] - merged['away_score']
merged['home_covers'] = merged['point_diff'] > -merged['home_spread']
merged['away_covers'] = merged['point_diff'] < -merged['home_spread']
merged['month'] = merged['date'].dt.month
merged['is_denver_home'] = merged['home_team'] == 'DEN'
merged['is_utah_home'] = merged['home_team'] == 'UTA'

# Test period
test_start = pd.Timestamp('2020-01-01')
test_data = merged[merged['date'] >= test_start].copy()
print(f"\nTest period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
print(f"Total games with spreads: {len(test_data)}")


def analyze_situation(df, name, bet_home=True):
    """Analyze ATS performance for a situation."""
    if len(df) < 30:
        return None

    if bet_home:
        wins = df['home_covers'].sum()
    else:
        wins = df['away_covers'].sum()

    n = len(df)
    win_rate = wins / n
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100  # -110 odds

    # Statistical significance
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
print("SITUATIONAL ANALYSIS")
print("=" * 70)

results = []

# 1. Back-to-back situations
print("\n--- BACK-TO-BACK ANALYSIS ---")

# Home on B2B, away rested
home_b2b_away_rested = test_data[(test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 2)]
r = analyze_situation(home_b2b_away_rested, "Home B2B, Away 2+ rest (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Home B2B, Away 2+ rest (bet AWAY): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Away on B2B, home rested
away_b2b_home_rested = test_data[(test_data['away_b2b'] == 1) & (test_data['home_rest'] >= 2)]
r = analyze_situation(away_b2b_home_rested, "Away B2B, Home 2+ rest (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Away B2B, Home 2+ rest (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Both teams on B2B
both_b2b = test_data[(test_data['home_b2b'] == 1) & (test_data['away_b2b'] == 1)]
r = analyze_situation(both_b2b, "Both B2B (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Both B2B (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# 2. Rest advantage
print("\n--- REST ADVANTAGE ANALYSIS ---")

# Large home rest advantage
home_rest_adv = test_data[test_data['rest_diff'] >= 3]
r = analyze_situation(home_rest_adv, "Home 3+ rest advantage (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Home 3+ rest advantage (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Large away rest advantage
away_rest_adv = test_data[test_data['rest_diff'] <= -3]
r = analyze_situation(away_rest_adv, "Away 3+ rest advantage (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Away 3+ rest advantage (bet AWAY): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# 3. Altitude advantage
print("\n--- ALTITUDE ANALYSIS ---")

denver_home = test_data[test_data['is_denver_home'] == True]
r = analyze_situation(denver_home, "Denver home (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Denver home (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

utah_home = test_data[test_data['is_utah_home'] == True]
r = analyze_situation(utah_home, "Utah home (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Utah home (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Denver home + opponent on B2B
denver_opp_b2b = test_data[(test_data['is_denver_home'] == True) & (test_data['away_b2b'] == 1)]
r = analyze_situation(denver_opp_b2b, "Denver home + opp B2B (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Denver home + opp B2B (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# 4. Early season
print("\n--- EARLY SEASON ANALYSIS ---")

early_season = test_data[test_data['month'].isin([10, 11])]
r = analyze_situation(early_season, "Oct-Nov (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Oct-Nov games (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

late_season = test_data[test_data['month'].isin([3, 4])]
r = analyze_situation(late_season, "Mar-Apr (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Mar-Apr games (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# 5. Spread value spots
print("\n--- SPREAD SIZE ANALYSIS ---")

# Big favorites
big_fav = test_data[test_data['home_spread'] <= -10]
r = analyze_situation(big_fav, "Home -10+ (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Home -10+ (bet AWAY/dog): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Big underdogs
big_dog = test_data[test_data['home_spread'] >= 10]
r = analyze_situation(big_dog, "Home +10+ (bet HOME/dog)", bet_home=True)
if r:
    results.append(r)
    print(f"Home +10+ (bet HOME/dog): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Close games
close_games = test_data[(test_data['home_spread'] >= -3) & (test_data['home_spread'] <= 3)]
r = analyze_situation(close_games, "Pick'em to 3 (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Pick'em to 3 (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# 6. Combination spots
print("\n--- COMBINATION SPOTS ---")

# Away B2B + road dog
away_b2b_dog = test_data[(test_data['away_b2b'] == 1) & (test_data['home_spread'] <= -3)]
r = analyze_situation(away_b2b_dog, "Away B2B + road dog (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Away B2B + road dog (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")

# Home rested (3+) + underdog
home_rested_dog = test_data[(test_data['home_rest'] >= 3) & (test_data['home_spread'] >= 3)]
r = analyze_situation(home_rested_dog, "Home 3+ rest + underdog (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Home 3+ rest + underdog (bet HOME): {r['n']} games, {r['win_rate']*100:.1f}% ATS, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")


# Summary
print("\n" + "=" * 70)
print("SUMMARY - TOP EDGES (by ROI, min p<0.20)")
print("=" * 70)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    # Filter for potential edges
    potential_edges = results_df[
        (results_df['roi'] > 0) &
        (results_df['p_value'] < 0.20) &
        (results_df['n'] >= 50)
    ].sort_values('roi', ascending=False)

    if len(potential_edges) > 0:
        print("\nPotential edges found:")
        for _, r in potential_edges.iterrows():
            print(f"  {r['name']}")
            print(f"    {r['n']} games, {r['win_rate']*100:.1f}% win, {r['roi']:.1f}% ROI, p={r['p_value']:.3f}")
            print()
    else:
        print("\nNo statistically significant edges found with p<0.20 and positive ROI")

    # Show all results sorted by ROI
    print("\n--- ALL SITUATIONS (sorted by ROI) ---")
    for _, r in results_df.sort_values('roi', ascending=False).iterrows():
        sig = "*" if r['p_value'] < 0.10 else ""
        print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig} {r['name']}")

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)
overall = analyze_situation(test_data, "All games (bet HOME)", bet_home=True)
print(f"Overall home ATS: {overall['win_rate']*100:.1f}% ({overall['n']} games)")
overall_away = analyze_situation(test_data, "All games (bet AWAY)", bet_home=False)
print(f"Overall away ATS: {overall_away['win_rate']*100:.1f}% ({overall_away['n']} games)")
