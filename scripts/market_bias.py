"""
Market Bias Analysis - Find where the market systematically over/under-reacts

Instead of trying to predict better than Vegas, find situations where
Vegas has known biases:
1. Fade large favorites (public loves betting favorites)
2. Divisional games (familiarity factor)
3. Revenge games
4. Home underdogs
5. Specific spread ranges
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

from scipy import stats

print("=" * 70)
print("MARKET BIAS ANALYSIS")
print("Finding systematic market inefficiencies")
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

DIVISIONS = {
    'Atlantic': ['BOS', 'BKN', 'NYK', 'PHI', 'TOR'],
    'Central': ['CHI', 'CLE', 'DET', 'IND', 'MIL'],
    'Southeast': ['ATL', 'CHA', 'MIA', 'ORL', 'WAS'],
    'Northwest': ['DEN', 'MIN', 'OKC', 'POR', 'UTA'],
    'Pacific': ['GSW', 'LAC', 'LAL', 'PHX', 'SAC'],
    'Southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS'],
}

def get_division(team):
    for div, teams in DIVISIONS.items():
        if team in teams:
            return div
    return None

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

# Calculate covers
merged['point_diff'] = merged['home_score'] - merged['away_score']
merged['home_covers'] = merged['point_diff'] > -merged['home_spread']
merged['away_covers'] = merged['point_diff'] < -merged['home_spread']
merged['year'] = merged['date'].dt.year

# Add features
merged['home_div'] = merged['home_team'].apply(get_division)
merged['away_div'] = merged['away_team'].apply(get_division)
merged['divisional'] = merged['home_div'] == merged['away_div']

# Focus on recent data (2020+)
recent = merged[merged['date'] >= '2020-01-01'].copy()
print(f"\nAnalyzing {len(recent)} games from 2020-present")


def analyze_situation(df, name, bet_home):
    """Analyze ATS performance for a situation."""
    if len(df) < 30:
        return None

    wins = df['home_covers'].sum() if bet_home else df['away_covers'].sum()
    n = len(df)
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


results = []

print("\n" + "=" * 70)
print("1. SPREAD SIZE ANALYSIS (Market bias toward favorites)")
print("=" * 70)

print("\n--- Betting the DOG by spread size ---")
for low, high in [(-15, -10), (-10, -7), (-7, -5), (-5, -3), (-3, 0), (0, 3), (3, 5), (5, 7), (7, 10), (10, 15)]:
    mask = (recent['home_spread'] >= low) & (recent['home_spread'] < high)
    df = recent[mask]
    # Bet the dog (away if home favored, home if away favored)
    if low < 0:  # Home favored, bet away
        r = analyze_situation(df, f"Spread [{low}, {high}) - bet AWAY (dog)", bet_home=False)
    else:  # Away favored, bet HOME (dog)
        r = analyze_situation(df, f"Spread [{low}, {high}) - bet HOME (dog)", bet_home=True)
    if r:
        results.append(r)
        print(f"  [{low:+3d}, {high:+3d}): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("2. HOME UNDERDOG ANALYSIS")
print("=" * 70)

print("\n--- Home underdogs by spread size ---")
for low, high in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 20)]:
    mask = (recent['home_spread'] >= low) & (recent['home_spread'] < high)
    r = analyze_situation(recent[mask], f"Home dog +{low} to +{high} (bet HOME)", bet_home=True)
    if r:
        results.append(r)
        print(f"  +{low} to +{high}: {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("3. DIVISIONAL GAMES")
print("=" * 70)

div_games = recent[recent['divisional'] == True]
non_div = recent[recent['divisional'] == False]

r = analyze_situation(div_games, "Divisional games (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Divisional (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

r = analyze_situation(non_div, "Non-divisional (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Non-divisional (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

# Divisional dogs
div_dogs = div_games[div_games['home_spread'] > 0]
r = analyze_situation(div_dogs, "Divisional home dogs (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Divisional home dogs (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("4. LARGE FAVORITES (Public loves favorites)")
print("=" * 70)

# Fade double-digit favorites
large_fav = recent[recent['home_spread'] <= -10]
r = analyze_situation(large_fav, "Fade home -10+ (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Fade home -10+ (bet AWAY): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

large_fav_away = recent[recent['home_spread'] >= 10]
r = analyze_situation(large_fav_away, "Fade away -10+ (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Fade away -10+ (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

# Extreme favorites
extreme = recent[recent['home_spread'] <= -12]
r = analyze_situation(extreme, "Fade home -12+ (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Fade home -12+ (bet AWAY): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("5. SMALL SPREAD GAMES (Pick'em type)")
print("=" * 70)

small_spread = recent[(recent['home_spread'] >= -2.5) & (recent['home_spread'] <= 2.5)]
r = analyze_situation(small_spread, "Pick'em to 2.5 (bet HOME)", bet_home=True)
if r:
    results.append(r)
    print(f"Pick'em to 2.5 (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

r = analyze_situation(small_spread, "Pick'em to 2.5 (bet AWAY)", bet_home=False)
if r:
    results.append(r)
    print(f"Pick'em to 2.5 (bet AWAY): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("6. SPECIFIC KEY NUMBERS")
print("=" * 70)

for num in [3, 6, 7]:
    # Home favored by exactly num or num+0.5
    key_home = recent[(recent['home_spread'] >= -num-0.5) & (recent['home_spread'] <= -num+0.5)]
    r = analyze_situation(key_home, f"Home -{num} (bet HOME)", bet_home=True)
    if r:
        results.append(r)
        print(f"Home -{num} (bet HOME): {r['win_rate']*100:.1f}% ATS (n={r['n']}, ROI={r['roi']:+.1f}%)")

print("\n" + "=" * 70)
print("7. YEAR-OVER-YEAR CONSISTENCY CHECK")
print("=" * 70)

# Check if fade large favorites is consistent
print("\nFade home -10+ by year:")
for year in sorted(recent['year'].unique()):
    year_data = recent[(recent['year'] == year) & (recent['home_spread'] <= -10)]
    if len(year_data) >= 10:
        ats = (~year_data['home_covers']).mean()  # Away covers
        print(f"  {year}: {ats*100:.1f}% ATS (n={len(year_data)})")

print("\n" + "=" * 70)
print("SUMMARY - TOP MARKET BIASES")
print("=" * 70)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    # Show all sorted by ROI
    print("\n--- All situations sorted by ROI ---")
    for _, r in results_df.sort_values('roi', ascending=False).head(15).iterrows():
        sig = "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.10 else ""
        print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {sig:2s} {r['name']}")

    # Highlight statistically significant
    significant = results_df[results_df['p_value'] < 0.10].sort_values('roi', ascending=False)
    if len(significant) > 0:
        print("\n--- STATISTICALLY SIGNIFICANT (p < 0.10) ---")
        for _, r in significant.iterrows():
            print(f"{r['roi']:+6.1f}% ROI  {r['win_rate']*100:5.1f}%  n={r['n']:4d}  p={r['p_value']:.3f} {r['name']}")
    else:
        print("\nNo statistically significant market biases found at p < 0.10")
