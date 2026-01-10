#!/usr/bin/env python3
"""
Validate B2B Backtest Results

Checks:
1. Spread calculation correctness
2. Data quality issues
3. Sample game verification
4. Period-by-period analysis
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

print("=" * 80)
print("B2B BACKTEST VALIDATION")
print("=" * 80)

# Load data
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')

# Team mapping
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

# Merge
merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'spread', 'total']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

test_start = pd.Timestamp('2022-10-01')
test_data = merged[merged['date'] >= test_start].copy()
test_data = test_data.dropna(subset=['spread'])

print(f"\nTest data: {len(test_data)} games")

# Calculate spread results
test_data['home_margin'] = test_data['home_score'] - test_data['away_score']
test_data['spread_result'] = test_data['home_margin'] + test_data['spread']
test_data['home_covers'] = test_data['spread_result'] > 0
test_data['away_covers'] = test_data['spread_result'] < 0
test_data['push'] = test_data['spread_result'] == 0

print("\n" + "=" * 80)
print("1. SPREAD INTERPRETATION CHECK")
print("=" * 80)

print("\nSpread value distribution:")
print(test_data['spread'].describe())
print(f"\nNegative spreads (home favored): {(test_data['spread'] < 0).sum()} ({(test_data['spread'] < 0).mean():.1%})")
print(f"Positive spreads (away favored): {(test_data['spread'] > 0).sum()} ({(test_data['spread'] > 0).mean():.1%})")
print(f"Zero spreads (pick'em): {(test_data['spread'] == 0).sum()}")

print("\n" + "=" * 80)
print("2. SAMPLE GAME VERIFICATION")
print("=" * 80)

# Show examples of each type
print("\nExample 1: Home Favored (negative spread)")
sample = test_data[test_data['spread'] < -5].iloc[0]
print(f"  Date: {sample['date'].date()}")
print(f"  Game: {sample['away_team']} @ {sample['home_team']}")
print(f"  Score: {sample['away_score']}-{sample['home_score']}")
print(f"  Spread: {sample['spread']:.1f} (home favored by {-sample['spread']:.1f})")
print(f"  Home margin: {sample['home_margin']:.1f}")
print(f"  Spread result: {sample['spread_result']:.1f}")
print(f"  Home covers: {sample['home_covers']}")
print(f"  Calculation: {sample['home_margin']:.1f} + ({sample['spread']:.1f}) = {sample['spread_result']:.1f}")
if sample['home_covers']:
    print(f"  ✓ Home covered (won by more than spread)")
else:
    print(f"  ✗ Home did not cover")

print("\nExample 2: Away Favored (positive spread)")
sample = test_data[test_data['spread'] > 5].iloc[0]
print(f"  Date: {sample['date'].date()}")
print(f"  Game: {sample['away_team']} @ {sample['home_team']}")
print(f"  Score: {sample['away_score']}-{sample['home_score']}")
print(f"  Spread: {sample['spread']:.1f} (away favored by {sample['spread']:.1f})")
print(f"  Home margin: {sample['home_margin']:.1f}")
print(f"  Spread result: {sample['spread_result']:.1f}")
print(f"  Home covers: {sample['home_covers']}")
print(f"  Calculation: {sample['home_margin']:.1f} + ({sample['spread']:.1f}) = {sample['spread_result']:.1f}")
if sample['home_covers']:
    print(f"  ✓ Home covered")
else:
    print(f"  ✗ Home did not cover (away won by more than spread)")

print("\nExample 3: Close game with spread")
sample = test_data[(test_data['spread'].abs() < 3) & (test_data['home_margin'].abs() < 5)].iloc[0]
print(f"  Date: {sample['date'].date()}")
print(f"  Game: {sample['away_team']} @ {sample['home_team']}")
print(f"  Score: {sample['away_score']}-{sample['home_score']}")
print(f"  Spread: {sample['spread']:.1f}")
print(f"  Home margin: {sample['home_margin']:.1f}")
print(f"  Spread result: {sample['spread_result']:.1f}")
print(f"  Home covers: {sample['home_covers']}")

print("\n" + "=" * 80)
print("3. OVERALL COVER RATES")
print("=" * 80)

home_cover_rate = test_data['home_covers'].mean()
away_cover_rate = test_data['away_covers'].mean()
push_rate = test_data['push'].mean()

print(f"\nHome covers: {test_data['home_covers'].sum()} / {len(test_data)} = {home_cover_rate:.1%}")
print(f"Away covers: {test_data['away_covers'].sum()} / {len(test_data)} = {away_cover_rate:.1%}")
print(f"Pushes: {test_data['push'].sum()} / {len(test_data)} = {push_rate:.1%}")
print(f"Total: {home_cover_rate + away_cover_rate + push_rate:.1%}")

print(f"\n⚠️  WARNING: Home cover rate is {home_cover_rate:.1%} (expected ~50%)")
if home_cover_rate > 0.60:
    print("   This suggests a data issue or calculation error!")

print("\n" + "=" * 80)
print("4. PERIOD-BY-PERIOD ANALYSIS")
print("=" * 80)

test_data['season'] = test_data['date'].dt.year
test_data.loc[test_data['date'].dt.month >= 10, 'season'] = test_data['date'].dt.year + 1

print("\nHome cover rate by season:")
for season in sorted(test_data['season'].unique()):
    season_data = test_data[test_data['season'] == season]
    cover_rate = season_data['home_covers'].mean()
    print(f"  {season-1}-{season}: {cover_rate:.1%} ({len(season_data)} games)")

print("\n" + "=" * 80)
print("5. SPREAD DISTRIBUTION CHECK")
print("=" * 80)

print("\nHome cover rate by spread range:")
for spread_min, spread_max in [(-20, -10), (-10, -5), (-5, -2.5), (-2.5, 0), (0, 2.5), (2.5, 5), (5, 10), (10, 20)]:
    mask = (test_data['spread'] >= spread_min) & (test_data['spread'] < spread_max)
    subset = test_data[mask]
    if len(subset) > 0:
        cover_rate = subset['home_covers'].mean()
        print(f"  Spread {spread_min:+5.1f} to {spread_max:+5.1f}: {cover_rate:.1%} ({len(subset):4d} games)")

print("\n" + "=" * 80)
print("6. DATA QUALITY CHECKS")
print("=" * 80)

# Check for duplicates
duplicates = test_data.duplicated(subset=['date', 'home_team', 'away_team'], keep=False)
print(f"\nDuplicate games: {duplicates.sum()}")

# Check for missing data
print(f"\nMissing spread data: {test_data['spread'].isna().sum()}")
print(f"Missing score data: {test_data['home_score'].isna().sum()}")

# Check for outliers
print(f"\nSpread outliers (|spread| > 20): {(test_data['spread'].abs() > 20).sum()}")
print(f"Score outliers (total > 300): {((test_data['home_score'] + test_data['away_score']) > 300).sum()}")

print("\n" + "=" * 80)
print("7. ALTERNATIVE SPREAD INTERPRETATION TEST")
print("=" * 80)

print("\nTesting if spread should be inverted...")
# Try inverting spread
test_data['spread_result_inverted'] = test_data['home_margin'] - test_data['spread']
test_data['home_covers_inverted'] = test_data['spread_result_inverted'] > 0

inverted_cover_rate = test_data['home_covers_inverted'].mean()
print(f"With inverted spread calculation:")
print(f"  Home covers: {inverted_cover_rate:.1%}")
print(f"  Away covers: {(~test_data['home_covers_inverted']).mean():.1%}")

if abs(inverted_cover_rate - 0.5) < abs(home_cover_rate - 0.5):
    print(f"\n⚠️  Inverted calculation is closer to 50%!")
    print(f"     Original: {home_cover_rate:.1%}, Inverted: {inverted_cover_rate:.1%}")
    print(f"     We may be applying the spread incorrectly!")

print("\n" + "=" * 80)
print("8. MANUAL VERIFICATION (Random Sample)")
print("=" * 80)

print("\nRandomly selected games for manual verification:")
np.random.seed(42)
sample_indices = np.random.choice(test_data.index, size=min(5, len(test_data)), replace=False)

for idx in sample_indices:
    game = test_data.loc[idx]
    print(f"\n  {game['date'].date()} - {game['away_team']} @ {game['home_team']}")
    print(f"    Final: {game['away_score']}-{game['home_score']} (home margin: {game['home_margin']:+.0f})")
    print(f"    Spread: {game['spread']:+.1f}")

    # Explain the spread
    if game['spread'] < 0:
        fav = "HOME"
        points = -game['spread']
    else:
        fav = "AWAY"
        points = game['spread']
    print(f"    {fav} favored by {points:.1f}")

    # Who covered?
    print(f"    Spread result: {game['home_margin']:.0f} + ({game['spread']:+.1f}) = {game['spread_result']:+.1f}")

    if game['home_covers']:
        print(f"    ✓ HOME covered")
    elif game['away_covers']:
        print(f"    ✓ AWAY covered")
    else:
        print(f"    PUSH")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if home_cover_rate > 0.60 or home_cover_rate < 0.40:
    print("\n🚨 VALIDATION FAILED")
    print(f"   Home cover rate ({home_cover_rate:.1%}) is significantly different from expected 50%")
    print("\n   Possible causes:")
    print("   1. Spread calculation is incorrect")
    print("   2. Spread data interpretation is wrong")
    print("   3. Data quality issues in historical odds")
    print("   4. Sample bias in test period")
    print("\n   Recommendation: Investigate and fix before using B2B strategy")
else:
    print("\n✅ VALIDATION PASSED")
    print(f"   Home cover rate ({home_cover_rate:.1%}) is within acceptable range")
    print("   B2B strategy results can be trusted")
