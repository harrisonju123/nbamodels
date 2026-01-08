#!/usr/bin/env python3
"""
Test B2B Situational Betting Strategy

Tests the hypothesis that betting against B2B road teams (or betting on
their opponents) is profitable.

Strategies to test:
1. Bet HOME when away team is on B2B
2. Bet AWAY when home team is on B2B
3. Bet on rested team vs tired team (rest advantage)
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy import stats

print("=" * 80)
print("B2B SITUATIONAL BETTING STRATEGY BACKTEST")
print("=" * 80)

# Load game data
games = pd.read_parquet('data/raw/games.parquet')
odds = pd.read_csv('data/raw/historical_odds.csv')

print(f"\nLoaded {len(games)} games")
print(f"Loaded {len(odds)} odds records")

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

# Normalize odds team names
odds['home_team'] = odds['home'].map(TEAM_MAP)
odds['away_team'] = odds['away'].map(TEAM_MAP)
odds['date'] = pd.to_datetime(odds['date'])

# Normalize games
games['date'] = pd.to_datetime(games['date'])

# Merge odds with games
merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'spread', 'total']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

print(f"Merged to {len(merged)} games with odds")
print(f"Games with spread data: {merged['spread'].notna().sum()} ({merged['spread'].notna().mean():.1%})")

# Calculate rest days and B2B status
def add_rest_features(df):
    """Add rest days and B2B indicators."""
    df = df.sort_values('date').copy()
    team_last_game = {}

    home_rest, away_rest = [], []
    home_b2b, away_b2b = [], []

    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        game_date = row['date']

        # Calculate rest days
        h_rest = (game_date - team_last_game.get(ht, game_date - pd.Timedelta(days=7))).days
        a_rest = (game_date - team_last_game.get(at, game_date - pd.Timedelta(days=7))).days

        home_rest.append(h_rest)
        away_rest.append(a_rest)

        # B2B = 1 day rest
        home_b2b.append(1 if h_rest == 1 else 0)
        away_b2b.append(1 if a_rest == 1 else 0)

        # Update last game
        team_last_game[ht] = game_date
        team_last_game[at] = game_date

    df['home_rest'] = home_rest
    df['away_rest'] = away_rest
    df['home_b2b'] = home_b2b
    df['away_b2b'] = away_b2b
    df['rest_advantage'] = df['home_rest'] - df['away_rest']

    return df

print("\nCalculating rest days and B2B status...")
merged = add_rest_features(merged)

# Filter to test period
test_start = pd.Timestamp('2022-10-01')
test_data = merged[merged['date'] >= test_start].copy()
test_data = test_data.dropna(subset=['spread'])

print(f"\nTest period: {len(test_data)} games (2022-2024)")
print(f"B2B games - Home: {test_data['home_b2b'].sum()}, Away: {test_data['away_b2b'].sum()}")

# Calculate actual spread result
# Positive spread result = home team beat the spread
test_data['home_margin'] = test_data['home_score'] - test_data['away_score']
test_data['spread_result'] = test_data['home_margin'] + test_data['spread']  # Positive = home covers
test_data['home_covers'] = test_data['spread_result'] > 0
test_data['away_covers'] = test_data['spread_result'] < 0


def backtest_strategy(df, name, bet_mask, bet_home):
    """
    Run spread backtest on filtered games.

    Args:
        df: Test data
        name: Strategy name
        bet_mask: Boolean mask for games to bet
        bet_home: True to bet home, False to bet away
    """
    games = df[bet_mask].copy()
    n = len(games)

    if n < 10:
        return None

    # Calculate wins
    if bet_home:
        wins = games['home_covers'].sum()
    else:
        wins = games['away_covers'].sum()

    win_rate = wins / n

    # ROI at -110 odds (risk 110 to win 100)
    # Win: +100, Loss: -110
    roi = (win_rate * (100/110) - (1 - win_rate)) * 100

    # Statistical test vs 50%
    z = (win_rate - 0.50) / np.sqrt(0.25 / n)
    p = 1 - stats.norm.cdf(z)

    return {
        'name': name,
        'n': n,
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi,
        'z_score': z,
        'p_value': p,
    }


print("\n" + "=" * 80)
print("B2B SITUATIONAL STRATEGIES")
print("=" * 80)

results = []

# Strategy 1: Bet HOME when away team is on B2B
print("\n--- FADE B2B ROAD TEAMS (Bet HOME) ---")
mask = test_data['away_b2b'] == 1
r = backtest_strategy(test_data, "Bet HOME vs B2B away team", mask, bet_home=True)
if r:
    results.append(r)
    print(f"  {r['name']}")
    print(f"    Games: {r['n']}")
    print(f"    Win Rate: {r['win_rate']*100:.1f}%")
    print(f"    ROI: {r['roi']:+.1f}%")
    print(f"    P-value: {r['p_value']:.4f} {'**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''}")

# Strategy 2: Bet AWAY when home team is on B2B
print("\n--- FADE B2B HOME TEAMS (Bet AWAY) ---")
mask = test_data['home_b2b'] == 1
r = backtest_strategy(test_data, "Bet AWAY vs B2B home team", mask, bet_home=False)
if r:
    results.append(r)
    print(f"  {r['name']}")
    print(f"    Games: {r['n']}")
    print(f"    Win Rate: {r['win_rate']*100:.1f}%")
    print(f"    ROI: {r['roi']:+.1f}%")
    print(f"    P-value: {r['p_value']:.4f} {'**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''}")

# Strategy 3: Bet on team with rest advantage >= 2 days
print("\n--- REST ADVANTAGE ---")
for advantage in [1, 2, 3]:
    # Bet home when home has rest advantage
    mask = test_data['rest_advantage'] >= advantage
    r = backtest_strategy(test_data, f"Bet HOME when rest advantage >= {advantage}", mask, bet_home=True)
    if r:
        results.append(r)
        print(f"  Rest advantage >= {advantage} (bet HOME): {r['n']:4d} games, {r['win_rate']*100:5.1f}% WR, {r['roi']:+6.1f}% ROI, p={r['p_value']:.4f}")

    # Bet away when away has rest advantage
    mask = test_data['rest_advantage'] <= -advantage
    r = backtest_strategy(test_data, f"Bet AWAY when rest advantage <= -{advantage}", mask, bet_home=False)
    if r:
        results.append(r)
        print(f"  Rest advantage <= -{advantage} (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:5.1f}% WR, {r['roi']:+6.1f}% ROI, p={r['p_value']:.4f}")

# Strategy 4: Both teams on B2B (coin flip?)
print("\n--- BOTH TEAMS B2B ---")
mask = (test_data['home_b2b'] == 1) & (test_data['away_b2b'] == 1)
r_home = backtest_strategy(test_data, "Both B2B - bet HOME", mask, bet_home=True)
r_away = backtest_strategy(test_data, "Both B2B - bet AWAY", mask, bet_home=False)
if r_home and r_away:
    print(f"  Both B2B - bet HOME: {r_home['n']:4d} games, {r_home['win_rate']*100:5.1f}% WR, {r_home['roi']:+6.1f}% ROI")
    print(f"  Both B2B - bet AWAY: {r_away['n']:4d} games, {r_away['win_rate']*100:5.1f}% WR, {r_away['roi']:+6.1f}% ROI")
    results.append(r_home)
    results.append(r_away)

# Strategy 5: Extreme rest disadvantage (opponent rested 3+ days, you're B2B)
print("\n--- EXTREME FATIGUE ---")
mask = (test_data['away_b2b'] == 1) & (test_data['home_rest'] >= 3)
r = backtest_strategy(test_data, "Bet HOME: away B2B + home rested 3+", mask, bet_home=True)
if r:
    results.append(r)
    print(f"  Away B2B + Home rested 3+ (bet HOME): {r['n']:4d} games, {r['win_rate']*100:5.1f}% WR, {r['roi']:+6.1f}% ROI, p={r['p_value']:.4f}")

mask = (test_data['home_b2b'] == 1) & (test_data['away_rest'] >= 3)
r = backtest_strategy(test_data, "Bet AWAY: home B2B + away rested 3+", mask, bet_home=False)
if r:
    results.append(r)
    print(f"  Home B2B + Away rested 3+ (bet AWAY): {r['n']:4d} games, {r['win_rate']*100:5.1f}% WR, {r['roi']:+6.1f}% ROI, p={r['p_value']:.4f}")

# Summary
print("\n" + "=" * 80)
print("RANKED B2B STRATEGIES (by ROI)")
print("=" * 80)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    # Filter to strategies with n >= 50
    significant = results_df[results_df['n'] >= 50].sort_values('roi', ascending=False)

    if len(significant) > 0:
        print("\nStrategies with n >= 50:")
        for _, r in significant.iterrows():
            sig = "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
            print(f"  {r['roi']:+6.1f}% ROI | {r['win_rate']*100:5.1f}% WR | n={r['n']:4d} | p={r['p_value']:.4f} {sig:2s} | {r['name']}")

    print("\nAll strategies:")
    for _, r in results_df.sort_values('roi', ascending=False).iterrows():
        sig = "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"  {r['roi']:+6.1f}% ROI | {r['win_rate']*100:5.1f}% WR | n={r['n']:4d} | p={r['p_value']:.4f} {sig} | {r['name']}")

# Baseline
print("\n" + "=" * 80)
print("BASELINE (ALL GAMES)")
print("=" * 80)
baseline_home = backtest_strategy(test_data, "Bet HOME (all games)", test_data['spread'].notna(), bet_home=True)
baseline_away = backtest_strategy(test_data, "Bet AWAY (all games)", test_data['spread'].notna(), bet_home=False)
print(f"All games - bet HOME: {baseline_home['win_rate']*100:.1f}% WR, {baseline_home['roi']:+.1f}% ROI")
print(f"All games - bet AWAY: {baseline_away['win_rate']*100:.1f}% WR, {baseline_away['roi']:+.1f}% ROI")
