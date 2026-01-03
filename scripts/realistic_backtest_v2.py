"""
Realistic Backtest V2

Key improvements:
1. Incremental feature building (no future data in rolling stats)
2. Walk-forward with model retraining
3. Simulates line movement/availability
4. Tracks bankroll and drawdowns
5. Conservative assumptions
"""

import sys
sys.path.insert(0, '.')
import os
from dotenv import load_dotenv
load_dotenv('.env')

import pandas as pd
import numpy as np
from datetime import timedelta
from src.models.dual_model import DualPredictionModel
from src.features.game_features import GameFeatureBuilder
from src.features.team_features import TeamFeatureBuilder

# Skip Four Factors (has leakage)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

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

print("=" * 70)
print("REALISTIC BACKTEST V2")
print("=" * 70)

# Parameters
MIN_TRAIN_GAMES = 1500  # Need at least this many games to train
RETRAIN_INTERVAL = 200  # Retrain every N games
DISAGREEMENT_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD = 0.55
LINE_MOVEMENT = 0.5  # Half point worse on average
BET_AVAILABILITY = 0.85  # 85% of bets actually available
STARTING_BANKROLL = 1000
BET_SIZE = 10  # Flat $10 bets

# Get test start index (2022+ season)
test_start_date = pd.Timestamp('2022-10-01')
test_start_idx = merged[merged['date'] >= test_start_date].index[0]

print(f"Train period: {merged.iloc[0]['date'].date()} to {merged.iloc[test_start_idx-1]['date'].date()}")
print(f"Test period: {merged.iloc[test_start_idx]['date'].date()} to {merged.iloc[-1]['date'].date()}")
print(f"Train games: {test_start_idx}")
print(f"Test games: {len(merged) - test_start_idx}")
print(f"Line movement: {LINE_MOVEMENT} points")
print(f"Bet availability: {BET_AVAILABILITY*100}%")
print()

# Tracking
bankroll = STARTING_BANKROLL
peak_bankroll = STARTING_BANKROLL
max_drawdown = 0
bets = []
daily_bankroll = []

builder = GameFeatureBuilder()
model = None
last_train_idx = 0

np.random.seed(42)

for i in range(test_start_idx, len(merged)):
    row = merged.iloc[i]
    
    # Skip if no spread data
    if pd.isna(row['home_spread']):
        continue
    
    # Retrain model periodically
    if model is None or (i - last_train_idx) >= RETRAIN_INTERVAL:
        # Build features using ONLY data up to this point
        train_data = merged.iloc[:i].copy()
        
        try:
            train_features = builder.build_game_features(train_data)
            feature_cols = [c for c in builder.get_feature_columns(train_features) 
                          if c != 'home_spread']
            
            X_train = train_features[feature_cols].fillna(0)
            y_train = train_features['home_win']
            
            if len(X_train) >= MIN_TRAIN_GAMES:
                model = DualPredictionModel()
                model.fit(X_train, y_train)
                model.feature_cols = feature_cols
                last_train_idx = i
        except Exception as e:
            continue
    
    if model is None:
        continue
    
    # Build features for this game using only past data
    try:
        # Include current game for feature building but don't use its outcome
        current_data = merged.iloc[:i+1].copy()
        current_features = builder.build_game_features(current_data)
        
        # Get the last row (current game)
        game_features = current_features.iloc[-1:].copy()
        
        X_game = game_features[model.feature_cols].fillna(0)
        preds = model.get_predictions(X_game)
        
        mlp_prob = preds['mlp_prob'].iloc[0]
        xgb_prob = preds['xgb_prob'].iloc[0]
        disagreement = abs(mlp_prob - xgb_prob)
        
        # Check if we should bet
        if disagreement >= DISAGREEMENT_THRESHOLD:
            # Simulate bet availability
            if np.random.random() > BET_AVAILABILITY:
                continue
            
            spread = row['home_spread']
            actual_diff = row['point_diff']
            
            if xgb_prob >= CONFIDENCE_THRESHOLD:
                # Bet home - line moves against us
                adjusted_spread = spread - LINE_MOVEMENT  # Worse spread for home
                home_covers = actual_diff > -adjusted_spread
                bet_side = 'home'
                won = home_covers
            elif xgb_prob <= (1 - CONFIDENCE_THRESHOLD):
                # Bet away - line moves against us  
                adjusted_spread = spread + LINE_MOVEMENT  # Worse spread for away
                away_covers = actual_diff < -adjusted_spread
                bet_side = 'away'
                won = away_covers
            else:
                continue
            
            # Calculate P&L
            if won:
                pnl = BET_SIZE * 0.909  # -110 odds
            else:
                pnl = -BET_SIZE
            
            bankroll += pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            drawdown = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, drawdown)
            
            bets.append({
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet_side': bet_side,
                'spread': spread,
                'adjusted_spread': adjusted_spread if bet_side == 'home' else adjusted_spread,
                'actual_diff': actual_diff,
                'won': won,
                'pnl': pnl,
                'bankroll': bankroll,
                'xgb_prob': xgb_prob,
                'disagreement': disagreement
            })
            
    except Exception as e:
        continue
    
    # Track daily bankroll
    if len(bets) > 0 and (len(daily_bankroll) == 0 or bets[-1]['date'] != daily_bankroll[-1][0]):
        daily_bankroll.append((row['date'], bankroll))

# Results
bets_df = pd.DataFrame(bets)
if len(bets_df) == 0:
    print("No bets placed!")
else:
    wins = bets_df['won'].sum()
    losses = len(bets_df) - wins
    total_pnl = bets_df['pnl'].sum()
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total bets: {len(bets_df)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {wins/len(bets_df)*100:.1f}%")
    print(f"")
    print(f"Starting bankroll: ${STARTING_BANKROLL}")
    print(f"Final bankroll: ${bankroll:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"ROI: {total_pnl/(len(bets_df)*BET_SIZE)*100:.1f}%")
    print(f"Max drawdown: {max_drawdown*100:.1f}%")
    print()
    
    # By season
    print("By Season:")
    bets_df['season'] = bets_df['date'].dt.year.apply(lambda y: y if pd.Timestamp(f'{y}-10-01') <= bets_df['date'].min() else y-1)
    for season in sorted(bets_df['date'].dt.year.unique()):
        season_bets = bets_df[bets_df['date'].dt.year == season]
        if len(season_bets) > 0:
            sw = season_bets['won'].sum()
            sl = len(season_bets) - sw
            swr = sw / len(season_bets) * 100
            sroi = season_bets['pnl'].sum() / (len(season_bets) * BET_SIZE) * 100
            print(f"  {season}: {sw}W-{sl}L ({swr:.1f}%) ROI: {sroi:.1f}%")
    
    print()
    print("=" * 70)
    print("COMPARISON TO PREVIOUS (INFLATED) RESULTS")
    print("=" * 70)
    print("Previous: 60% win rate, 14.6% ROI")
    print(f"Realistic: {wins/len(bets_df)*100:.1f}% win rate, {total_pnl/(len(bets_df)*BET_SIZE)*100:.1f}% ROI")
