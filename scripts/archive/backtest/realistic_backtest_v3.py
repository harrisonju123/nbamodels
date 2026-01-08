"""
Realistic Backtest V3 - Optimized

Key improvements over V2:
1. Pre-builds features ONCE (no per-game rebuilding)
2. Uses temporal filtering for train/test split
3. Walk-forward retraining at intervals
4. Line movement simulation
5. Bet availability simulation
6. Excludes leaky features (Four Factors, home_spread)
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

# Skip Four Factors (has data leakage - uses game-level stats)
original_add_four_factors = TeamFeatureBuilder.add_four_factors
TeamFeatureBuilder.add_four_factors = lambda self, df: df

print("=" * 70)
print("REALISTIC BACKTEST V3 - OPTIMIZED")
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

# Fix spread sign: negative = home favored
odds['home_spread'] = odds.apply(
    lambda r: -r['spread'] if r['whos_favored'] == 'home' else r['spread'],
    axis=1
)

# Merge games with odds
merged = games.merge(
    odds[['date', 'home_team', 'away_team', 'home_spread']],
    on=['date', 'home_team', 'away_team'],
    how='left'
).sort_values('date').reset_index(drop=True)

print(f"Total games: {len(merged)}")
print(f"Games with spreads: {merged['home_spread'].notna().sum()}")

# Build ALL features ONCE upfront
print("\nBuilding features for all games (this takes a minute)...")
builder = GameFeatureBuilder()
all_features = builder.build_game_features(merged)

# Get feature columns (exclude leaky ones)
feature_cols = [c for c in builder.get_feature_columns(all_features)
                if c != 'home_spread'  # Don't use spread as feature for ATS
                and 'OREB_PCT' not in c  # Game-level stat
                and 'EFG_PCT' not in c   # Game-level stat
                and 'TS_PCT' not in c    # Game-level stat
                and 'TOV_PCT' not in c]  # Game-level stat

print(f"Feature columns: {len(feature_cols)}")

# Merge spreads into features
all_features = all_features.merge(
    merged[['game_id', 'home_spread', 'point_diff']].drop_duplicates(),
    on='game_id',
    how='left',
    suffixes=('', '_merged')
)

# Use merged point_diff if original is missing
if 'point_diff_merged' in all_features.columns:
    all_features['point_diff'] = all_features['point_diff'].fillna(all_features['point_diff_merged'])
    all_features = all_features.drop(columns=['point_diff_merged'])

# Parameters
MIN_TRAIN_GAMES = 1500
RETRAIN_INTERVAL = 300  # Retrain every N test games
DISAGREEMENT_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD = 0.55
LINE_MOVEMENT = 0.5  # Half point worse on average
BET_AVAILABILITY = 0.85  # 85% of bets actually available
STARTING_BANKROLL = 1000
BET_SIZE = 10  # Flat $10 bets

# Test period: 2022+ season
test_start_date = pd.Timestamp('2022-10-01')
train_data = all_features[all_features['date'] < test_start_date].copy()
test_data = all_features[all_features['date'] >= test_start_date].copy()

print(f"\nTrain period: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
print(f"Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
print(f"Train games: {len(train_data)}")
print(f"Test games: {len(test_data)}")
print(f"Test games with spreads: {test_data['home_spread'].notna().sum()}")
print(f"\nLine movement: {LINE_MOVEMENT} points")
print(f"Bet availability: {BET_AVAILABILITY*100}%")
print(f"Disagreement threshold: {DISAGREEMENT_THRESHOLD}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print()

# Tracking
bankroll = STARTING_BANKROLL
peak_bankroll = STARTING_BANKROLL
max_drawdown = 0
bets = []

np.random.seed(42)

# Initial model training
print("Training initial model...")
X_train = train_data[feature_cols].fillna(0)
y_train = train_data['home_win']
model = DualPredictionModel()
model.fit(X_train, y_train)
print("Initial model trained.\n")

# Walk-forward testing
test_games_processed = 0
last_retrain_count = 0

for idx, row in test_data.iterrows():
    # Skip if no spread data
    if pd.isna(row['home_spread']):
        continue

    test_games_processed += 1

    # Retrain model periodically
    if (test_games_processed - last_retrain_count) >= RETRAIN_INTERVAL:
        # Use all data up to this point for retraining
        retrain_mask = all_features['date'] < row['date']
        retrain_data = all_features[retrain_mask]

        if len(retrain_data) >= MIN_TRAIN_GAMES:
            X_retrain = retrain_data[feature_cols].fillna(0)
            y_retrain = retrain_data['home_win']
            model = DualPredictionModel()
            model.fit(X_retrain, y_retrain)
            last_retrain_count = test_games_processed
            print(f"Retrained model at game {test_games_processed} ({row['date'].date()})")

    # Get prediction for this game
    X_game = pd.DataFrame([row[feature_cols].fillna(0)])
    preds = model.get_predictions(X_game)

    mlp_prob = preds['mlp_prob'].iloc[0]
    xgb_prob = preds['xgb_prob'].iloc[0]
    disagreement = abs(mlp_prob - xgb_prob)

    # Check if we should bet (model disagreement filter)
    if disagreement >= DISAGREEMENT_THRESHOLD:
        # Simulate bet availability
        if np.random.random() > BET_AVAILABILITY:
            continue

        spread = row['home_spread']
        actual_diff = row['point_diff']

        if xgb_prob >= CONFIDENCE_THRESHOLD:
            # Bet home to cover - line moves against us
            adjusted_spread = spread - LINE_MOVEMENT
            # Spread coverage: home covers if actual_diff + adjusted_spread > 0
            spread_result = actual_diff + adjusted_spread
            home_covers = spread_result > 0
            bet_side = 'home'
            won = home_covers
        elif xgb_prob <= (1 - CONFIDENCE_THRESHOLD):
            # Bet away to cover - line moves against us
            adjusted_spread = spread + LINE_MOVEMENT
            # Spread coverage: away covers if actual_diff + adjusted_spread < 0
            spread_result = actual_diff + adjusted_spread
            away_covers = spread_result < 0
            bet_side = 'away'
            won = away_covers
        else:
            continue

        # Calculate P&L at -110 odds
        if won:
            pnl = BET_SIZE * 0.909  # Win $9.09 on $10 bet
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
            'adjusted_spread': adjusted_spread,
            'actual_diff': actual_diff,
            'won': won,
            'pnl': pnl,
            'bankroll': bankroll,
            'xgb_prob': xgb_prob,
            'mlp_prob': mlp_prob,
            'disagreement': disagreement
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
    breakeven = 0.524  # 52.4% needed at -110 odds
    print(f"\nBreak-even win rate: {breakeven*100:.1f}%")
    print(f"Edge over break-even: {(wins/len(bets_df) - breakeven)*100:.1f}%")

    # Statistical significance
    from scipy import stats
    z_score = (wins/len(bets_df) - 0.50) / np.sqrt(0.25 / len(bets_df))
    p_value = 1 - stats.norm.cdf(z_score)
    print(f"\nZ-score vs 50%: {z_score:.2f}")
    print(f"P-value (one-tailed): {p_value:.4f}")

    # By season
    print("\n" + "-" * 40)
    print("By Season:")
    print("-" * 40)
    bets_df['year'] = bets_df['date'].dt.year
    for year in sorted(bets_df['year'].unique()):
        year_bets = bets_df[bets_df['year'] == year]
        if len(year_bets) > 0:
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

    # By confidence level
    print("\n" + "-" * 40)
    print("By Confidence Level:")
    print("-" * 40)
    bets_df['confidence'] = bets_df['xgb_prob'].apply(
        lambda p: max(p, 1-p)  # Convert to confidence regardless of side
    )
    for conf_min, conf_max in [(0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]:
        conf_bets = bets_df[(bets_df['confidence'] >= conf_min) & (bets_df['confidence'] < conf_max)]
        if len(conf_bets) > 0:
            cw = conf_bets['won'].sum()
            cl = len(conf_bets) - cw
            cwr = cw / len(conf_bets) * 100
            croi = conf_bets['pnl'].sum() / (len(conf_bets) * BET_SIZE) * 100
            print(f"  {conf_min:.0%}-{conf_max:.0%}: {cw}W-{cl}L ({cwr:.1f}%) ROI: {croi:.1f}%")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("Previous (potential leakage): 60.0% win rate, 14.6% ROI")
    print(f"Realistic (this test):        {wins/len(bets_df)*100:.1f}% win rate, {total_pnl/(len(bets_df)*BET_SIZE)*100:.1f}% ROI")

    # Save results
    bets_df.to_csv('data/realistic_backtest_results.csv', index=False)
    print(f"\nResults saved to data/realistic_backtest_results.csv")
