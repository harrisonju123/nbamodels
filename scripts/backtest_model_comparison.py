#!/usr/bin/env python3
"""
Backtest Model Comparison: Baseline vs Tuned

Compares actual predictions from baseline and tuned models on historical data.
Uses real model predictions (not simulated) for accurate ROI calculation.

Usage:
    python scripts/backtest_model_comparison.py
    python scripts/backtest_model_comparison.py --min-edge 0.07
    python scripts/backtest_model_comparison.py --seasons 2023 2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from loguru import logger

from src.models.spread_model import SpreadPredictionModel
from src.features.game_features import GameFeatureBuilder

# Team name mapping (full name -> abbreviation)
TEAM_NAME_MAP = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}


def load_models() -> Tuple[SpreadPredictionModel, SpreadPredictionModel]:
    """Load baseline and tuned models."""
    baseline_path = Path("models/spread_model_baseline_proper.pkl")
    tuned_path = Path("models/spread_model_tuned_proper.pkl")

    if not baseline_path.exists():
        logger.error(f"Baseline model not found: {baseline_path}")
        return None, None

    if not tuned_path.exists():
        logger.error(f"Tuned model not found: {tuned_path}")
        logger.info("Using baseline model only")
        baseline = SpreadPredictionModel.load(str(baseline_path))
        return baseline, None

    logger.info("Loading models...")
    baseline = SpreadPredictionModel.load(str(baseline_path))
    tuned = SpreadPredictionModel.load(str(tuned_path))

    logger.success(f"Loaded baseline model with {len(baseline.feature_columns)} features")
    logger.success(f"Loaded tuned model with {len(tuned.feature_columns)} features")

    return baseline, tuned


def load_historical_odds() -> pd.DataFrame:
    """Load all historical odds data."""
    from glob import glob

    odds_files = glob("data/historical_odds/odds_*.parquet")
    if not odds_files:
        logger.warning("No historical odds files found")
        return pd.DataFrame()

    odds_list = []
    for file in odds_files:
        try:
            df = pd.read_parquet(file)
            odds_list.append(df)
        except Exception as e:
            logger.warning(f"Could not load {file}: {e}")

    if not odds_list:
        return pd.DataFrame()

    odds = pd.concat(odds_list, ignore_index=True)

    # Filter to spread market only
    odds = odds[odds['market'] == 'spread'].copy()

    # Get home team spreads (negative means favored)
    home_spreads = odds[odds['team'] == 'home'].copy()
    home_spreads = home_spreads.rename(columns={'line': 'spread', 'odds': 'spread_odds', 'implied_prob': 'spread_implied_prob'})
    home_spreads = home_spreads[['game_id', 'spread', 'spread_odds', 'spread_implied_prob', 'bookmaker']]

    # Take median spread across bookmakers for each game
    median_spreads = home_spreads.groupby('game_id').agg({
        'spread': 'median',
        'spread_odds': 'median',
        'spread_implied_prob': 'median'
    }).reset_index()

    logger.info(f"Loaded odds for {len(median_spreads)} games")
    return median_spreads


def load_backtest_data(seasons: list = None) -> pd.DataFrame:
    """Load historical games with odds for backtesting."""
    games = pd.read_parquet("data/raw/games.parquet")

    # Filter to specified seasons or default to recent seasons
    if seasons:
        games = games[games['season'].isin(seasons)]
    else:
        games = games[games['season'] >= 2023]  # Default: 2023-2024 seasons

    # Only completed games
    games = games[games['status'] == 'Final'].copy()

    # Load historical odds
    odds = load_historical_odds()

    if not odds.empty:
        # First get team names from odds to match with games
        from glob import glob
        odds_files = glob("data/historical_odds/odds_*.parquet")
        odds_raw = pd.concat([pd.read_parquet(f) for f in odds_files], ignore_index=True)

        # Get game info (date, teams) from odds data
        game_info = odds_raw[['game_id', 'commence_time', 'home_team', 'away_team']].drop_duplicates()
        game_info['game_date'] = pd.to_datetime(game_info['commence_time']).dt.date

        # Map team names to abbreviations
        game_info['home_team'] = game_info['home_team'].map(TEAM_NAME_MAP)
        game_info['away_team'] = game_info['away_team'].map(TEAM_NAME_MAP)

        # Merge odds with game info
        odds = odds.merge(game_info[['game_id', 'game_date', 'home_team', 'away_team']], on='game_id')

        # Prepare games for merge
        games['game_date'] = pd.to_datetime(games['date']).dt.date

        # Merge on date and teams
        games = games.merge(
            odds[['game_date', 'home_team', 'away_team', 'spread', 'spread_odds', 'spread_implied_prob']],
            on=['game_date', 'home_team', 'away_team'],
            how='left'
        )

        # Filter to games with spread data
        games_with_odds = games[games['spread'].notna()].copy()

        logger.info(f"Loaded {len(games_with_odds)} games from seasons {sorted(games_with_odds['season'].unique()) if len(games_with_odds) > 0 else []} with spread data")

        if len(games_with_odds) == 0:
            logger.warning("No games matched with odds data - check team name formatting")
            # Show sample for debugging
            logger.info(f"Sample game teams: {games[['home_team', 'away_team']].head(2).to_dict('records')}")
            logger.info(f"Sample odds teams: {odds[['home_team', 'away_team']].head(2).to_dict('records')}")
            return pd.DataFrame()

        return games_with_odds
    else:
        logger.warning("No odds data available - using simulated spreads")
        # Fallback: estimate spread from scores (this is for older data without odds)
        games['spread'] = 0  # Neutral spread assumption
        games['spread_odds'] = -110
        games['spread_implied_prob'] = 0.5238  # -110 implied probability
        return games


def backtest_model(
    model: SpreadPredictionModel,
    games: pd.DataFrame,
    feature_builder: GameFeatureBuilder,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Backtest a model on historical games using real spreads.

    Returns:
        DataFrame with bet recommendations and outcomes
    """
    # Save spread data and outcomes before feature building
    outcome_data = games[['game_id', 'home_score', 'away_score', 'spread', 'spread_odds', 'spread_implied_prob']].copy()

    # Build features
    logger.info("Building features for backtest data...")
    features_df = feature_builder.build_game_features(games)

    # Merge outcome data back
    features_df = features_df.merge(outcome_data, on='game_id', how='left')

    # Deduplicate - ensure one row per game
    features_df = features_df.drop_duplicates(subset=['game_id'], keep='first')

    # Get feature columns (ensure no leakage)
    exclude_cols = [
        # Identifiers
        'game_id', 'date', 'season', 'home_team', 'away_team',
        'home_team_id', 'away_team_id', 'status',
        # Direct target leakage
        'home_score', 'away_score', 'home_win', 'away_win',
        'point_diff', 'home_point_diff', 'away_point_diff',
        'total_points', 'home_margin', 'away_margin',
        # Derived from scores
        'home_cover', 'away_cover', 'over', 'under',
        'actual_total', 'actual_spread', 'actual_margin',
        # Odds data (don't use as features)
        'spread', 'spread_odds', 'spread_implied_prob'
    ]

    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if c in model.feature_columns]

    # Make predictions
    logger.info(f"Making predictions with {len(feature_cols)} features...")
    X = features_df[feature_cols].fillna(0)
    model_probs = model.predict_proba(X)

    # Add predictions to features
    features_df['model_prob'] = model_probs
    features_df['model_pred'] = (model_probs > 0.5).astype(int)

    # Use real market data
    features_df['implied_prob'] = features_df['spread_implied_prob']
    features_df['odds'] = features_df['spread_odds']

    # Calculate actual spread coverage
    # Home team covers if: (home_score - away_score) > -spread
    # Example: If spread is -3.5 (home favored), home covers if they win by 4+
    features_df['point_margin'] = features_df['home_score'] - features_df['away_score']
    features_df['home_covered'] = (features_df['point_margin'] > -features_df['spread']).astype(int)

    # Calculate edge
    features_df['edge'] = features_df['model_prob'] - features_df['implied_prob']

    # Filter to bets with sufficient edge
    bets = features_df[features_df['edge'] >= min_edge].copy()

    logger.info(f"Found {len(bets)} bets with edge >= {min_edge:.1%}")

    if bets.empty:
        return pd.DataFrame()

    # Convert American odds to decimal
    def american_to_decimal(odds):
        """Convert American odds to decimal odds."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    decimal_odds = bets['odds'].apply(american_to_decimal)

    # Calculate Kelly bet sizing
    b = decimal_odds - 1
    p = bets['model_prob']
    q = 1 - p

    kelly = ((p * b - q) / b).clip(lower=0) * kelly_fraction
    bets['kelly_fraction'] = kelly
    bets['stake'] = 100  # $100 unit stake
    bets['bet_size'] = bets['stake'] * bets['kelly_fraction']

    # Determine outcomes based on SPREAD COVERAGE (not just game winner)
    bets['actual_outcome'] = bets['home_covered']
    bets['predicted_correctly'] = (bets['model_pred'] == bets['actual_outcome']).astype(int)

    # Calculate profit/loss using actual odds
    bets['win'] = bets['predicted_correctly'].astype(bool)
    bets['profit'] = np.where(
        bets['win'],
        bets['bet_size'] * (decimal_odds - 1),  # Win: stake * (decimal_odds - 1)
        -bets['bet_size']  # Lose: lose full stake
    )

    return bets


def calculate_metrics(bets: pd.DataFrame, model_name: str) -> Dict:
    """Calculate comprehensive performance metrics."""
    if bets.empty:
        return {'model': model_name, 'total_bets': 0}

    total_bets = len(bets)
    wins = bets['win'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_profit = bets['profit'].sum()
    total_staked = bets['bet_size'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # Edge analysis
    avg_edge = bets['edge'].mean()
    avg_model_prob = bets['model_prob'].mean()

    # By edge bucket
    edge_buckets = pd.cut(
        bets['edge'] * 100,
        bins=[0, 5, 7, 10, np.inf],
        labels=['5-7%', '7-10%', '10-15%', '15%+']
    )

    by_edge = bets.groupby(edge_buckets, observed=False).agg({
        'win': lambda x: x.sum() / len(x) if len(x) > 0 else 0,
        'profit': 'sum',
        'bet_size': 'sum'
    }).rename(columns={'win': 'win_rate'})

    by_edge['roi'] = (by_edge['profit'] / by_edge['bet_size'] * 100).fillna(0)
    by_edge['count'] = bets.groupby(edge_buckets, observed=False).size()

    return {
        'model': model_name,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'avg_edge': avg_edge,
        'avg_model_prob': avg_model_prob,
        'by_edge': by_edge
    }


def print_comparison_report(baseline_metrics: Dict, tuned_metrics: Dict = None):
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("📊 MODEL BACKTEST COMPARISON")
    print("=" * 80)

    # Baseline results
    print(f"\n{'='*40}")
    print(f"📈 BASELINE MODEL")
    print(f"{'='*40}")
    print(f"Total Bets:   {baseline_metrics['total_bets']}")
    print(f"Win Rate:     {baseline_metrics['win_rate']:.1%} ({baseline_metrics['wins']}W-{baseline_metrics['losses']}L)")
    print(f"Total Profit: ${baseline_metrics['total_profit']:,.2f}")
    print(f"ROI:          {baseline_metrics['roi']:+.2f}%")
    print(f"Avg Edge:     {baseline_metrics['avg_edge']:.2%}")

    if not baseline_metrics['by_edge'].empty:
        print(f"\n{'Edge':<12} {'Bets':<8} {'Win Rate':<12} {'ROI':<10}")
        print("-" * 50)
        for edge, row in baseline_metrics['by_edge'].iterrows():
            if row['count'] > 0:
                print(f"{edge:<12} {int(row['count']):<8} {row['win_rate']:>6.1%}       {row['roi']:>7.1f}%")

    # Tuned results (if available)
    if tuned_metrics and tuned_metrics['total_bets'] > 0:
        print(f"\n{'='*40}")
        print(f"🎯 TUNED MODEL")
        print(f"{'='*40}")
        print(f"Total Bets:   {tuned_metrics['total_bets']}")
        print(f"Win Rate:     {tuned_metrics['win_rate']:.1%} ({tuned_metrics['wins']}W-{tuned_metrics['losses']}L)")
        print(f"Total Profit: ${tuned_metrics['total_profit']:,.2f}")
        print(f"ROI:          {tuned_metrics['roi']:+.2f}%")
        print(f"Avg Edge:     {tuned_metrics['avg_edge']:.2%}")

        if not tuned_metrics['by_edge'].empty:
            print(f"\n{'Edge':<12} {'Bets':<8} {'Win Rate':<12} {'ROI':<10}")
            print("-" * 50)
            for edge, row in tuned_metrics['by_edge'].iterrows():
                if row['count'] > 0:
                    print(f"{edge:<12} {int(row['count']):<8} {row['win_rate']:>6.1%}       {row['roi']:>7.1f}%")

        # Comparison
        print(f"\n{'='*40}")
        print(f"📊 IMPROVEMENT (Tuned vs Baseline)")
        print(f"{'='*40}")

        roi_diff = tuned_metrics['roi'] - baseline_metrics['roi']
        wr_diff = tuned_metrics['win_rate'] - baseline_metrics['win_rate']
        profit_diff = tuned_metrics['total_profit'] - baseline_metrics['total_profit']

        print(f"ROI:          {roi_diff:+.2f}% ({baseline_metrics['roi']:.2f}% → {tuned_metrics['roi']:.2f}%)")
        print(f"Win Rate:     {wr_diff:+.2%} ({baseline_metrics['win_rate']:.1%} → {tuned_metrics['win_rate']:.1%})")
        print(f"Total Profit: ${profit_diff:+,.2f}")

        if roi_diff > 0:
            print(f"\n✅ Tuned model shows +{roi_diff:.2f}% ROI improvement!")
        else:
            print(f"\n⚠️  Baseline model performing better by {abs(roi_diff):.2f}% ROI")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Backtest model comparison')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 5%%)')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2023, 2024],
                       help='Seasons to backtest (default: 2023 2024)')
    parser.add_argument('--output', type=str, help='Save results to CSV')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MODEL BACKTEST COMPARISON")
    logger.info("=" * 60)

    # Load models
    baseline, tuned = load_models()
    if baseline is None:
        return 1

    # Load backtest data
    games = load_backtest_data(args.seasons)
    if games.empty:
        logger.error("No games found for backtesting")
        return 1

    # Initialize feature builder
    feature_builder = GameFeatureBuilder(
        use_referee_features=False,
        use_news_features=False,
        use_sentiment_features=False
    )

    # Backtest baseline model
    logger.info(f"\n{'='*60}")
    logger.info("Backtesting BASELINE model...")
    logger.info(f"{'='*60}")
    baseline_bets = backtest_model(
        baseline, games, feature_builder,
        min_edge=args.min_edge
    )
    baseline_metrics = calculate_metrics(baseline_bets, "Baseline")

    # Backtest tuned model (if available)
    tuned_metrics = None
    if tuned is not None:
        logger.info(f"\n{'='*60}")
        logger.info("Backtesting TUNED model...")
        logger.info(f"{'='*60}")
        tuned_bets = backtest_model(
            tuned, games, feature_builder,
            min_edge=args.min_edge
        )
        tuned_metrics = calculate_metrics(tuned_bets, "Tuned")

    # Print comparison report
    print_comparison_report(baseline_metrics, tuned_metrics)

    # Save if requested
    if args.output and tuned is not None:
        comparison_df = pd.DataFrame([
            {
                'model': 'baseline',
                **{k: v for k, v in baseline_metrics.items() if k != 'by_edge'}
            },
            {
                'model': 'tuned',
                **{k: v for k, v in tuned_metrics.items() if k != 'by_edge'}
            }
        ])
        comparison_df.to_csv(args.output, index=False)
        logger.info(f"Saved comparison to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
