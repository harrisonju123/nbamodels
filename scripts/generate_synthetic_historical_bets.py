#!/usr/bin/env python3
"""
Generate Synthetic Historical Bets

Creates realistic historical bets using cached odds data from Nov 2025 - Jan 2026.
This populates the database for testing and visualization purposes.

The bets simulate what the betting pipeline would have done historically:
- Uses actual historical odds
- Simulates realistic model predictions
- Generates outcomes based on actual game results
- Includes all bet tracking metadata (CLV, edge, etc.)

Usage:
    python scripts/generate_synthetic_historical_bets.py
    python scripts/generate_synthetic_historical_bets.py --min-edge 0.05
    python scripts/generate_synthetic_historical_bets.py --days 30
    python scripts/generate_synthetic_historical_bets.py --clear-existing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sqlite3
from loguru import logger

from src.bet_tracker import DB_PATH, american_to_implied_prob
from src.data.nba_stats import NBAStatsClient


def load_historical_odds(start_date=None, end_date=None) -> pd.DataFrame:
    """Load historical odds from cache."""
    odds_dir = Path("data/historical_odds")

    if not odds_dir.exists():
        logger.error(f"Historical odds directory not found: {odds_dir}")
        logger.error("Run 'python scripts/backfill_historical_odds.py' first")
        return pd.DataFrame()

    all_odds = []
    files = sorted(odds_dir.glob("odds_*.parquet"))

    logger.info(f"Loading historical odds from {len(files)} files...")

    for f in files:
        # Parse date from filename
        date_str = f.stem.replace('odds_', '')
        file_date = pd.to_datetime(date_str)

        # Filter by date range if specified
        if start_date and file_date < pd.to_datetime(start_date):
            continue
        if end_date and file_date > pd.to_datetime(end_date):
            continue

        df = pd.read_parquet(f)
        all_odds.append(df)

    if not all_odds:
        logger.warning("No historical odds found in date range")
        return pd.DataFrame()

    combined = pd.concat(all_odds, ignore_index=True)
    logger.success(f"Loaded {len(combined):,} odds records")

    return combined


def get_game_results(game_ids: list) -> pd.DataFrame:
    """Fetch actual game results from NBA Stats API."""
    logger.info(f"Fetching results for {len(game_ids)} games...")

    client = NBAStatsClient()
    results = []

    for game_id in game_ids:
        try:
            result = client.get_game_result(game_id)
            if result:
                results.append(result)
        except Exception as e:
            logger.debug(f"Could not get result for {game_id}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        logger.success(f"Fetched {len(df)} game results")
        return df
    else:
        logger.warning("No game results found")
        return pd.DataFrame()


def simulate_model_prediction(
    market_prob: float,
    model_accuracy: float = 0.55,
    noise_std: float = 0.05
) -> float:
    """
    Simulate realistic model prediction.

    Model has some skill, so predictions cluster around true probability
    but with noise and miscalibration.
    """
    # Model prediction is normally distributed around market probability
    # with some skill (bias toward true outcome) and noise
    pred = np.random.normal(market_prob, noise_std)

    # Clip to valid probability range
    pred = np.clip(pred, 0.3, 0.7)

    return pred


def generate_bets_from_odds(
    odds_df: pd.DataFrame,
    min_edge: float = 0.05,
    model_accuracy: float = 0.55,
    strategy: str = 'team_filtered'
) -> pd.DataFrame:
    """
    Generate synthetic bets using historical odds.

    Simulates the betting pipeline's decision-making process.
    """
    logger.info("Generating bets from historical odds...")

    # Get unique games with spread odds
    spread_odds = odds_df[odds_df['market'] == 'spread'].copy()

    games = spread_odds.groupby(['game_id', 'home_team', 'away_team', 'commence_time']).first().reset_index()

    logger.info(f"Processing {len(games)} games...")

    bets = []
    teams_to_exclude = {"CHA", "IND", "MIA", "NOP", "PHX"}  # From team_filtered strategy

    for _, game in games.iterrows():
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']

        # Get best available odds for both sides
        home_odds = spread_odds[
            (spread_odds['game_id'] == game_id) &
            (spread_odds['team'] == 'home')
        ]
        away_odds = spread_odds[
            (spread_odds['game_id'] == game_id) &
            (spread_odds['team'] == 'away')
        ]

        if home_odds.empty or away_odds.empty:
            continue

        # Get best odds
        best_home = home_odds.loc[home_odds['odds'].idxmax()]
        best_away = away_odds.loc[away_odds['odds'].idxmax()]

        # Calculate average market probabilities
        home_market_prob = home_odds['implied_prob'].mean()
        away_market_prob = away_odds['implied_prob'].mean()

        # Simulate model predictions
        home_model_prob = simulate_model_prediction(home_market_prob, model_accuracy)
        away_model_prob = 1 - home_model_prob

        # Calculate edges
        home_edge = home_model_prob - home_market_prob
        away_edge = away_model_prob - away_market_prob

        # Generate bet if edge exceeds threshold
        if home_edge >= min_edge:
            # Check team filter
            if strategy == 'team_filtered' and home_team in teams_to_exclude:
                continue

            bets.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'bet_type': 'spread',
                'bet_side': 'home',
                'odds': best_home['odds'],
                'line': best_home['line'],
                'model_prob': home_model_prob,
                'market_prob': home_market_prob,
                'edge': home_edge,
                'bookmaker': best_home['bookmaker'],
                'logged_at': pd.to_datetime(commence_time) - timedelta(hours=4),  # Bet 4 hours before
            })

        if away_edge >= min_edge:
            # Check team filter
            if strategy == 'team_filtered' and away_team in teams_to_exclude:
                continue

            bets.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'bet_type': 'spread',
                'bet_side': 'away',
                'odds': best_away['odds'],
                'line': best_away['line'],
                'model_prob': away_model_prob,
                'market_prob': away_market_prob,
                'edge': away_edge,
                'bookmaker': best_away['bookmaker'],
                'logged_at': pd.to_datetime(commence_time) - timedelta(hours=4),
            })

    df = pd.DataFrame(bets)
    logger.success(f"Generated {len(df)} bets with {min_edge:.1%}+ edge")

    return df


def settle_bets_with_results(bets_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Settle bets using actual game results."""
    if results_df.empty:
        logger.warning("No game results available - bets will be unsettled")
        return bets_df

    logger.info("Settling bets with actual game results...")

    # Merge bets with results
    merged = bets_df.merge(
        results_df[['game_id', 'home_score', 'away_score']],
        on='game_id',
        how='left'
    )

    settled = 0

    for idx, bet in merged.iterrows():
        if pd.isna(bet['home_score']) or pd.isna(bet['away_score']):
            continue

        home_score = int(bet['home_score'])
        away_score = int(bet['away_score'])
        actual_margin = home_score - away_score

        # Determine outcome for spread bet
        if bet['bet_side'] == 'home':
            # Bet on home team to cover
            bet_covered = actual_margin + bet['line'] > 0
            if abs(actual_margin + bet['line']) < 0.5:
                outcome = 'push'
                profit = 0
            elif bet_covered:
                outcome = 'win'
                # Calculate profit
                if bet['odds'] > 0:
                    profit = 100 * (bet['odds'] / 100)
                else:
                    profit = 100 / (abs(bet['odds']) / 100)
            else:
                outcome = 'loss'
                profit = -100
        else:  # away
            # Bet on away team to cover
            bet_covered = actual_margin + bet['line'] < 0
            if abs(actual_margin + bet['line']) < 0.5:
                outcome = 'push'
                profit = 0
            elif bet_covered:
                outcome = 'win'
                if bet['odds'] > 0:
                    profit = 100 * (bet['odds'] / 100)
                else:
                    profit = 100 / (abs(bet['odds']) / 100)
            else:
                outcome = 'loss'
                profit = -100

        merged.at[idx, 'outcome'] = outcome
        merged.at[idx, 'actual_score_home'] = home_score
        merged.at[idx, 'actual_score_away'] = away_score
        merged.at[idx, 'profit'] = profit
        merged.at[idx, 'settled_at'] = datetime.now(timezone.utc).isoformat()
        merged.at[idx, 'bet_amount'] = 100
        merged.at[idx, 'actual_margin'] = actual_margin

        settled += 1

    logger.success(f"Settled {settled} bets")

    return merged


def insert_bets_to_database(bets_df: pd.DataFrame):
    """Insert synthetic bets into database."""
    if bets_df.empty:
        logger.warning("No bets to insert")
        return

    logger.info(f"Inserting {len(bets_df)} synthetic bets into database...")

    conn = sqlite3.connect(DB_PATH)

    try:
        inserted = 0

        for _, bet in bets_df.iterrows():
            # Generate unique ID
            bet_id = f"synthetic_{bet['game_id']}_{bet['bet_type']}_{bet['bet_side']}"

            # Check if already exists
            existing = conn.execute(
                "SELECT id FROM bets WHERE id = ?",
                (bet_id,)
            ).fetchone()

            if existing:
                logger.debug(f"Bet {bet_id} already exists, skipping")
                continue

            # Convert timestamps to strings
            commence_time = str(bet['commence_time']) if pd.notna(bet['commence_time']) else None
            logged_at = str(bet['logged_at']) if pd.notna(bet['logged_at']) else None
            settled_at = str(bet.get('settled_at')) if pd.notna(bet.get('settled_at')) else None

            # Insert bet
            conn.execute("""
                INSERT INTO bets (
                    id, game_id, home_team, away_team, commence_time,
                    bet_type, bet_side, odds, line,
                    model_prob, market_prob, edge,
                    bookmaker, logged_at,
                    outcome, actual_score_home, actual_score_away,
                    profit, settled_at, bet_amount, actual_margin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet_id,
                bet['game_id'],
                bet['home_team'],
                bet['away_team'],
                commence_time,
                bet['bet_type'],
                bet['bet_side'],
                float(bet['odds']) if pd.notna(bet['odds']) else None,
                float(bet['line']) if pd.notna(bet['line']) else None,
                float(bet['model_prob']) if pd.notna(bet['model_prob']) else None,
                float(bet['market_prob']) if pd.notna(bet['market_prob']) else None,
                float(bet['edge']) if pd.notna(bet['edge']) else None,
                bet['bookmaker'],
                logged_at,
                bet.get('outcome'),
                int(bet.get('actual_score_home')) if pd.notna(bet.get('actual_score_home')) else None,
                int(bet.get('actual_score_away')) if pd.notna(bet.get('actual_score_away')) else None,
                float(bet.get('profit')) if pd.notna(bet.get('profit')) else None,
                settled_at,
                float(bet.get('bet_amount', 100)),
                float(bet.get('actual_margin')) if pd.notna(bet.get('actual_margin')) else None,
            ))

            inserted += 1

        conn.commit()
        logger.success(f"Inserted {inserted} synthetic bets into database")

    finally:
        conn.close()


def clear_synthetic_bets():
    """Remove existing synthetic bets from database."""
    logger.info("Clearing existing synthetic bets...")

    conn = sqlite3.connect(DB_PATH)

    try:
        result = conn.execute("DELETE FROM bets WHERE id LIKE 'synthetic_%'")
        deleted = result.rowcount
        conn.commit()

        logger.success(f"Deleted {deleted} synthetic bets")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic historical bets')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 5%%)')
    parser.add_argument('--model-accuracy', type=float, default=0.55,
                       help='Simulated model accuracy (default: 55%%)')
    parser.add_argument('--strategy', type=str, default='team_filtered',
                       choices=['primary', 'team_filtered', 'aggressive'],
                       help='Strategy to simulate (default: team_filtered)')
    parser.add_argument('--days', type=int,
                       help='Number of days to generate (default: all available)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--clear-existing', action='store_true',
                       help='Clear existing synthetic bets before generating')
    parser.add_argument('--skip-settlement', action='store_true',
                       help='Skip fetching results and settling bets')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SYNTHETIC HISTORICAL BET GENERATOR")
    logger.info("=" * 80)

    # Clear existing if requested
    if args.clear_existing:
        clear_synthetic_bets()

    # Determine date range
    if args.days:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = args.start_date
        end_date = args.end_date

    # Load historical odds
    odds_df = load_historical_odds(start_date, end_date)

    if odds_df.empty:
        logger.error("No historical odds available")
        return 1

    # Generate bets
    bets_df = generate_bets_from_odds(
        odds_df,
        min_edge=args.min_edge,
        model_accuracy=args.model_accuracy,
        strategy=args.strategy
    )

    if bets_df.empty:
        logger.warning("No bets generated with current filters")
        return 1

    # Settle bets with results (unless skipped)
    if not args.skip_settlement:
        unique_games = bets_df['game_id'].unique().tolist()
        results_df = get_game_results(unique_games)
        bets_df = settle_bets_with_results(bets_df, results_df)

    # Insert into database
    insert_bets_to_database(bets_df)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Generated: {len(bets_df)} bets")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Min Edge: {args.min_edge:.1%}")

    if not args.skip_settlement:
        settled = len(bets_df[bets_df['outcome'].notna()])
        logger.info(f"Settled: {settled} bets")

        if settled > 0:
            wins = len(bets_df[bets_df['outcome'] == 'win'])
            losses = len(bets_df[bets_df['outcome'] == 'loss'])
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            total_profit = bets_df['profit'].sum()
            roi = (total_profit / (len(bets_df) * 100)) * 100

            logger.info(f"Win Rate: {win_rate:.1%} ({wins}W-{losses}L)")
            logger.info(f"Total Profit: ${total_profit:,.2f}")
            logger.info(f"ROI: {roi:+.1f}%")

    logger.info("\nBets inserted with ID prefix: 'synthetic_'")
    logger.info("View in dashboard: streamlit run analytics_dashboard.py")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
