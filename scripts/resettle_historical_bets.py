"""
Re-settle Historical Paper Trading Bets

Critical script to fix historical bet outcomes after spread coverage bug fix.

The spread coverage bug caused incorrect win/loss determinations for spread bets.
This script:
1. Loads all settled spread bets from database
2. Re-calculates outcomes using CORRECT spread coverage logic
3. Updates outcomes and profits in database
4. Generates before/after comparison report

Date: January 4, 2026
Reason: Spread bug discovered - see docs/BACKTEST_BUG_FINDINGS.md
"""

import sys
sys.path.insert(0, '.')

import sqlite3
import pandas as pd
from datetime import datetime
from loguru import logger

DB_PATH = "data/bets/bets.db"


def get_settled_spread_bets():
    """Load all settled spread bets from database and join with game scores."""
    import os

    conn = sqlite3.connect(DB_PATH)

    # First, try to load bets with scores already populated
    query = """
        SELECT
            id,
            game_id,
            home_team,
            away_team,
            bet_type,
            bet_side,
            line,
            odds,
            bet_amount,
            outcome,
            actual_score_home,
            actual_score_away,
            profit,
            settled_at
        FROM bets
        WHERE bet_type = 'spread'
          AND outcome IS NOT NULL
        ORDER BY settled_at
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} settled spread bets from database")

    # Check if scores are missing
    missing_scores = df['actual_score_home'].isna().sum()
    if missing_scores > 0:
        logger.warning(f"  {missing_scores} bets missing actual scores")

        # Try to load games.parquet to get scores
        games_path = 'data/raw/games.parquet'
        if os.path.exists(games_path):
            logger.info(f"  Loading game scores from {games_path}...")
            games_df = pd.read_parquet(games_path)

            # Create game_id mapping if needed
            if 'GAME_ID' in games_df.columns:
                games_df = games_df.rename(columns={'GAME_ID': 'game_id'})

            # Merge scores
            for idx, row in df[df['actual_score_home'].isna()].iterrows():
                # Try to find game by matching home/away teams
                game_match = games_df[
                    (games_df['home_team_abbrev'] == row['home_team']) &
                    (games_df['away_team_abbrev'] == row['away_team'])
                ]

                if len(game_match) > 0:
                    game = game_match.iloc[0]
                    df.at[idx, 'actual_score_home'] = game.get('home_score', game.get('PTS_home', None))
                    df.at[idx, 'actual_score_away'] = game.get('away_score', game.get('PTS_away', None))

            filled = (~df['actual_score_home'].isna()).sum()
            logger.info(f"  Filled scores for {filled}/{len(df)} bets")
        else:
            logger.warning(f"  Games file not found at {games_path}")
            logger.warning("  Cannot re-settle bets without actual scores")
            logger.warning("  Skipping bets without scores...")
            df = df[~df['actual_score_home'].isna()]

    # Filter to only bets with scores
    df = df[~df['actual_score_home'].isna()].copy()
    logger.info(f"Re-settling {len(df)} bets with actual scores")

    return df


def calculate_correct_outcome(row):
    """
    Calculate correct spread bet outcome using fixed logic.

    Spread coverage logic:
    - spread is from home team perspective
    - Home covers if: point_diff + spread > 0
    - Away covers if: point_diff + spread < 0
    - Push if: abs(point_diff + spread) < 0.1

    Args:
        row: DataFrame row with bet details

    Returns:
        tuple: (outcome, profit)
    """
    home_score = row['actual_score_home']
    away_score = row['actual_score_away']
    point_diff = home_score - away_score
    spread = row['line']
    bet_side = row['bet_side']
    bet_amount = row['bet_amount']
    odds = row['odds']

    # Calculate spread result
    spread_result = point_diff + spread

    # Determine outcome
    if abs(spread_result) < 0.1:
        outcome = 'push'
        profit = 0.0
    elif bet_side == 'home':
        # Home covers if spread_result > 0
        if spread_result > 0:
            outcome = 'win'
            # Convert American odds to decimal if needed
            if odds < 0:
                decimal_odds = 1 + (100 / abs(odds))
            else:
                decimal_odds = 1 + (odds / 100)
            profit = bet_amount * (decimal_odds - 1)
        else:
            outcome = 'loss'
            profit = -bet_amount
    elif bet_side == 'away':
        # Away covers if spread_result < 0
        if spread_result < 0:
            outcome = 'win'
            # Convert American odds to decimal if needed
            if odds < 0:
                decimal_odds = 1 + (100 / abs(odds))
            else:
                decimal_odds = 1 + (odds / 100)
            profit = bet_amount * (decimal_odds - 1)
        else:
            outcome = 'loss'
            profit = -bet_amount
    else:
        logger.error(f"Unknown bet_side: {bet_side} for bet {row['id']}")
        return None, None

    return outcome, profit


def resettle_bets(dry_run=True):
    """
    Re-settle all historical spread bets with correct logic.

    Args:
        dry_run: If True, only show what would change without updating DB

    Returns:
        DataFrame with comparison of old vs new outcomes
    """
    logger.info("=" * 70)
    logger.info("RE-SETTLING HISTORICAL PAPER TRADING BETS")
    logger.info("=" * 70)

    # Load all settled spread bets
    bets_df = get_settled_spread_bets()

    if len(bets_df) == 0:
        logger.warning("No settled spread bets found in database")
        return pd.DataFrame()

    # Calculate correct outcomes
    logger.info("Calculating correct outcomes with fixed spread logic...")

    results = []
    changes = 0

    for idx, row in bets_df.iterrows():
        old_outcome = row['outcome']
        old_profit = row['profit']

        new_outcome, new_profit = calculate_correct_outcome(row)

        if new_outcome is None:
            continue

        # Check if outcome changed
        changed = (old_outcome != new_outcome)
        if changed:
            changes += 1

        results.append({
            'id': row['id'],
            'game_id': row['game_id'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'bet_side': row['bet_side'],
            'line': row['line'],
            'home_score': row['actual_score_home'],
            'away_score': row['actual_score_away'],
            'point_diff': row['actual_score_home'] - row['actual_score_away'],
            'old_outcome': old_outcome,
            'new_outcome': new_outcome,
            'old_profit': old_profit,
            'new_profit': new_profit,
            'changed': changed,
            'profit_diff': new_profit - old_profit if old_profit is not None else new_profit
        })

    results_df = pd.DataFrame(results)

    # Report summary
    logger.info("\n" + "=" * 70)
    logger.info("RE-SETTLEMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total spread bets: {len(results_df)}")
    logger.info(f"Outcomes changed: {changes} ({changes/len(results_df)*100:.1f}%)")

    if len(results_df) > 0:
        # Old performance
        old_wins = (results_df['old_outcome'] == 'win').sum()
        old_losses = (results_df['old_outcome'] == 'loss').sum()
        old_pushes = (results_df['old_outcome'] == 'push').sum()
        old_total_profit = results_df['old_profit'].sum()

        # New performance
        new_wins = (results_df['new_outcome'] == 'win').sum()
        new_losses = (results_df['new_outcome'] == 'loss').sum()
        new_pushes = (results_df['new_outcome'] == 'push').sum()
        new_total_profit = results_df['new_profit'].sum()

        logger.info("\nOLD (BUGGY) RESULTS:")
        logger.info(f"  Wins: {old_wins} ({old_wins/(old_wins+old_losses)*100:.1f}%)")
        logger.info(f"  Losses: {old_losses}")
        logger.info(f"  Pushes: {old_pushes}")
        logger.info(f"  Total Profit: ${old_total_profit:.2f}")

        logger.info("\nNEW (CORRECT) RESULTS:")
        logger.info(f"  Wins: {new_wins} ({new_wins/(new_wins+new_losses)*100:.1f}%)")
        logger.info(f"  Losses: {new_losses}")
        logger.info(f"  Pushes: {new_pushes}")
        logger.info(f"  Total Profit: ${new_total_profit:.2f}")

        logger.info("\nIMPACT:")
        logger.info(f"  Win rate change: {(new_wins/(new_wins+new_losses) - old_wins/(old_wins+old_losses))*100:+.1f}pp")
        logger.info(f"  Profit change: ${new_total_profit - old_total_profit:+.2f}")

    # Show examples of changes
    if changes > 0:
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLES OF CHANGED OUTCOMES (first 10):")
        logger.info("=" * 70)

        changed_bets = results_df[results_df['changed']].head(10)
        for idx, bet in changed_bets.iterrows():
            logger.info(f"\nBet ID: {bet['id']}")
            logger.info(f"  Game: {bet['away_team']} @ {bet['home_team']}")
            logger.info(f"  Bet: {bet['bet_side']} {bet['line']:+.1f}")
            logger.info(f"  Score: {bet['away_team']} {bet['away_score']} - {bet['home_team']} {bet['home_score']}")
            logger.info(f"  Point Diff: {bet['point_diff']:+.1f}")
            logger.info(f"  Spread Result: {bet['point_diff'] + bet['line']:+.1f}")
            logger.info(f"  OLD: {bet['old_outcome']} (${bet['old_profit']:+.2f})")
            logger.info(f"  NEW: {bet['new_outcome']} (${bet['new_profit']:+.2f})")

    # Update database if not dry run
    if not dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("UPDATING DATABASE...")
        logger.info("=" * 70)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        updates = 0
        for idx, row in results_df.iterrows():
            cursor.execute("""
                UPDATE bets
                SET outcome = ?,
                    profit = ?,
                    settled_at = ?
                WHERE id = ?
            """, (
                row['new_outcome'],
                row['new_profit'],
                datetime.now().isoformat(),
                row['id']
            ))
            updates += 1

        conn.commit()
        conn.close()

        logger.info(f"✅ Updated {updates} bets in database")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("DRY RUN - No changes made to database")
        logger.info("Run with --commit flag to apply changes")
        logger.info("=" * 70)

    return results_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-settle historical paper trading bets with corrected spread logic"
    )
    parser.add_argument(
        '--commit',
        action='store_true',
        help='Actually update database (default is dry-run)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )

    args = parser.parse_args()

    # Run re-settlement
    results_df = resettle_bets(dry_run=not args.commit)

    # Export if requested
    if args.export and len(results_df) > 0:
        results_df.to_csv(args.export, index=False)
        logger.info(f"\n✅ Exported results to {args.export}")

    logger.info("\n" + "=" * 70)
    logger.info("RE-SETTLEMENT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
