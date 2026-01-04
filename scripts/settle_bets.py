#!/usr/bin/env python3
"""
Settle Pending Bets

Fetches actual game results and settles pending bets.
Automatically updates bankroll when bets are settled.

Usage:
    # Settle all pending bets
    python scripts/settle_bets.py

    # Dry run (show what would be settled)
    python scripts/settle_bets.py --dry-run

    # Settle specific game
    python scripts/settle_bets.py --game-id <game_id>
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger

from src.bet_tracker import DB_PATH
from src.bankroll.bankroll_manager import BankrollManager
from src.data import NBAStatsClient


def get_pending_bets(game_id: Optional[str] = None) -> List[Dict]:
    """
    Get all pending bets (bets without outcomes).

    Args:
        game_id: Optional game_id to filter

    Returns:
        List of pending bet records
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT *
        FROM bets
        WHERE outcome IS NULL
        AND commence_time < datetime('now')
    """

    params = []
    if game_id:
        query += " AND game_id = ?"
        params.append(game_id)

    query += " ORDER BY commence_time ASC"

    pending = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(bet) for bet in pending]


def get_game_result(home_team: str, away_team: str, commence_time: str) -> Optional[Dict]:
    """
    Get actual game result from NBA Stats API.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        commence_time: Game start time (ISO format)

    Returns:
        Dict with game result: {home_score, away_score, home_win}
        None if game not found or not finished
    """
    try:
        stats_client = NBAStatsClient()

        # Parse commence time
        game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))

        # Search for game within ±1 day window
        start_date = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (game_date + timedelta(days=1)).strftime('%Y-%m-%d')

        games = stats_client.get_games(start_date, end_date)

        if games.empty:
            logger.debug(f"No games found for {home_team} vs {away_team} on {game_date.date()}")
            return None

        # Find matching game
        game = games[
            (games['home_team'] == home_team) &
            (games['away_team'] == away_team)
        ]

        if game.empty:
            logger.debug(f"Game not found: {away_team} @ {home_team}")
            return None

        game_row = game.iloc[0]

        # Check if game is finished
        if 'home_score' not in game_row or game_row['home_score'] is None:
            logger.debug(f"Game not finished yet: {away_team} @ {home_team}")
            return None

        return {
            'home_score': float(game_row['home_score']),
            'away_score': float(game_row['away_score']),
            'home_win': game_row['home_score'] > game_row['away_score'],
            'final_margin': game_row['home_score'] - game_row['away_score'],
        }

    except Exception as e:
        logger.error(f"Error fetching game result: {e}")
        return None


def calculate_bet_outcome(bet: Dict, game_result: Dict) -> Dict:
    """
    Calculate bet outcome based on game result.

    Args:
        bet: Bet record dict
        game_result: Game result dict

    Returns:
        Dict with outcome, profit, actual_margin
    """
    bet_side = bet['bet_side']
    line = float(bet['line'])
    odds = float(bet['odds'])
    bet_amount = float(bet['bet_amount']) if bet['bet_amount'] else 100.0

    actual_margin = game_result['final_margin']

    # Determine if bet covers the spread
    if bet_side == 'home':
        # Home bet: need home_score + line > away_score
        # Equivalent: actual_margin > -line
        covers = actual_margin > -line
    else:
        # Away bet: need away_score + abs(line) > home_score
        # Equivalent: actual_margin < line
        covers = actual_margin < line

    # Check for push
    if abs(actual_margin + line) < 0.01:  # Push (margin exactly equals line)
        outcome = 'push'
        profit = 0.0
    elif covers:
        outcome = 'win'
        # Calculate profit based on American odds
        if odds > 0:
            profit = bet_amount * (odds / 100)
        else:
            profit = bet_amount * (100 / abs(odds))
    else:
        outcome = 'loss'
        profit = -bet_amount

    return {
        'outcome': outcome,
        'profit': profit,
        'actual_margin': actual_margin,
    }


def settle_bet(
    bet: Dict,
    outcome: str,
    profit: float,
    actual_margin: float,
    dry_run: bool = False
) -> bool:
    """
    Settle a bet by updating the database and bankroll.

    Args:
        bet: Bet record dict
        outcome: 'win', 'loss', or 'push'
        profit: Profit amount
        actual_margin: Actual game margin
        dry_run: If True, don't actually update database

    Returns:
        True if successful
    """
    bet_id = bet['id']

    if dry_run:
        logger.info(f"[DRY RUN] Would settle bet {bet_id}: {outcome} ({profit:+.2f})")
        return True

    try:
        conn = sqlite3.connect(DB_PATH)

        # Update bet record
        conn.execute("""
            UPDATE bets
            SET outcome = ?,
                profit = ?,
                actual_margin = ?,
                settled_at = ?
            WHERE id = ?
        """, (outcome, profit, actual_margin, datetime.now().isoformat(), bet_id))

        conn.commit()
        conn.close()

        # Update bankroll
        bankroll_mgr = BankrollManager()
        bankroll_mgr.record_bet_outcome(
            bet_id=bet_id,
            profit=profit,
            notes=f"Auto-settled: {outcome}"
        )

        logger.info(f"✓ Settled bet {bet_id}: {outcome} ({profit:+.2f})")
        return True

    except Exception as e:
        logger.error(f"Error settling bet {bet_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Settle pending bets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be settled without updating')
    parser.add_argument('--game-id', type=str, help='Settle specific game only')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🏁 BET SETTLEMENT")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE - No changes will be made")
    logger.info("")

    # Get pending bets
    pending_bets = get_pending_bets(game_id=args.game_id)

    if not pending_bets:
        logger.info("✓ No pending bets to settle")
        return 0

    logger.info(f"Found {len(pending_bets)} pending bets")
    logger.info("")

    settled_count = 0
    not_finished_count = 0
    error_count = 0

    for bet in pending_bets:
        logger.info(f"Processing: {bet['away_team']} @ {bet['home_team']} ({bet['bet_side']})")

        # Get game result
        game_result = get_game_result(
            bet['home_team'],
            bet['away_team'],
            bet['commence_time']
        )

        if not game_result:
            logger.warning(f"  ⏳ Game not finished or not found - skipping")
            not_finished_count += 1
            continue

        # Calculate outcome
        bet_outcome = calculate_bet_outcome(bet, game_result)

        logger.info(f"  Final: {game_result['home_score']:.0f} - {game_result['away_score']:.0f} "
                   f"(margin: {game_result['final_margin']:+.1f})")
        logger.info(f"  Result: {bet_outcome['outcome'].upper()} ({bet_outcome['profit']:+.2f})")

        # Settle bet
        success = settle_bet(
            bet,
            bet_outcome['outcome'],
            bet_outcome['profit'],
            bet_outcome['actual_margin'],
            dry_run=args.dry_run
        )

        if success:
            settled_count += 1
        else:
            error_count += 1

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("📊 SETTLEMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total pending: {len(pending_bets)}")
    logger.info(f"Settled: {settled_count}")
    logger.info(f"Not finished: {not_finished_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("")

    if not args.dry_run and settled_count > 0:
        # Show updated bankroll
        bankroll_mgr = BankrollManager()
        stats = bankroll_mgr.get_bankroll_stats()
        logger.info("💰 Updated Bankroll:")
        logger.info(f"   Current: ${stats['current']:,.2f}")
        logger.info(f"   Profit:  ${stats['total_profit']:+,.2f}")
        logger.info(f"   ROI:     {stats['roi']:.2f}%")
        logger.info("")

    logger.info("=" * 80)
    logger.info("✅ Settlement complete")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Settlement failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
