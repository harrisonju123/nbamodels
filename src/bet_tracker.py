"""
Bet Tracking System

Logs recommended bets, tracks outcomes, and calculates performance metrics.
Uses SQLite for persistent storage.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger


DB_PATH = "data/bets/bets.db"


def _get_connection() -> sqlite3.Connection:
    """Get database connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT,
            bet_type TEXT NOT NULL,
            bet_side TEXT NOT NULL,
            odds REAL,
            line REAL,
            bet_amount REAL,
            model_prob REAL,
            market_prob REAL,
            edge REAL,
            kelly REAL,
            bookmaker TEXT,
            logged_at TEXT NOT NULL,
            outcome TEXT,
            actual_score_home INTEGER,
            actual_score_away INTEGER,
            profit REAL,
            settled_at TEXT,
            closing_odds REAL,
            closing_line REAL,
            clv REAL,
            clv_updated_at TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_game_id ON bets(game_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_outcome ON bets(outcome)
    """)
    # Additional indexes for common query patterns
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_logged_at ON bets(logged_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_settled_at ON bets(settled_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_bookmaker ON bets(bookmaker)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_bet_type ON bets(bet_type)
    """)

    # Create line_snapshots table for hourly odds tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS line_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            snapshot_time TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            side TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            odds INTEGER,
            line REAL,
            implied_prob REAL,
            no_vig_prob REAL,
            UNIQUE(game_id, snapshot_time, bet_type, side, bookmaker)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_game ON line_snapshots(game_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_time ON line_snapshots(snapshot_time)
    """)
    # Composite index for common lookup pattern (multi-snapshot CLV queries)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
        ON line_snapshots(game_id, bet_type, side, snapshot_time)
    """)
    # Additional indexes for line movement queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_game_time
        ON line_snapshots(game_id, snapshot_time DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_bookmaker
        ON line_snapshots(game_id, bookmaker, snapshot_time DESC)
    """)

    # Create opening_lines table for tracking opening lines
    conn.execute("""
        CREATE TABLE IF NOT EXISTS opening_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            side TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            opening_odds INTEGER,
            opening_line REAL,
            opening_implied_prob REAL,
            opening_no_vig_prob REAL,
            commence_time TEXT,
            is_true_opener BOOLEAN DEFAULT 0,
            UNIQUE(game_id, bet_type, side, bookmaker)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_opening_game ON opening_lines(game_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_opening_lookup
        ON opening_lines(game_id, bet_type, side, bookmaker)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_opening_time ON opening_lines(first_seen_at)
    """)

    # Migrate existing tables: add new columns if missing
    cursor = conn.execute("PRAGMA table_info(bets)")
    columns = {row[1] for row in cursor.fetchall()}

    if "bet_amount" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN bet_amount REAL")
    if "closing_odds" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN closing_odds REAL")
    if "closing_line" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN closing_line REAL")
    if "clv" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv REAL")
    if "clv_updated_at" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv_updated_at TEXT")

    # Multi-snapshot CLV columns
    if "booked_hours_before" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN booked_hours_before REAL")
    if "optimal_book_time" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN optimal_book_time REAL")
    if "clv_at_1hr" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv_at_1hr REAL")
    if "clv_at_4hr" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv_at_4hr REAL")
    if "clv_at_12hr" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv_at_12hr REAL")
    if "clv_at_24hr" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN clv_at_24hr REAL")
    if "line_velocity" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN line_velocity REAL")
    if "max_clv_achieved" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN max_clv_achieved REAL")

    # Opening line tracking columns
    if "opening_odds" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN opening_odds REAL")
    if "opening_line" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN opening_line REAL")
    if "opening_line_captured_at" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN opening_line_captured_at TEXT")

    # Closing line automation columns
    if "provisional_closing_odds" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN provisional_closing_odds REAL")
    if "provisional_closing_line" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN provisional_closing_line REAL")
    if "closing_line_source" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN closing_line_source TEXT")
    if "snapshot_coverage" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN snapshot_coverage REAL")

    # Multi-strategy framework columns
    if "strategy_type" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN strategy_type TEXT")
    if "player_id" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN player_id TEXT")
    if "player_name" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN player_name TEXT")
    if "prop_type" not in columns:
        conn.execute("ALTER TABLE bets ADD COLUMN prop_type TEXT")

    # Add index for strategy queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bets_strategy ON bets(strategy_type)
    """)

    conn.commit()
    return conn


def log_bet(
    game_id: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    bet_type: str,  # 'moneyline', 'spread', 'totals', 'player_prop'
    bet_side: str,  # 'home', 'away', 'over', 'under', 'cover'
    odds: float,
    line: Optional[float],  # spread line or total line
    model_prob: float,
    market_prob: float,
    edge: float,
    kelly: float,
    bookmaker: str = None,
    bet_amount: Optional[float] = None,
    strategy_type: Optional[str] = None,  # NEW: Strategy that generated the bet
    player_id: Optional[str] = None,  # NEW: For player props
    player_name: Optional[str] = None,  # NEW: For player props
    prop_type: Optional[str] = None,  # NEW: For player props (PTS, REB, AST, etc.)
) -> Dict:
    """
    Log a new bet recommendation.

    Args:
        strategy_type: Strategy that generated bet (spread, totals, live, arbitrage, player_props)
        player_id: Player ID for prop bets
        player_name: Player name for prop bets
        prop_type: Prop type for player props (PTS, REB, AST, 3PM, STL, BLK)

    Returns the logged bet record.
    """
    conn = _get_connection()
    bet_id = f"{game_id}_{bet_type}_{bet_side}"

    logged_at = datetime.now().isoformat()

    # ALWAYS calculate Kelly percentage (for display and analysis)
    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
    b = decimal_odds - 1
    if b > 0 and model_prob > 0:
        # Full Kelly percentage
        kelly_pct = (model_prob * b - (1 - model_prob)) / b
        kelly_pct = max(0, kelly_pct)  # Don't bet on negative Kelly
    else:
        kelly_pct = 0

    # Calculate bet amount using Kelly if not provided
    if bet_amount is None:
        kelly_fraction = 0.10  # Use 10% fractional Kelly for safety
        bankroll = 1000.0  # Default paper trading bankroll
        bet_amount = max(10.0, min(50.0, kelly_pct * kelly_fraction * bankroll))  # Min $10, max $50

    # Use INSERT OR IGNORE to prevent race condition duplicates
    cursor = conn.execute("""
        INSERT OR IGNORE INTO bets (
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line, model_prob, market_prob,
            edge, kelly, bookmaker, logged_at, bet_amount,
            strategy_type, player_id, player_name, prop_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet_id, game_id, home_team, away_team, str(commence_time),
        bet_type, bet_side, odds, line, model_prob, market_prob,
        edge, kelly_pct, bookmaker, logged_at, bet_amount,
        strategy_type, player_id, player_name, prop_type
    ))

    conn.commit()

    result = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    conn.close()

    logger.info(f"Logged bet: {away_team} @ {home_team} - {bet_type} {bet_side} at {odds:+.0f} (${bet_amount:.2f})")
    return dict(result)


def log_manual_bet(
    game_id: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    bet_type: str,  # 'moneyline', 'spread', 'totals'
    bet_side: str,  # 'home', 'away', 'over', 'under'
    odds: float,  # American odds (e.g., -110, +150)
    bet_amount: float,  # Actual dollar amount bet
    line: Optional[float] = None,  # spread line or total line
    bookmaker: str = None,
    model_prob: Optional[float] = None,
) -> Dict:
    """
    Log a manually placed bet with actual odds and amount.

    Args:
        game_id: Unique game identifier
        home_team: Home team name
        away_team: Away team name
        commence_time: Game start time
        bet_type: 'moneyline', 'spread', or 'totals'
        bet_side: 'home', 'away', 'over', 'under'
        odds: American odds you got (e.g., -110, +150)
        bet_amount: Dollar amount wagered
        line: Spread or total line (if applicable)
        bookmaker: Sportsbook name (optional)
        model_prob: Model's predicted probability (optional)

    Returns the logged bet record.
    """
    conn = _get_connection()

    # Generate unique ID with timestamp to allow multiple bets on same game
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bet_id = f"{game_id}_{bet_type}_{bet_side}_{timestamp}"

    # Calculate implied probability from odds
    if odds > 0:
        market_prob = 100 / (odds + 100)
        decimal_odds = (odds / 100) + 1
    else:
        market_prob = abs(odds) / (abs(odds) + 100)
        decimal_odds = (100 / abs(odds)) + 1

    # Calculate edge if model_prob provided
    edge = (model_prob - market_prob) if model_prob else None

    # Calculate Kelly percentage if model_prob provided
    kelly_pct = None
    if model_prob:
        b = decimal_odds - 1
        if b > 0 and model_prob > 0:
            kelly_pct = (model_prob * b - (1 - model_prob)) / b
            kelly_pct = max(0, kelly_pct)

    logged_at = datetime.now().isoformat()

    # Use INSERT OR IGNORE to prevent race condition duplicates
    cursor = conn.execute("""
        INSERT OR IGNORE INTO bets (
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line, bet_amount,
            model_prob, market_prob, edge, kelly, bookmaker, logged_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet_id, game_id, home_team, away_team, str(commence_time),
        bet_type, bet_side, odds, line, bet_amount,
        model_prob, market_prob, edge, kelly_pct, bookmaker, logged_at
    ))

    conn.commit()

    result = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    conn.close()

    logger.info(f"Logged manual bet: {away_team} @ {home_team} - {bet_type} {bet_side} ${bet_amount:.0f} at {odds:+.0f}")
    return dict(result)


def log_bets_from_predictions(predictions: Dict[str, pd.DataFrame]) -> int:
    """
    Log all recommended bets from a predictions dictionary.

    Returns count of new bets logged.
    """
    count = 0

    # Moneyline bets
    if "moneyline" in predictions and not predictions["moneyline"].empty:
        df = predictions["moneyline"]
        for _, row in df.iterrows():
            if row.get("bet_home"):
                log_bet(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=row["commence_time"],
                    bet_type="moneyline",
                    bet_side="home",
                    odds=row["best_home_odds"],
                    line=None,
                    model_prob=row["model_home_prob"],
                    market_prob=row["market_home_prob"],
                    edge=row["home_edge"],
                    kelly=row["home_kelly"],
                    bookmaker=row.get("home_book"),
                )
                count += 1
            elif row.get("bet_away"):
                log_bet(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=row["commence_time"],
                    bet_type="moneyline",
                    bet_side="away",
                    odds=row["best_away_odds"],
                    line=None,
                    model_prob=row["model_away_prob"],
                    market_prob=row["market_away_prob"],
                    edge=row["away_edge"],
                    kelly=row["away_kelly"],
                    bookmaker=row.get("away_book"),
                )
                count += 1

    # Spread bets
    if "spread" in predictions and not predictions["spread"].empty:
        df = predictions["spread"]
        for _, row in df.iterrows():
            if row.get("bet_cover"):
                log_bet(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=row["commence_time"],
                    bet_type="spread",
                    bet_side="cover",
                    odds=row["best_spread_odds"],
                    line=row["spread_line"],
                    model_prob=row["model_cover_prob"],
                    market_prob=row["market_cover_prob"],
                    edge=row["cover_edge"],
                    kelly=row["kelly"],
                    bookmaker=row.get("spread_book"),
                )
                count += 1

    # Totals bets
    if "totals" in predictions and not predictions["totals"].empty:
        df = predictions["totals"]
        for _, row in df.iterrows():
            if row.get("bet_over"):
                log_bet(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=row["commence_time"],
                    bet_type="totals",
                    bet_side="over",
                    odds=row["best_over_odds"],
                    line=row["total_line"],
                    model_prob=row["model_over_prob"],
                    market_prob=row["market_over_prob"],
                    edge=row["over_edge"],
                    kelly=row["over_kelly"],
                    bookmaker=row.get("over_book"),
                )
                count += 1
            elif row.get("bet_under"):
                log_bet(
                    game_id=row["game_id"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=row["commence_time"],
                    bet_type="totals",
                    bet_side="under",
                    odds=row["best_under_odds"],
                    line=row["total_line"],
                    model_prob=row["model_under_prob"],
                    market_prob=row["market_under_prob"],
                    edge=row["under_edge"],
                    kelly=row["under_kelly"],
                    bookmaker=row.get("under_book"),
                )
                count += 1

    logger.info(f"Logged {count} new bets")
    return count


def settle_bet(
    bet_id: str,
    home_score: int,
    away_score: int,
) -> Optional[Dict]:
    """
    Settle a bet based on final scores.

    Returns the updated bet record.
    """
    conn = _get_connection()

    bet = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()

    if not bet:
        logger.warning(f"Bet not found: {bet_id}")
        conn.close()
        return None

    bet = dict(bet)

    if bet["outcome"] is not None:
        logger.info(f"Bet already settled: {bet_id}")
        conn.close()
        return bet

    # Determine outcome
    point_diff = home_score - away_score
    total_points = home_score + away_score

    if bet["bet_type"] == "moneyline":
        if bet["bet_side"] == "home":
            outcome = "win" if point_diff > 0 else "loss"
        else:  # away
            outcome = "win" if point_diff < 0 else "loss"

    elif bet["bet_type"] == "spread":
        # Spread coverage logic:
        # spread is from home team perspective
        # Home covers if: point_diff + spread > 0
        # Away covers if: point_diff + spread < 0
        # Push if: point_diff + spread == 0 (within tolerance)

        spread = bet["line"]
        spread_result = point_diff + spread

        # Check for push first (within 0.1 point tolerance)
        if abs(spread_result) < 0.1:
            outcome = "push"
        elif bet["bet_side"] == "home":
            # Home covers if spread_result > 0
            outcome = "win" if spread_result > 0 else "loss"
        elif bet["bet_side"] == "away":
            # Away covers if spread_result < 0
            outcome = "win" if spread_result < 0 else "loss"
        else:
            # Invalid bet_side - this is a critical error
            logger.error(f"Invalid bet_side for spread bet: {bet['bet_side']} (bet_id: {bet_id})")
            conn.close()
            raise ValueError(f"Invalid bet_side '{bet['bet_side']}' for spread bet {bet_id}. Must be 'home' or 'away'.")

    elif bet["bet_type"] == "totals":
        line = bet["line"]
        if bet["bet_side"] == "over":
            if total_points > line:
                outcome = "win"
            elif total_points < line:
                outcome = "loss"
            else:
                outcome = "push"
        elif bet["bet_side"] == "under":
            if total_points < line:
                outcome = "win"
            elif total_points > line:
                outcome = "loss"
            else:
                outcome = "push"
        else:
            # Invalid bet_side for totals
            logger.error(f"Invalid bet_side for totals bet: {bet['bet_side']} (bet_id: {bet_id})")
            conn.close()
            raise ValueError(f"Invalid bet_side '{bet['bet_side']}' for totals bet {bet_id}. Must be 'over' or 'under'.")
    else:
        # Unknown bet_type
        logger.error(f"Unknown bet_type: {bet['bet_type']} (bet_id: {bet_id})")
        conn.close()
        raise ValueError(f"Unknown bet_type '{bet['bet_type']}' for bet {bet_id}. Must be 'spread' or 'totals'.")

    # Calculate profit using actual bet amount (default $100 if not set)
    wager = bet.get("bet_amount") or 100.0

    if outcome == "win":
        odds = bet["odds"]
        if odds > 0:
            profit = wager * (odds / 100)
        else:
            profit = wager * (100 / abs(odds))
    elif outcome == "loss":
        profit = -wager
    else:  # push
        profit = 0

    settled_at = datetime.now().isoformat()

    try:
        conn.execute("""
            UPDATE bets SET
                outcome = ?,
                actual_score_home = ?,
                actual_score_away = ?,
                profit = ?,
                settled_at = ?
            WHERE id = ?
        """, (outcome, home_score, away_score, profit, settled_at, bet_id))

        conn.commit()

        result = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    except Exception as e:
        logger.error(f"Failed to settle bet {bet_id}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(f"Settled bet: {bet_id} -> {outcome} (${profit:+.2f})")
    return dict(result)


def _calculate_bet_outcome(bet: Dict, home_score: int, away_score: int) -> tuple:
    """
    Calculate bet outcome without database access (for batch processing).

    Args:
        bet: Bet dictionary with bet_type, bet_side, line, odds, bet_amount
        home_score: Final home team score
        away_score: Final away team score

    Returns:
        Tuple of (outcome, profit)
    """
    point_diff = home_score - away_score
    total_points = home_score + away_score

    # Determine outcome based on bet type
    if bet["bet_type"] == "moneyline":
        if bet["bet_side"] == "home":
            outcome = "win" if point_diff > 0 else "loss"
        else:  # away
            outcome = "win" if point_diff < 0 else "loss"

    elif bet["bet_type"] == "spread":
        spread = bet["line"]
        spread_result = point_diff + spread

        # Check for push (within 0.1 point tolerance)
        if abs(spread_result) < 0.1:
            outcome = "push"
        elif bet["bet_side"] == "home":
            outcome = "win" if spread_result > 0 else "loss"
        elif bet["bet_side"] == "away":
            outcome = "win" if spread_result < 0 else "loss"
        else:
            raise ValueError(f"Invalid bet_side '{bet['bet_side']}' for spread bet")

    elif bet["bet_type"] == "totals":
        line = bet["line"]
        if bet["bet_side"] == "over":
            if total_points > line:
                outcome = "win"
            elif total_points < line:
                outcome = "loss"
            else:
                outcome = "push"
        elif bet["bet_side"] == "under":
            if total_points < line:
                outcome = "win"
            elif total_points > line:
                outcome = "loss"
            else:
                outcome = "push"
        else:
            raise ValueError(f"Invalid bet_side '{bet['bet_side']}' for totals bet")
    else:
        raise ValueError(f"Unknown bet_type '{bet['bet_type']}'")

    # Calculate profit
    wager = bet.get("bet_amount") or 100.0

    if outcome == "win":
        odds = bet["odds"]
        if odds > 0:
            profit = wager * (odds / 100)
        else:
            profit = wager * (100 / abs(odds))
    elif outcome == "loss":
        profit = -wager
    else:  # push
        profit = 0

    return outcome, profit


def settle_all_pending(games_df: pd.DataFrame) -> int:
    """
    Settle all pending bets using game results (optimized batch version).

    games_df should have columns: game_id, home_score, away_score

    Returns count of bets settled.
    """
    conn = _get_connection()

    try:
        # Get all pending bets (single query)
        pending = conn.execute(
            "SELECT * FROM bets WHERE outcome IS NULL"
        ).fetchall()

        if not pending:
            conn.close()
            return 0

        # Prepare batch updates
        updates = []
        settled_at = datetime.now().isoformat()

        for bet in pending:
            bet_dict = dict(bet)
            game_id = bet_dict["game_id"]

            # Find corresponding game
            game = games_df[games_df["game_id"] == game_id]
            if game.empty:
                continue

            row = game.iloc[0]
            if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                continue

            home_score = int(row["home_score"])
            away_score = int(row["away_score"])

            # Calculate outcome
            try:
                outcome, profit = _calculate_bet_outcome(bet_dict, home_score, away_score)

                updates.append((
                    outcome,
                    home_score,
                    away_score,
                    profit,
                    settled_at,
                    bet_dict["id"]
                ))

                logger.debug(f"Prepared settlement: {bet_dict['id']} -> {outcome} (${profit:+.2f})")

            except ValueError as e:
                logger.error(f"Error calculating outcome for bet {bet_dict['id']}: {e}")
                continue

        # Batch update (single query)
        if updates:
            conn.executemany("""
                UPDATE bets SET
                    outcome = ?,
                    actual_score_home = ?,
                    actual_score_away = ?,
                    profit = ?,
                    settled_at = ?
                WHERE id = ?
            """, updates)

            conn.commit()
            logger.info(f"Settled {len(updates)} bets in batch")

        return len(updates)

    except Exception as e:
        logger.error(f"Error in batch settlement: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal."""
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


def _decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


def calculate_clv(
    opening_odds: float,
    closing_odds: float,
    opening_line: Optional[float] = None,
    closing_line: Optional[float] = None,
    bet_type: str = "moneyline",
) -> float:
    """
    Calculate Closing Line Value (CLV).

    For moneyline: CLV = closing_implied_prob - opening_implied_prob
    For spread/totals: CLV also considers line movement

    Positive CLV = you got better odds than closing (good)
    Negative CLV = closing odds were better (bad)

    Returns CLV as a percentage.
    """
    if bet_type == "moneyline":
        # CLV based on implied probability difference
        open_prob = _decimal_to_implied_prob(_american_to_decimal(opening_odds))
        close_prob = _decimal_to_implied_prob(_american_to_decimal(closing_odds))
        # You win if closing prob is higher (you got in when market thought it less likely)
        return close_prob - open_prob

    elif bet_type == "spread":
        # For spreads, line movement is key
        # If you bet home -3 and it closes at home -5, you have +2 points of CLV
        if opening_line is not None and closing_line is not None:
            # Line CLV in points (simplified to % assuming ~10% per point)
            line_clv = (closing_line - opening_line) * 0.03  # ~3% per point
            # Also factor in odds movement
            open_prob = _decimal_to_implied_prob(_american_to_decimal(opening_odds))
            close_prob = _decimal_to_implied_prob(_american_to_decimal(closing_odds))
            odds_clv = close_prob - open_prob
            return line_clv + odds_clv
        else:
            # Just use odds
            open_prob = _decimal_to_implied_prob(_american_to_decimal(opening_odds))
            close_prob = _decimal_to_implied_prob(_american_to_decimal(closing_odds))
            return close_prob - open_prob

    elif bet_type == "totals":
        # Similar to spread - line movement matters
        if opening_line is not None and closing_line is not None:
            # For over bets: if line goes up, you have positive CLV
            # For under bets: if line goes down, you have positive CLV
            line_clv = (closing_line - opening_line) * 0.02  # ~2% per point for totals
            open_prob = _decimal_to_implied_prob(_american_to_decimal(opening_odds))
            close_prob = _decimal_to_implied_prob(_american_to_decimal(closing_odds))
            odds_clv = close_prob - open_prob
            return line_clv + odds_clv
        else:
            open_prob = _decimal_to_implied_prob(_american_to_decimal(opening_odds))
            close_prob = _decimal_to_implied_prob(_american_to_decimal(closing_odds))
            return close_prob - open_prob

    return 0.0


def update_closing_line(
    bet_id: str,
    closing_odds: float,
    closing_line: Optional[float] = None,
) -> Optional[Dict]:
    """
    Update a bet with closing line information and calculate CLV.

    Call this just before game time to record the closing line.

    Returns the updated bet record.
    """
    conn = _get_connection()

    bet = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()

    if not bet:
        logger.warning(f"Bet not found: {bet_id}")
        conn.close()
        return None

    bet = dict(bet)

    # Calculate CLV
    opening_odds = bet["odds"]
    opening_line = bet.get("line")

    clv = calculate_clv(
        opening_odds=opening_odds,
        closing_odds=closing_odds,
        opening_line=opening_line,
        closing_line=closing_line,
        bet_type=bet["bet_type"],
    )

    clv_updated_at = datetime.now().isoformat()

    conn.execute("""
        UPDATE bets SET
            closing_odds = ?,
            closing_line = ?,
            clv = ?,
            clv_updated_at = ?
        WHERE id = ?
    """, (closing_odds, closing_line, clv, clv_updated_at, bet_id))

    conn.commit()

    result = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    conn.close()

    logger.info(f"Updated CLV for {bet_id}: {clv:+.2%}")
    return dict(result)


def update_closing_lines_from_odds(odds_data: Dict) -> int:
    """
    Batch update closing lines from odds API data.

    odds_data format: {
        game_id: {
            'moneyline': {'home': odds, 'away': odds},
            'spread': {'line': float, 'odds': float},
            'totals': {'line': float, 'over_odds': float, 'under_odds': float}
        }
    }

    Returns count of bets updated.
    """
    conn = _get_connection()

    # Get pending bets that don't have closing lines yet
    pending = conn.execute("""
        SELECT id, game_id, bet_type, bet_side
        FROM bets
        WHERE closing_odds IS NULL AND outcome IS NULL
    """).fetchall()

    conn.close()

    count = 0
    for bet in pending:
        game_id = bet["game_id"]
        bet_type = bet["bet_type"]
        bet_side = bet["bet_side"]

        if game_id not in odds_data:
            continue

        game_odds = odds_data[game_id]

        closing_odds = None
        closing_line = None

        if bet_type == "moneyline":
            if "moneyline" in game_odds:
                if bet_side == "home":
                    closing_odds = game_odds["moneyline"].get("home")
                else:
                    closing_odds = game_odds["moneyline"].get("away")

        elif bet_type == "spread":
            if "spread" in game_odds:
                closing_odds = game_odds["spread"].get("odds")
                closing_line = game_odds["spread"].get("line")

        elif bet_type == "totals":
            if "totals" in game_odds:
                closing_line = game_odds["totals"].get("line")
                if bet_side == "over":
                    closing_odds = game_odds["totals"].get("over_odds")
                else:
                    closing_odds = game_odds["totals"].get("under_odds")

        if closing_odds is not None:
            update_closing_line(bet["id"], closing_odds, closing_line)
            count += 1

    logger.info(f"Updated closing lines for {count} bets")
    return count


def get_clv_summary() -> Dict:
    """Calculate CLV performance statistics."""
    conn = _get_connection()

    result = conn.execute("""
        SELECT
            COUNT(*) as total,
            AVG(clv) as avg_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_count,
            SUM(CASE WHEN clv <= 0 THEN 1 ELSE 0 END) as negative_clv_count,
            MIN(clv) as min_clv,
            MAX(clv) as max_clv
        FROM bets
        WHERE clv IS NOT NULL
    """).fetchone()

    conn.close()

    if result["total"] == 0:
        return {
            "total_with_clv": 0,
            "avg_clv": 0,
            "positive_clv_rate": 0,
            "min_clv": 0,
            "max_clv": 0,
        }

    return {
        "total_with_clv": result["total"],
        "avg_clv": result["avg_clv"] or 0,
        "positive_clv_rate": (result["positive_clv_count"] or 0) / result["total"],
        "min_clv": result["min_clv"] or 0,
        "max_clv": result["max_clv"] or 0,
    }


def get_clv_by_type() -> pd.DataFrame:
    """Get CLV breakdown by bet type."""
    conn = _get_connection()

    df = pd.read_sql_query("""
        SELECT
            bet_type,
            COUNT(*) as bets,
            AVG(clv) as avg_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv,
            MIN(clv) as min_clv,
            MAX(clv) as max_clv
        FROM bets
        WHERE clv IS NOT NULL
        GROUP BY bet_type
    """, conn)

    conn.close()

    if not df.empty:
        df["positive_clv_rate"] = df["positive_clv"] / df["bets"]

    return df


def get_bet_history() -> pd.DataFrame:
    """Get all bets as a DataFrame."""
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM bets ORDER BY logged_at DESC", conn)
    conn.close()
    return df


def get_pending_bets() -> pd.DataFrame:
    """Get unsettled bets."""
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM bets WHERE outcome IS NULL ORDER BY commence_time",
        conn
    )
    conn.close()
    return df


def get_settled_bets() -> pd.DataFrame:
    """Get settled bets."""
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM bets WHERE outcome IS NOT NULL ORDER BY settled_at DESC",
        conn
    )
    conn.close()
    return df


def get_performance_summary() -> Dict:
    """Calculate overall betting performance including CLV."""
    conn = _get_connection()

    result = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
            SUM(CASE WHEN outcome IS NOT NULL THEN profit ELSE 0 END) as total_profit,
            SUM(CASE WHEN outcome IS NOT NULL THEN COALESCE(bet_amount, 100) ELSE 0 END) as total_wagered,
            AVG(CASE WHEN outcome IS NOT NULL THEN edge ELSE NULL END) as avg_edge,
            AVG(CASE WHEN outcome IS NOT NULL THEN odds ELSE NULL END) as avg_odds,
            AVG(clv) as avg_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_count,
            SUM(CASE WHEN clv IS NOT NULL THEN 1 ELSE 0 END) as clv_count
        FROM bets
        WHERE outcome IS NOT NULL
    """).fetchone()

    conn.close()

    if result["total"] == 0:
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": 0,
            "total_profit": 0,
            "total_wagered": 0,
            "roi": 0,
            "avg_edge": 0,
            "avg_odds": 0,
            "avg_clv": 0,
            "positive_clv_rate": 0,
            "clv_bets": 0,
        }

    wins = result["wins"] or 0
    losses = result["losses"] or 0
    total_wagered = result["total_wagered"] or 0
    clv_count = result["clv_count"] or 0

    return {
        "total_bets": result["total"],
        "wins": wins,
        "losses": losses,
        "pushes": result["pushes"] or 0,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_profit": result["total_profit"] or 0,
        "total_wagered": total_wagered,
        "roi": (result["total_profit"] or 0) / total_wagered if total_wagered > 0 else 0,
        "avg_edge": result["avg_edge"] or 0,
        "avg_odds": result["avg_odds"] or 0,
        "avg_clv": result["avg_clv"] or 0,
        "positive_clv_rate": (result["positive_clv_count"] or 0) / clv_count if clv_count > 0 else 0,
        "clv_bets": clv_count,
    }


def get_performance_by_type() -> pd.DataFrame:
    """Get performance breakdown by bet type."""
    conn = _get_connection()

    df = pd.read_sql_query("""
        SELECT
            bet_type,
            COUNT(*) as bets,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(profit) as profit,
            AVG(edge) as avg_edge
        FROM bets
        WHERE outcome IS NOT NULL
        GROUP BY bet_type
    """, conn)

    conn.close()

    if df.empty:
        return df

    df["win_rate"] = df.apply(
        lambda r: r["wins"] / (r["wins"] + r["losses"]) if (r["wins"] + r["losses"]) > 0 else 0,
        axis=1
    )

    return df


def get_daily_performance() -> pd.DataFrame:
    """Get daily P&L breakdown."""
    conn = _get_connection()

    df = pd.read_sql_query("""
        SELECT
            DATE(settled_at) as date,
            COUNT(*) as bets,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(profit) as profit
        FROM bets
        WHERE outcome IS NOT NULL
        GROUP BY DATE(settled_at)
        ORDER BY date
    """, conn)

    conn.close()
    return df


def clear_bet_history():
    """Clear all bet history (use with caution!)."""
    conn = _get_connection()
    conn.execute("DELETE FROM bets")
    conn.commit()
    conn.close()
    logger.info("Bet history cleared")


# =============================================================================
# Enhanced Analytics for Alpha Monitoring
# =============================================================================


def get_rolling_clv(window: int = 50) -> pd.DataFrame:
    """
    Calculate rolling CLV metrics over a specified window.

    Args:
        window: Number of bets to include in rolling calculation

    Returns:
        DataFrame with rolling CLV statistics per bet
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                id,
                logged_at,
                settled_at,
                bet_type,
                edge,
                clv,
                outcome,
                profit
            FROM bets
            WHERE clv IS NOT NULL AND outcome IS NOT NULL
            ORDER BY settled_at ASC
        """, conn)
    finally:
        conn.close()

    if df.empty or len(df) < window:
        return pd.DataFrame()

    # Calculate rolling metrics
    df['rolling_clv_mean'] = df['clv'].rolling(window=window, min_periods=1).mean()
    df['rolling_clv_std'] = df['clv'].rolling(window=window, min_periods=1).std()
    df['rolling_clv_sum'] = df['clv'].rolling(window=window, min_periods=1).sum()

    # Rolling positive CLV rate
    df['clv_positive'] = (df['clv'] > 0).astype(int)
    df['rolling_positive_clv_rate'] = df['clv_positive'].rolling(window=window, min_periods=1).mean()

    # Rolling win rate for comparison
    df['is_win'] = (df['outcome'] == 'win').astype(int)
    df['rolling_win_rate'] = df['is_win'].rolling(window=window, min_periods=1).mean()

    # Rolling profit
    df['rolling_profit'] = df['profit'].rolling(window=window, min_periods=1).sum()

    # Add bet index for trend analysis
    df['bet_index'] = range(len(df))

    return df[['bet_index', 'id', 'logged_at', 'settled_at', 'bet_type',
               'clv', 'rolling_clv_mean', 'rolling_clv_std', 'rolling_positive_clv_rate',
               'rolling_win_rate', 'profit', 'rolling_profit']]


def get_clv_trend(window: int = 50) -> Dict:
    """
    Detect if CLV is improving, stable, or declining.

    Args:
        window: Window for trend calculation

    Returns:
        Dictionary with trend analysis
    """
    rolling_df = get_rolling_clv(window=window)

    if rolling_df.empty or len(rolling_df) < window:
        return {
            'trend': 'insufficient_data',
            'slope': 0,
            'recent_clv': 0,
            'historical_clv': 0,
            'trend_strength': 0,
            'samples': len(rolling_df) if not rolling_df.empty else 0,
        }

    # Use recent vs historical comparison
    recent = rolling_df.tail(window // 2)
    historical = rolling_df.head(window // 2)

    recent_clv = recent['clv'].mean()
    historical_clv = historical['clv'].mean()
    clv_change = recent_clv - historical_clv

    # Calculate trend using linear regression on rolling CLV
    from scipy import stats
    x = rolling_df['bet_index'].values
    y = rolling_df['rolling_clv_mean'].values

    # Handle NaN values - initialize defaults before conditional
    slope = 0.0
    r_value = 0.0

    mask = ~np.isnan(y)
    if mask.sum() >= 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])

    # Determine trend
    if slope > 0.0001:  # Positive slope threshold
        trend = 'improving'
    elif slope < -0.0001:  # Negative slope threshold
        trend = 'declining'
    else:
        trend = 'stable'

    return {
        'trend': trend,
        'slope': slope,
        'recent_clv': recent_clv,
        'historical_clv': historical_clv,
        'clv_change': clv_change,
        'trend_strength': abs(slope) * 10000,  # Scaled for readability
        'samples': len(rolling_df),
        'r_squared': r_value ** 2,
    }


def get_performance_by_edge_bucket() -> pd.DataFrame:
    """
    Analyze performance by edge size buckets.

    Returns:
        DataFrame with performance metrics per edge bucket
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                edge,
                clv,
                outcome,
                profit,
                bet_amount
            FROM bets
            WHERE outcome IS NOT NULL AND edge IS NOT NULL
        """, conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Define edge buckets
    def edge_bucket(edge):
        if edge < 0.02:
            return '0-2%'
        elif edge < 0.05:
            return '2-5%'
        elif edge < 0.08:
            return '5-8%'
        elif edge < 0.10:
            return '8-10%'
        else:
            return '10%+'

    df['edge_bucket'] = df['edge'].apply(edge_bucket)

    # Calculate metrics per bucket
    results = []
    for bucket in ['0-2%', '2-5%', '5-8%', '8-10%', '10%+']:
        bucket_df = df[df['edge_bucket'] == bucket]

        if bucket_df.empty:
            continue

        wins = (bucket_df['outcome'] == 'win').sum()
        losses = (bucket_df['outcome'] == 'loss').sum()
        total = len(bucket_df)

        # Calculate wagered (default to $100 if bet_amount is null)
        wagered = bucket_df['bet_amount'].fillna(100).sum()

        results.append({
            'edge_bucket': bucket,
            'bets': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_profit': bucket_df['profit'].sum(),
            'avg_profit': bucket_df['profit'].mean(),
            'roi': bucket_df['profit'].sum() / wagered if wagered > 0 else 0,
            'avg_clv': bucket_df['clv'].mean() if bucket_df['clv'].notna().any() else 0,
            'positive_clv_rate': (bucket_df['clv'] > 0).mean() if bucket_df['clv'].notna().any() else 0,
        })

    return pd.DataFrame(results)


def get_rolling_sharpe(window: int = 50, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calculate rolling Sharpe ratio for betting performance.

    Args:
        window: Rolling window size
        risk_free_rate: Risk-free rate (default 0)

    Returns:
        DataFrame with rolling Sharpe ratios
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                id,
                settled_at,
                profit,
                bet_amount
            FROM bets
            WHERE outcome IS NOT NULL AND profit IS NOT NULL
            ORDER BY settled_at ASC
        """, conn)
    finally:
        conn.close()

    if df.empty or len(df) < window:
        return pd.DataFrame()

    # Calculate return per bet (profit / bet_amount)
    # Replace 0 and null bet_amounts with 100 to avoid division by zero
    df['bet_amount'] = df['bet_amount'].fillna(100).replace(0, 100)
    df['return'] = df['profit'] / df['bet_amount']

    # Rolling mean return and std
    df['rolling_mean_return'] = df['return'].rolling(window=window, min_periods=window).mean()
    df['rolling_std_return'] = df['return'].rolling(window=window, min_periods=window).std()

    # Sharpe ratio (no annualization needed for bets)
    df['rolling_sharpe'] = (df['rolling_mean_return'] - risk_free_rate) / df['rolling_std_return']

    # Handle infinite values
    df['rolling_sharpe'] = df['rolling_sharpe'].replace([np.inf, -np.inf], np.nan)

    # Add bet index
    df['bet_index'] = range(len(df))

    return df[['bet_index', 'id', 'settled_at', 'return', 'rolling_mean_return',
               'rolling_std_return', 'rolling_sharpe']].dropna()


def get_performance_decay_metrics(recent_window: int = 50, historical_window: int = 200) -> Dict:
    """
    Compare recent performance to historical baseline.

    Args:
        recent_window: Number of recent bets to analyze
        historical_window: Historical baseline size

    Returns:
        Dictionary with decay detection metrics
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                clv,
                outcome,
                profit,
                edge,
                bet_amount
            FROM bets
            WHERE outcome IS NOT NULL
            ORDER BY settled_at ASC
        """, conn)
    finally:
        conn.close()

    if df.empty:
        return {'status': 'no_data', 'total_bets': 0}

    total_bets = len(df)

    if total_bets < recent_window:
        return {'status': 'insufficient_data', 'total_bets': total_bets}

    # Split into recent and historical
    recent = df.tail(recent_window)
    historical = df.head(min(historical_window, total_bets - recent_window))

    if len(historical) < 20:
        return {'status': 'insufficient_historical_data', 'total_bets': total_bets}

    # Calculate metrics for both periods
    def calc_metrics(data):
        wins = (data['outcome'] == 'win').sum()
        losses = (data['outcome'] == 'loss').sum()
        wagered = data['bet_amount'].fillna(100).sum()

        return {
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'avg_clv': data['clv'].mean() if data['clv'].notna().any() else 0,
            'positive_clv_rate': (data['clv'] > 0).mean() if data['clv'].notna().any() else 0,
            'roi': data['profit'].sum() / wagered if wagered > 0 else 0,
            'avg_edge': data['edge'].mean() if data['edge'].notna().any() else 0,
        }

    recent_metrics = calc_metrics(recent)
    historical_metrics = calc_metrics(historical)

    # Calculate changes - use threshold to avoid near-zero division issues
    changes = {}
    for key in recent_metrics:
        hist_val = historical_metrics[key]
        recent_val = recent_metrics[key]
        if abs(hist_val) > 1e-10:  # Use threshold instead of exact zero check
            changes[f'{key}_change'] = (recent_val - hist_val) / abs(hist_val)
        elif abs(recent_val) < 1e-10:
            # Both zero, no change
            changes[f'{key}_change'] = 0.0
        else:
            # Historical zero but recent non-zero - cap at 1000% change
            changes[f'{key}_change'] = 10.0 if recent_val > 0 else -10.0
            logger.warning(f"Capping {key}_change at ±1000% due to zero historical value")

    # Determine if there's decay
    decay_indicators = 0
    if changes.get('win_rate_change', 0) < -0.1:  # Win rate dropped > 10%
        decay_indicators += 1
    if changes.get('avg_clv_change', 0) < -0.2:  # CLV dropped > 20%
        decay_indicators += 1
    if changes.get('roi_change', 0) < -0.2:  # ROI dropped > 20%
        decay_indicators += 1
    if recent_metrics['avg_clv'] < 0:  # Negative CLV
        decay_indicators += 1

    if decay_indicators >= 3:
        status = 'significant_decay'
    elif decay_indicators >= 2:
        status = 'moderate_decay'
    elif decay_indicators >= 1:
        status = 'minor_decay'
    else:
        status = 'healthy'

    return {
        'status': status,
        'total_bets': total_bets,
        'recent_window': recent_window,
        'historical_window': len(historical),
        'recent': recent_metrics,
        'historical': historical_metrics,
        'changes': changes,
        'decay_indicators': decay_indicators,
    }


# =============================================================================
# Regime Detection Integration
# =============================================================================


def get_regime_status() -> Dict:
    """
    Get current regime status using SeasonalRegimeDetector.

    Loads settled bet history, populates the regime detector,
    and returns the current regime classification with recommendations.

    Returns:
        Dictionary with regime info:
        - regime: current regime (normal, volatile, edge_decay, hot_streak)
        - should_pause: whether to pause betting
        - pause_reason: reason if pausing recommended
        - phase: current season phase
        - metrics: rolling performance metrics
        - alerts: list of active alerts
    """
    try:
        from src.monitoring import SeasonalRegimeDetector, Regime
    except ImportError:
        return {
            'regime': 'normal',
            'should_pause': False,
            'pause_reason': '',
            'phase': 'unknown',
            'metrics': {},
            'alerts': [],
            'error': 'Regime detection module not available',
        }

    # Get settled bets with CLV
    conn = _get_connection()
    try:
        df = pd.read_sql_query("""
            SELECT
                id,
                settled_at,
                clv,
                outcome,
                profit,
                bet_amount
            FROM bets
            WHERE outcome IS NOT NULL
            ORDER BY settled_at ASC
        """, conn)
    finally:
        conn.close()

    if df.empty or len(df) < 10:
        return {
            'regime': 'normal',
            'should_pause': False,
            'pause_reason': '',
            'phase': 'early_season',
            'metrics': {
                'total_bets': len(df),
                'win_rate': 0,
                'avg_clv': 0,
            },
            'alerts': [],
            'message': f'Insufficient data ({len(df)} bets). Need at least 10 settled bets.',
        }

    # Initialize regime detector
    detector = SeasonalRegimeDetector(
        clv_threshold=0.0,  # Positive CLV is good
        win_rate_threshold=0.524,  # ~-110 breakeven
        lookback_window=min(50, len(df)),
        min_samples_for_detection=10,
    )

    # Populate detector with bet history
    for _, bet in df.iterrows():
        # Calculate ROI for this bet
        wager = bet['bet_amount'] or 100
        roi = (bet['profit'] or 0) / wager if wager > 0 else 0
        clv = bet['clv'] or 0
        won = bet['outcome'] == 'win'
        timestamp = pd.Timestamp(bet['settled_at']) if bet['settled_at'] else None

        detector.add_performance_sample(
            clv=clv,
            win=won,
            roi=roi,
            timestamp=timestamp,
        )

    # Get regime analysis
    current_regime = detector.get_current_regime()
    should_pause, pause_reason = detector.should_pause_betting()
    alerts = detector.detect_performance_decay()

    # Calculate recent metrics
    recent_bets = df.tail(min(50, len(df)))
    wins = (recent_bets['outcome'] == 'win').sum()
    losses = (recent_bets['outcome'] == 'loss').sum()

    metrics = {
        'total_bets': len(df),
        'recent_bets': len(recent_bets),
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'avg_clv': recent_bets['clv'].mean() if recent_bets['clv'].notna().any() else 0,
        'rolling_profit': recent_bets['profit'].sum() if recent_bets['profit'].notna().any() else 0,
        'positive_clv_rate': (recent_bets['clv'] > 0).mean() if recent_bets['clv'].notna().any() else 0,
    }

    # Determine season phase (approximate based on current date)
    from datetime import datetime
    today = datetime.now()
    # NBA season: Oct-Apr
    if today.month in [10, 11]:
        phase = 'early_season'
    elif today.month in [12, 1, 2]:
        phase = 'pre_allstar'
    elif today.month in [3]:
        phase = 'post_allstar'
    else:
        phase = 'playoff_push'

    # Format alerts for display
    alert_list = []
    for alert in alerts:
        alert_list.append({
            'severity': alert.severity,
            'message': alert.message,
            'action': alert.recommended_action,
        })

    return {
        'regime': current_regime.value,
        'should_pause': should_pause,
        'pause_reason': pause_reason,
        'phase': phase,
        'metrics': metrics,
        'alerts': alert_list,
        'n_alerts': len(alerts),
        'regime_color': _get_regime_color(current_regime),
    }


def _get_regime_color(regime) -> str:
    """Get display color for regime."""
    from src.monitoring import Regime
    colors = {
        Regime.NORMAL: 'green',
        Regime.VOLATILE: 'orange',
        Regime.EDGE_DECAY: 'red',
        Regime.HOT_STREAK: 'blue',
    }
    return colors.get(regime, 'gray')


def get_regime_recommendation() -> str:
    """
    Get a human-readable recommendation based on current regime.

    Returns:
        Recommendation string for display in UI
    """
    status = get_regime_status()

    regime = status['regime']
    metrics = status.get('metrics', {})
    win_rate = metrics.get('win_rate', 0)
    avg_clv = metrics.get('avg_clv', 0)

    if status['should_pause']:
        return f"PAUSE BETTING: {status['pause_reason']}"

    if regime == 'edge_decay':
        return f"CAUTION: Edge decay detected. CLV: {avg_clv:.1%}, Win Rate: {win_rate:.1%}. Consider reducing bet sizes."

    if regime == 'volatile':
        return f"HIGH VOLATILITY: Performance swinging. Win Rate: {win_rate:.1%}. Use conservative sizing."

    if regime == 'hot_streak':
        return f"HOT STREAK: {win_rate:.1%} win rate, {avg_clv:.1%} CLV. Don't overextend - variance will regress."

    # Normal regime
    if avg_clv > 0.01:
        return f"NORMAL: +CLV ({avg_clv:.1%}). Strategy performing well."
    elif avg_clv > 0:
        return f"NORMAL: Marginal +CLV ({avg_clv:.1%}). Continue monitoring."
    else:
        return f"NORMAL: Slight -CLV ({avg_clv:.1%}). Watch for decay signals."


# ========== Multi-Snapshot CLV Functions ==========


def calculate_multi_snapshot_clv(
    bet_id: str,
    snapshot_times: List[str] = None
) -> Dict[str, float]:
    """
    Calculate CLV at multiple time points before game.

    Args:
        bet_id: Bet identifier
        snapshot_times: Optional list of time windows to check ['1hr', '4hr', '12hr', '24hr']

    Returns:
        Dict with clv_at_1hr, clv_at_4hr, etc.
    """
    if snapshot_times is None:
        snapshot_times = ['1hr', '4hr', '12hr', '24hr']

    conn = _get_connection()

    # Get bet details
    bet = conn.execute("""
        SELECT * FROM bets WHERE id = ?
    """, (bet_id,)).fetchone()

    if not bet:
        conn.close()
        return {}

    from datetime import datetime, timedelta, timezone

    commence_time = datetime.fromisoformat(bet['commence_time'])
    # Ensure timezone awareness
    if commence_time.tzinfo is None:
        commence_time = commence_time.replace(tzinfo=timezone.utc)
    results = {}

    for time_window in snapshot_times:
        # Parse time window (e.g., '1hr', '4hr')
        hours = int(time_window.replace('hr', ''))
        snapshot_target = commence_time - timedelta(hours=hours)

        # Find closest snapshot to target time (within 2 hour window)
        window_start = snapshot_target - timedelta(hours=1)
        window_end = snapshot_target + timedelta(hours=1)

        snapshots = conn.execute("""
            SELECT odds, line FROM line_snapshots
            WHERE game_id = ?
            AND bet_type = ?
            AND side = ?
            AND snapshot_time BETWEEN ? AND ?
            ORDER BY ABS(JULIANDAY(snapshot_time) - JULIANDAY(?))
            LIMIT 1
        """, (
            bet['game_id'],
            bet['bet_type'],
            bet['bet_side'],
            window_start.isoformat(),
            window_end.isoformat(),
            snapshot_target.isoformat()
        )).fetchone()

        if snapshots and bet['closing_odds'] is not None:
            # Calculate CLV using calculate_clv function defined in this module
            clv = calculate_clv(
                opening_odds=snapshots['odds'],
                closing_odds=bet['closing_odds'],
                opening_line=snapshots['line'],
                closing_line=bet['closing_line'],
                bet_type=bet['bet_type']
            )
            results[f'clv_at_{time_window}'] = clv
        else:
            results[f'clv_at_{time_window}'] = None

    conn.close()
    return results


def analyze_optimal_booking_time(
    bet_type: str,
    bet_side: str = None,
    lookback_days: int = 90
) -> Dict:
    """
    Analyze historical bets to find optimal booking timing.

    Args:
        bet_type: 'moneyline', 'spread', or 'totals'
        bet_side: Optional filter by side
        lookback_days: Days of history to analyze

    Returns:
        - optimal_hours_before: Best time to book this bet type
        - avg_clv_by_hour: CLV breakdown by booking window
        - recommendation: 'book_early', 'book_late', 'neutral'
    """
    conn = _get_connection()

    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

    # Get bets with multi-snapshot CLV
    query = """
        SELECT
            booked_hours_before,
            clv_at_1hr,
            clv_at_4hr,
            clv_at_12hr,
            clv_at_24hr,
            clv,
            outcome
        FROM bets
        WHERE bet_type = ?
        AND logged_at >= ?
        AND clv IS NOT NULL
    """
    params = [bet_type, cutoff_date]

    if bet_side:
        query += " AND bet_side = ?"
        params.append(bet_side)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return {
            "optimal_hours_before": None,
            "avg_clv_by_hour": {},
            "recommendation": "insufficient_data"
        }

    # Calculate average CLV by time window
    avg_clv_by_hour = {}
    for col in ['clv_at_1hr', 'clv_at_4hr', 'clv_at_12hr', 'clv_at_24hr']:
        if col in df.columns:
            hour = col.replace('clv_at_', '').replace('hr', '')
            avg_clv_by_hour[int(hour)] = df[col].mean()

    # Find optimal timing
    if avg_clv_by_hour:
        optimal_hours = max(avg_clv_by_hour, key=avg_clv_by_hour.get)
        optimal_clv = avg_clv_by_hour[optimal_hours]

        # Recommendation logic
        if optimal_hours >= 12:
            recommendation = "book_early"
        elif optimal_hours <= 4:
            recommendation = "book_late"
        else:
            recommendation = "neutral"
    else:
        optimal_hours = None
        recommendation = "insufficient_data"

    return {
        "optimal_hours_before": optimal_hours,
        "avg_clv_by_hour": avg_clv_by_hour,
        "recommendation": recommendation,
        "sample_size": len(df)
    }


def calculate_line_velocity(
    game_id: str,
    bet_type: str,
    bet_side: str,
    window_hours: int = 4
) -> Optional[float]:
    """
    Calculate line movement velocity (points/hour or prob%/hour).

    Args:
        game_id: Game identifier
        bet_type: Type of bet
        bet_side: Side of bet
        window_hours: Time window for velocity calculation

    Returns:
        Line velocity (positive = moving against us, negative = moving in our favor)
        None if insufficient data
    """
    conn = _get_connection()

    from datetime import datetime, timedelta

    # Get snapshots for this game
    snapshots = pd.read_sql_query("""
        SELECT snapshot_time, odds, line
        FROM line_snapshots
        WHERE game_id = ?
        AND bet_type = ?
        AND side = ?
        ORDER BY snapshot_time DESC
        LIMIT 10
    """, conn, params=(game_id, bet_type, bet_side))

    conn.close()

    if len(snapshots) < 2:
        return None

    # Calculate velocity over window
    recent = snapshots.iloc[0]
    older = snapshots[snapshots['snapshot_time'] <=
                      (datetime.fromisoformat(recent['snapshot_time']) -
                       timedelta(hours=window_hours)).isoformat()]

    if older.empty:
        return None

    older_snap = older.iloc[0]

    # For spreads/totals, use line movement
    if bet_type in ['spread', 'totals'] and recent['line'] and older_snap['line']:
        line_change = recent['line'] - older_snap['line']
        hours_diff = (datetime.fromisoformat(recent['snapshot_time']) -
                     datetime.fromisoformat(older_snap['snapshot_time'])).total_seconds() / 3600

        if hours_diff > 0:
            return line_change / hours_diff

    # For moneyline, use implied probability change
    if recent['odds'] and older_snap['odds']:
        # Use american_to_prob function defined in this module
        recent_prob = american_to_prob(recent['odds'])
        older_prob = american_to_prob(older_snap['odds'])
        prob_change = recent_prob - older_prob

        hours_diff = (datetime.fromisoformat(recent['snapshot_time']) -
                     datetime.fromisoformat(older_snap['snapshot_time'])).total_seconds() / 3600

        if hours_diff > 0:
            return prob_change / hours_diff

    return None


def get_enhanced_clv_summary() -> Dict:
    """
    Extended CLV analysis including:
    - CLV by booking timing window
    - CLV by bookmaker
    - Optimal booking recommendations
    - Line movement velocity correlations
    """
    conn = _get_connection()

    # Basic CLV summary
    basic_summary = get_clv_summary()

    # CLV by time window
    time_window_query = """
        SELECT
            AVG(clv_at_1hr) as avg_clv_1hr,
            AVG(clv_at_4hr) as avg_clv_4hr,
            AVG(clv_at_12hr) as avg_clv_12hr,
            AVG(clv_at_24hr) as avg_clv_24hr,
            AVG(line_velocity) as avg_line_velocity,
            AVG(max_clv_achieved) as avg_max_clv
        FROM bets
        WHERE clv IS NOT NULL
    """
    time_windows = conn.execute(time_window_query).fetchone()

    # CLV by bookmaker
    bookmaker_query = """
        SELECT
            bookmaker,
            COUNT(*) as count,
            AVG(clv) as avg_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_count
        FROM bets
        WHERE clv IS NOT NULL AND bookmaker IS NOT NULL
        GROUP BY bookmaker
        ORDER BY avg_clv DESC
    """
    bookmaker_df = pd.read_sql_query(bookmaker_query, conn)

    # Velocity correlation with outcome
    velocity_query = """
        SELECT
            AVG(CASE WHEN outcome = 'win' THEN line_velocity ELSE NULL END) as avg_velocity_wins,
            AVG(CASE WHEN outcome = 'loss' THEN line_velocity ELSE NULL END) as avg_velocity_losses
        FROM bets
        WHERE line_velocity IS NOT NULL AND outcome IS NOT NULL
    """
    velocity_corr = conn.execute(velocity_query).fetchone()

    conn.close()

    return {
        **basic_summary,
        "clv_by_time_window": {
            "1hr": time_windows['avg_clv_1hr'] if time_windows['avg_clv_1hr'] else None,
            "4hr": time_windows['avg_clv_4hr'] if time_windows['avg_clv_4hr'] else None,
            "12hr": time_windows['avg_clv_12hr'] if time_windows['avg_clv_12hr'] else None,
            "24hr": time_windows['avg_clv_24hr'] if time_windows['avg_clv_24hr'] else None,
        },
        "avg_line_velocity": time_windows['avg_line_velocity'] if time_windows['avg_line_velocity'] else None,
        "avg_max_clv_achieved": time_windows['avg_max_clv'] if time_windows['avg_max_clv'] else None,
        "clv_by_bookmaker": bookmaker_df.to_dict('records'),
        "velocity_correlation": {
            "wins": velocity_corr['avg_velocity_wins'] if velocity_corr['avg_velocity_wins'] else None,
            "losses": velocity_corr['avg_velocity_losses'] if velocity_corr['avg_velocity_losses'] else None
        }
    }


# ========== Odds Conversion Helper Functions ==========


def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability (includes bookmaker vig).

    Args:
        odds: American odds (e.g., +150, -110)

    Returns:
        Implied probability (0 to 1)

    Examples:
        >>> american_to_implied_prob(+150)  # Underdog
        0.400  # 40% implied probability
        >>> american_to_implied_prob(-150)  # Favorite
        0.600  # 60% implied probability

    Note:
        This includes the bookmaker's vig (overround). For true probabilities,
        use no-vig probabilities from the odds API when available.

    Raises:
        ValueError: If odds are invalid (not numeric, NaN, inf, or out of range)
    """
    import numpy as np

    # Validate numeric type
    if not isinstance(odds, (int, float)):
        raise ValueError(f"Odds must be a number, got {type(odds)}")

    # Check for NaN or inf
    if np.isnan(odds) or np.isinf(odds):
        raise ValueError(f"Odds cannot be NaN or infinite, got {odds}")

    # Validate odds value
    if odds == 0:
        raise ValueError("Odds cannot be 0 (use -100 for even money)")
    if odds == -100:
        raise ValueError("Odds of -100 are invalid (would be even money at 1.00 decimal)")

    # Validate reasonable range (typical sports betting range)
    if not (-10000 <= odds <= 10000):
        raise ValueError(f"Odds out of reasonable range [-10000, +10000]: {odds}")

    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


# Alias for backward compatibility
def american_to_prob(odds: float) -> float:
    """
    Deprecated: Use american_to_implied_prob() instead.

    Convert American odds to implied probability (includes vig).
    """
    return american_to_implied_prob(odds)


if __name__ == "__main__":
    # Test
    print("Bet History:")
    df = get_bet_history()
    print(f"  Total bets: {len(df)}")

    pending = get_pending_bets()
    print(f"  Pending: {len(pending)}")

    settled = get_settled_bets()
    print(f"  Settled: {len(settled)}")

    if not settled.empty:
        summary = get_performance_summary()
        print("\nPerformance Summary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2%}" if 'rate' in k or 'roi' in k else f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
