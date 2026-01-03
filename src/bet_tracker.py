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

    conn.commit()
    return conn


def log_bet(
    game_id: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    bet_type: str,  # 'moneyline', 'spread', 'totals'
    bet_side: str,  # 'home', 'away', 'over', 'under', 'cover'
    odds: float,
    line: Optional[float],  # spread line or total line
    model_prob: float,
    market_prob: float,
    edge: float,
    kelly: float,
    bookmaker: str = None,
) -> Dict:
    """
    Log a new bet recommendation.

    Returns the logged bet record.
    """
    conn = _get_connection()
    bet_id = f"{game_id}_{bet_type}_{bet_side}"

    # Check for duplicate
    existing = conn.execute(
        "SELECT * FROM bets WHERE id = ?", (bet_id,)
    ).fetchone()

    if existing:
        logger.info(f"Bet already logged: {bet_id}")
        conn.close()
        return dict(existing)

    logged_at = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO bets (
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line, model_prob, market_prob,
            edge, kelly, bookmaker, logged_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet_id, game_id, home_team, away_team, str(commence_time),
        bet_type, bet_side, odds, line, model_prob, market_prob,
        edge, kelly, bookmaker, logged_at
    ))

    conn.commit()

    result = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    conn.close()

    logger.info(f"Logged bet: {away_team} @ {home_team} - {bet_type} {bet_side} at {odds:+.0f}")
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
    else:
        market_prob = abs(odds) / (abs(odds) + 100)

    # Calculate edge if model_prob provided
    edge = (model_prob - market_prob) if model_prob else None

    logged_at = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO bets (
            id, game_id, home_team, away_team, commence_time,
            bet_type, bet_side, odds, line, bet_amount,
            model_prob, market_prob, edge, bookmaker, logged_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet_id, game_id, home_team, away_team, str(commence_time),
        bet_type, bet_side, odds, line, bet_amount,
        model_prob, market_prob, edge, bookmaker, logged_at
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
        # Home covers if point_diff + spread > 0
        spread = bet["line"]
        margin = point_diff + spread
        if margin > 0:
            outcome = "win"
        elif margin < 0:
            outcome = "loss"
        else:
            outcome = "push"

    elif bet["bet_type"] == "totals":
        line = bet["line"]
        if bet["bet_side"] == "over":
            if total_points > line:
                outcome = "win"
            elif total_points < line:
                outcome = "loss"
            else:
                outcome = "push"
        else:  # under
            if total_points < line:
                outcome = "win"
            elif total_points > line:
                outcome = "loss"
            else:
                outcome = "push"
    else:
        outcome = None

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
    conn.close()

    logger.info(f"Settled bet: {bet_id} -> {outcome} (${profit:+.2f})")
    return dict(result)


def settle_all_pending(games_df: pd.DataFrame) -> int:
    """
    Settle all pending bets using game results.

    games_df should have columns: game_id, home_score, away_score

    Returns count of bets settled.
    """
    conn = _get_connection()
    pending = conn.execute(
        "SELECT id, game_id FROM bets WHERE outcome IS NULL"
    ).fetchall()
    conn.close()

    count = 0
    for bet in pending:
        game = games_df[games_df["game_id"] == bet["game_id"]]
        if game.empty:
            continue

        row = game.iloc[0]
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue

        settle_bet(bet["id"], int(row["home_score"]), int(row["away_score"]))
        count += 1

    return count


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
        else:
            changes[f'{key}_change'] = 0 if abs(recent_val) < 1e-10 else float('inf')

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
