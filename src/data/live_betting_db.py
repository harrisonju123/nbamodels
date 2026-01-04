"""
Live Betting Database

Database schema and initialization for live betting system.
Stores game state snapshots, live odds, edge alerts, and paper bets.
"""

import sqlite3
import os
from loguru import logger

DB_PATH = "data/bets/live_betting.db"


def init_database():
    """Initialize live betting database with all tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info(f"Initializing live betting database at {DB_PATH}")

    # Table 1: Live game state snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_game_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            game_date TEXT NOT NULL,
            quarter INTEGER,
            time_remaining TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_team TEXT,
            away_team TEXT,
            game_status TEXT,
            game_clock INTEGER,
            period_time_remaining TEXT,
            UNIQUE(game_id, timestamp)
        )
    """)

    # Indexes for live_game_state
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_game_date
        ON live_game_state(game_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_game_id
        ON live_game_state(game_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_game_timestamp
        ON live_game_state(timestamp)
    """)

    logger.info("✓ Created live_game_state table")

    # Table 2: Live odds snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_odds_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            home_odds INTEGER,
            away_odds INTEGER,
            over_odds INTEGER,
            under_odds INTEGER,
            spread_value REAL,
            total_value REAL,
            UNIQUE(game_id, timestamp, bookmaker, market)
        )
    """)

    # Indexes for live_odds_snapshot
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_odds_game
        ON live_odds_snapshot(game_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_odds_time
        ON live_odds_snapshot(timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_odds_market
        ON live_odds_snapshot(market)
    """)

    logger.info("✓ Created live_odds_snapshot table")

    # Table 3: Edge alerts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_edge_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            bet_side TEXT,
            model_prob REAL,
            market_prob REAL,
            edge REAL,
            quarter INTEGER,
            score_diff INTEGER,
            time_remaining TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_team TEXT,
            away_team TEXT,
            line_value REAL,
            odds INTEGER,
            confidence TEXT,
            acted_on BOOLEAN DEFAULT 0,
            paper_bet_id INTEGER,
            dismissed BOOLEAN DEFAULT 0
        )
    """)

    # Indexes for live_edge_alerts
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_alerts_game
        ON live_edge_alerts(game_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_alerts_confidence
        ON live_edge_alerts(confidence)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_alerts_timestamp
        ON live_edge_alerts(timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_alerts_acted
        ON live_edge_alerts(acted_on)
    """)

    logger.info("✓ Created live_edge_alerts table")

    # Table 4: Paper bets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_paper_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            placed_at TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            bet_side TEXT NOT NULL,
            odds INTEGER,
            line_value REAL,
            stake REAL,
            expected_edge REAL,
            model_prob REAL,

            -- Game state at bet time
            quarter INTEGER,
            score_diff_at_bet INTEGER,
            home_score_at_bet INTEGER,
            away_score_at_bet INTEGER,
            time_remaining_at_bet TEXT,
            home_team TEXT,
            away_team TEXT,

            -- Settlement
            outcome TEXT DEFAULT 'pending',
            profit REAL,
            settled_at TEXT,
            final_home_score INTEGER,
            final_away_score INTEGER,
            final_spread_result REAL,

            -- Metadata
            bookmaker TEXT,
            notes TEXT
        )
    """)

    # Indexes for live_paper_bets
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_paper_game
        ON live_paper_bets(game_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_paper_outcome
        ON live_paper_bets(outcome)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_paper_placed
        ON live_paper_bets(placed_at)
    """)

    logger.info("✓ Created live_paper_bets table")

    conn.commit()
    conn.close()

    logger.success(f"Live betting database initialized successfully at {DB_PATH}")


def get_connection():
    """Get a connection to the live betting database."""
    return sqlite3.connect(DB_PATH)


def get_stats():
    """Get statistics about the live betting database."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    stats = {}

    # Count records in each table
    tables = [
        'live_game_state',
        'live_odds_snapshot',
        'live_edge_alerts',
        'live_paper_bets'
    ]

    for table in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = count

    # Get date range of data
    result = cursor.execute("""
        SELECT MIN(game_date) as min_date, MAX(game_date) as max_date
        FROM live_game_state
    """).fetchone()

    if result and result['min_date']:
        stats['date_range'] = {
            'start': result['min_date'],
            'end': result['max_date']
        }

    # Count pending paper bets
    pending = cursor.execute("""
        SELECT COUNT(*) FROM live_paper_bets WHERE outcome = 'pending'
    """).fetchone()[0]
    stats['pending_bets'] = pending

    # Count settled paper bets
    settled = cursor.execute("""
        SELECT COUNT(*) FROM live_paper_bets WHERE outcome != 'pending'
    """).fetchone()[0]
    stats['settled_bets'] = settled

    # Get win rate if we have settled bets
    if settled > 0:
        wins = cursor.execute("""
            SELECT COUNT(*) FROM live_paper_bets WHERE outcome = 'win'
        """).fetchone()[0]
        stats['win_rate'] = wins / settled

        # Get total profit
        total_profit = cursor.execute("""
            SELECT COALESCE(SUM(profit), 0) FROM live_paper_bets
            WHERE outcome != 'pending'
        """).fetchone()[0]
        stats['total_profit'] = total_profit

        # Get ROI
        total_stake = cursor.execute("""
            SELECT COALESCE(SUM(stake), 0) FROM live_paper_bets
            WHERE outcome != 'pending'
        """).fetchone()[0]
        stats['roi'] = (total_profit / total_stake) if total_stake > 0 else 0

    conn.close()

    return stats


def clear_old_data(days: int = 30):
    """
    Clear old data from database.

    Args:
        days: Keep data from last N days, delete older
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    conn = get_connection()
    cursor = conn.cursor()

    # Clear old game states
    deleted = cursor.execute("""
        DELETE FROM live_game_state WHERE game_date < ?
    """, (cutoff,)).rowcount
    logger.info(f"Deleted {deleted} old game state records")

    # Clear old odds snapshots
    deleted = cursor.execute("""
        DELETE FROM live_odds_snapshot
        WHERE timestamp < ?
    """, (cutoff,)).rowcount
    logger.info(f"Deleted {deleted} old odds snapshots")

    # Clear old alerts (but keep those with paper bets)
    deleted = cursor.execute("""
        DELETE FROM live_edge_alerts
        WHERE timestamp < ? AND paper_bet_id IS NULL
    """, (cutoff,)).rowcount
    logger.info(f"Deleted {deleted} old edge alerts")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Initialize database
    init_database()

    # Show stats
    print("\n=== Database Statistics ===")
    stats = get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
