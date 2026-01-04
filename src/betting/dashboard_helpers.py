"""
Helper functions for dashboard to display CLV, steam moves, and line movement.
"""

import sqlite3
from typing import Dict, Optional
from pathlib import Path
from loguru import logger


def get_game_line_data(game_id: str, db_path: str = "data/bets/bets.db") -> Dict:
    """
    Get opening line, current movement, and CLV data for a game.

    Returns dict with:
        - opening_line: Opening spread line (if available)
        - line_movement: Points moved from opening
        - steam_detected: Whether steam move detected
        - rlm_detected: Whether RLM detected
        - avg_clv_pct: Historical CLV % for similar bets
    """
    result = {
        'opening_line': None,
        'line_movement': None,
        'steam_detected': False,
        'rlm_detected': False,
        'avg_clv_pct': None
    }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get opening spread line (average across books for home side)
        cursor.execute("""
            SELECT AVG(opening_line)
            FROM opening_lines
            WHERE game_id = ?
            AND bet_type = 'spread'
            AND side = 'home'
        """, (game_id,))

        row = cursor.fetchone()
        if row and row[0] is not None:
            result['opening_line'] = row[0]

        # Get historical CLV for this team's spread bets
        # (Would need more sophisticated logic in production)
        cursor.execute("""
            SELECT AVG(clv)
            FROM bets
            WHERE bet_type = 'spread'
            AND clv IS NOT NULL
            AND game_id NOT LIKE 'test_%'
            LIMIT 100
        """)

        row = cursor.fetchone()
        if row and row[0] is not None:
            result['avg_clv_pct'] = row[0]

        # Steam/RLM detection would require line_snapshots table
        # which needs hourly data collection
        cursor.execute("SELECT COUNT(*) FROM line_snapshots WHERE game_id = ?", (game_id,))
        snapshot_count = cursor.fetchone()[0]

        if snapshot_count >= 2:
            # Could detect steam if we have multiple snapshots
            # For now, just mark as False since we don't have the data yet
            pass

        conn.close()

    except Exception as e:
        logger.warning(f"Could not get line data for {game_id}: {e}")

    return result


def get_latest_clv_for_games() -> Dict[str, Dict]:
    """
    Get CLV and line movement data for all today's games.

    Returns dict mapping game_id -> line data dict
    """
    # For now, return empty dict since we need more historical data
    # This will be populated as games are played and snapshots collected
    return {}
