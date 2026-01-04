"""
Referee Data Collection Client

Fetches referee assignments and historical statistics from NBA.com API.
Tracks referee tendencies for totals, pace, and home/away bias.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
import requests
from loguru import logger

from src.utils.constants import (
    API_TIMEOUT_SECONDS,
    API_RATE_LIMIT_DELAY,
    BETS_DB_PATH,
)


class RefereeDataClient:
    """Client for fetching NBA referee data and assignments."""

    NBA_STATS_BASE_URL = "https://stats.nba.com/stats"

    # Standard NBA Stats API headers
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
    }

    def __init__(self, db_path: str = None):
        """
        Initialize referee data client.

        Args:
            db_path: Path to SQLite database (defaults to BETS_DB_PATH)
        """
        self.db_path = db_path or BETS_DB_PATH
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._init_tables()

    def _init_tables(self):
        """Initialize referee database tables."""
        conn = self._get_connection()
        try:
            # Referee assignments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS referee_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    ref_name TEXT NOT NULL,
                    ref_role TEXT,
                    collected_at TEXT NOT NULL,
                    UNIQUE(game_id, ref_name)
                )
            """)

            # Referee statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS referee_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_name TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    games_worked INTEGER DEFAULT 0,
                    avg_total_points REAL,
                    avg_home_score REAL,
                    avg_away_score REAL,
                    avg_fouls_per_game REAL,
                    home_win_rate REAL,
                    over_rate REAL,
                    pace_factor REAL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(ref_name, season)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ref_assignments_game
                ON referee_assignments(game_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ref_assignments_date
                ON referee_assignments(game_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ref_stats_name
                ON referee_stats(ref_name)
            """)

            conn.commit()
            logger.debug("Referee tables initialized")
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make a request to NBA Stats API with error handling.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response dict (empty dict on error)
        """
        url = f"{self.NBA_STATS_BASE_URL}/{endpoint}"
        params = params or {}

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=API_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            time.sleep(API_RATE_LIMIT_DELAY)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"NBA Stats API request failed: {e}")
            return {}

    def get_todays_referees(self, game_date: str = None) -> pd.DataFrame:
        """
        Fetch referee assignments for today's games.

        Args:
            game_date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            DataFrame with referee assignments
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Format date for NBA API (MMDDYYYY)
        date_obj = datetime.strptime(game_date, "%Y-%m-%d")
        nba_date = date_obj.strftime("%m/%d/%Y")

        logger.info(f"Fetching referee assignments for {game_date}")

        # Fetch scoreboard with officials
        data = self._make_request("scoreboardv2", {
            "GameDate": nba_date,
            "LeagueID": "00",
            "DayOffset": "0"
        })

        if not data:
            logger.warning(f"No scoreboard data for {game_date}")
            return pd.DataFrame()

        # Parse game headers and officials
        records = []
        result_sets = data.get("resultSets", [])

        # Find GameHeader and Officials result sets
        game_headers = None
        officials = None

        for rs in result_sets:
            if rs.get("name") == "GameHeader":
                game_headers = rs
            elif rs.get("name") == "Officials":
                officials = rs

        if not game_headers or not officials:
            logger.warning("Missing GameHeader or Officials data")
            return pd.DataFrame()

        # Parse officials
        headers = officials.get("headers", [])
        rows = officials.get("rowSet", [])

        for row in rows:
            row_dict = dict(zip(headers, row))
            records.append({
                "game_id": row_dict.get("GAME_ID"),
                "game_date": game_date,
                "ref_name": row_dict.get("OFFICIAL_ID"),  # Name might be in FIRST_NAME/LAST_NAME
                "ref_role": None,  # Will need to determine from position
                "collected_at": datetime.now().isoformat(),
            })

        df = pd.DataFrame(records)
        logger.info(f"Found {len(df)} referee assignments for {game_date}")
        return df

    def save_referee_assignments(self, assignments_df: pd.DataFrame) -> int:
        """
        Save referee assignments to database.

        Args:
            assignments_df: DataFrame with referee assignments

        Returns:
            Number of records inserted
        """
        if assignments_df.empty:
            return 0

        conn = self._get_connection()
        try:
            records = []
            for _, row in assignments_df.iterrows():
                records.append((
                    row["game_id"],
                    row["game_date"],
                    row["ref_name"],
                    row.get("ref_role"),
                    row["collected_at"],
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO referee_assignments
                (game_id, game_date, ref_name, ref_role, collected_at)
                VALUES (?, ?, ?, ?, ?)
            """, records)

            conn.commit()
            logger.info(f"Saved {len(records)} referee assignments")
            return len(records)
        except Exception as e:
            logger.error(f"Error saving referee assignments: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_referee_assignments(
        self,
        game_id: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Retrieve referee assignments from database.

        Args:
            game_id: Specific game ID
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            DataFrame with referee assignments
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM referee_assignments WHERE 1=1"
            params = []

            if game_id:
                query += " AND game_id = ?"
                params.append(game_id)
            if start_date:
                query += " AND game_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND game_date <= ?"
                params.append(end_date)

            query += " ORDER BY game_date DESC, game_id"

            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def calculate_referee_stats(self, ref_name: str, season: int = 2025) -> Dict:
        """
        Calculate historical statistics for a referee.

        Args:
            ref_name: Referee name
            season: NBA season (e.g., 2025 for 2024-25)

        Returns:
            Dictionary with referee statistics
        """
        # This would require historical game data with referee assignments
        # For now, return placeholder stats
        logger.warning(f"Referee stats calculation not yet implemented for {ref_name}")

        return {
            "ref_name": ref_name,
            "season": season,
            "games_worked": 0,
            "avg_total_points": None,
            "avg_home_score": None,
            "avg_away_score": None,
            "avg_fouls_per_game": None,
            "home_win_rate": None,
            "over_rate": None,
            "pace_factor": 1.0,  # Default neutral
            "updated_at": datetime.now().isoformat(),
        }

    def save_referee_stats(self, stats: Dict) -> bool:
        """
        Save referee statistics to database.

        Args:
            stats: Dictionary with referee stats

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO referee_stats
                (ref_name, season, games_worked, avg_total_points, avg_home_score,
                 avg_away_score, avg_fouls_per_game, home_win_rate, over_rate,
                 pace_factor, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats["ref_name"],
                stats["season"],
                stats["games_worked"],
                stats["avg_total_points"],
                stats["avg_home_score"],
                stats["avg_away_score"],
                stats["avg_fouls_per_game"],
                stats["home_win_rate"],
                stats["over_rate"],
                stats["pace_factor"],
                stats["updated_at"],
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving referee stats: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_referee_stats(
        self,
        ref_name: str = None,
        season: int = None
    ) -> pd.DataFrame:
        """
        Retrieve referee statistics from database.

        Args:
            ref_name: Specific referee name
            season: Specific season

        Returns:
            DataFrame with referee statistics
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM referee_stats WHERE 1=1"
            params = []

            if ref_name:
                query += " AND ref_name = ?"
                params.append(ref_name)
            if season:
                query += " AND season = ?"
                params.append(season)

            query += " ORDER BY season DESC, ref_name"

            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()


if __name__ == "__main__":
    # Test the referee data client
    client = RefereeDataClient()

    # Fetch today's referee assignments
    assignments = client.get_todays_referees()
    print(f"\\nFound {len(assignments)} referee assignments")
    if not assignments.empty:
        print(assignments.head())

    # Save assignments if any
    if not assignments.empty:
        count = client.save_referee_assignments(assignments)
        print(f"\\nSaved {count} assignments to database")
