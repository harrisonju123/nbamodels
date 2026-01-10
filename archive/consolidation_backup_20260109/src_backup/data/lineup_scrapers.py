"""
Lineup Scrapers - ESPN API Client

Fetches confirmed starting lineups from ESPN's NBA scoreboard API.
Tracks lineup changes and player availability for real-time predictions.
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
    TEAM_NAME_TO_ABBREV,
)


class ESPNLineupClient:
    """Client for fetching NBA lineup data from ESPN's API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    # Standard ESPN API headers
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    def __init__(self, db_path: str = None):
        """
        Initialize lineup scraper.

        Args:
            db_path: Path to SQLite database (defaults to BETS_DB_PATH)
        """
        self.db_path = db_path or BETS_DB_PATH
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._init_tables()

    def _init_tables(self):
        """Initialize lineup database tables."""
        conn = self._get_connection()
        try:
            # Confirmed lineups table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confirmed_lineups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    team_abbrev TEXT NOT NULL,
                    player_id INTEGER,
                    player_name TEXT NOT NULL,
                    is_starter BOOLEAN DEFAULT 0,
                    position TEXT,
                    status TEXT,
                    source TEXT DEFAULT 'espn',
                    confirmed_at TEXT,
                    collected_at TEXT NOT NULL,
                    UNIQUE(game_id, team_abbrev, player_name)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineups_game
                ON confirmed_lineups(game_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineups_team
                ON confirmed_lineups(team_abbrev)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineups_date
                ON confirmed_lineups(game_date)
            """)

            conn.commit()
            logger.debug("Lineup tables initialized")
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
        Make a request to ESPN API with error handling.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response dict (empty dict on error)
        """
        url = f"{self.BASE_URL}/{endpoint}"
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
            logger.error(f"ESPN API request failed: {e}")
            return {}

    def get_todays_lineups(self, game_date: str = None) -> pd.DataFrame:
        """
        Fetch lineup information for today's games.

        Args:
            game_date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            DataFrame with lineup information
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Format date for ESPN API (YYYYMMDD)
        date_obj = datetime.strptime(game_date, "%Y-%m-%d")
        espn_date = date_obj.strftime("%Y%m%d")

        logger.info(f"Fetching lineups for {game_date}")

        # Fetch scoreboard
        data = self._make_request("scoreboard", {
            "dates": espn_date
        })

        if not data or "events" not in data:
            logger.warning(f"No scoreboard data for {game_date}")
            return pd.DataFrame()

        # Parse lineup data from each game
        records = []
        for event in data.get("events", []):
            game_id = event.get("id")
            status = event.get("status", {}).get("type", {}).get("name")

            # Get competitions (should be 1 for regular games)
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            competition = competitions[0]

            # Extract teams and rosters
            competitors = competition.get("competitors", [])
            for competitor in competitors:
                team_info = competitor.get("team", {})
                team_name = team_info.get("displayName", "")
                team_abbrev = team_info.get("abbreviation", "UNK")

                # Normalize team abbreviation
                if team_name in TEAM_NAME_TO_ABBREV:
                    team_abbrev = TEAM_NAME_TO_ABBREV[team_name]

                # Check if roster/lineup information is available
                roster = competitor.get("roster", [])
                if not roster:
                    logger.debug(f"No roster for {team_abbrev} in game {game_id}")
                    continue

                # Extract player information
                for player in roster:
                    athlete = player.get("athlete", {})

                    # Check if player is in starting lineup
                    is_starter = player.get("starter", False)

                    # Get player status
                    player_status = athlete.get("status", {}).get("type")

                    records.append({
                        "game_id": game_id,
                        "game_date": game_date,
                        "team_abbrev": team_abbrev,
                        "player_id": athlete.get("id"),
                        "player_name": athlete.get("displayName"),
                        "is_starter": 1 if is_starter else 0,
                        "position": athlete.get("position", {}).get("abbreviation"),
                        "status": player_status or "active",
                        "source": "espn",
                        "confirmed_at": datetime.now().isoformat() if is_starter else None,
                        "collected_at": datetime.now().isoformat(),
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            starter_count = len(df[df['is_starter']==1])
            logger.info(f"Found {len(df)} player records ({starter_count} starters)")
        else:
            logger.info("No lineup data available yet (normal if lineups not yet posted)")
        return df

    def get_game_lineup(self, game_id: str) -> Dict[str, List[str]]:
        """
        Get confirmed lineup for a specific game.

        Args:
            game_id: ESPN game ID

        Returns:
            Dictionary with 'home' and 'away' starter lists
        """
        conn = self._get_connection()
        try:
            # Query confirmed starters for this game
            query = """
                SELECT team_abbrev, player_name
                FROM confirmed_lineups
                WHERE game_id = ? AND is_starter = 1
                ORDER BY team_abbrev, player_name
            """

            df = pd.read_sql_query(query, conn, params=(game_id,))

            if df.empty:
                return {"home": [], "away": []}

            # Group by team
            lineups = {}
            for team, group in df.groupby("team_abbrev"):
                lineups[team] = group["player_name"].tolist()

            # Need to determine which is home/away (requires additional game info)
            # For now, return as team keys
            return lineups

        finally:
            conn.close()

    def save_lineups(self, lineups_df: pd.DataFrame) -> int:
        """
        Save lineup data to database.

        Args:
            lineups_df: DataFrame with lineup information

        Returns:
            Number of records inserted/updated
        """
        if lineups_df.empty:
            return 0

        conn = self._get_connection()
        try:
            records = []
            for _, row in lineups_df.iterrows():
                records.append((
                    row["game_id"],
                    row["game_date"],
                    row["team_abbrev"],
                    row.get("player_id"),
                    row["player_name"],
                    row["is_starter"],
                    row.get("position"),
                    row.get("status"),
                    row.get("source", "espn"),
                    row.get("confirmed_at"),
                    row["collected_at"],
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO confirmed_lineups
                (game_id, game_date, team_abbrev, player_id, player_name,
                 is_starter, position, status, source, confirmed_at, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            conn.commit()
            logger.info(f"Saved {len(records)} lineup records")
            return len(records)
        except Exception as e:
            logger.error(f"Error saving lineups: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_lineups(
        self,
        game_id: str = None,
        game_date: str = None,
        team_abbrev: str = None,
        starters_only: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve lineup data from database.

        Args:
            game_id: Specific game ID
            game_date: Game date (YYYY-MM-DD)
            team_abbrev: Team abbreviation
            starters_only: Only return confirmed starters

        Returns:
            DataFrame with lineup data
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM confirmed_lineups WHERE 1=1"
            params = []

            if game_id:
                query += " AND game_id = ?"
                params.append(game_id)
            if game_date:
                query += " AND game_date = ?"
                params.append(game_date)
            if team_abbrev:
                query += " AND team_abbrev = ?"
                params.append(team_abbrev)
            if starters_only:
                query += " AND is_starter = 1"

            query += " ORDER BY game_date DESC, game_id, team_abbrev, is_starter DESC"

            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()


if __name__ == "__main__":
    # Test the lineup scraper
    client = ESPNLineupClient()

    # Fetch today's lineups
    lineups = client.get_todays_lineups()
    print(f"\\nFound {len(lineups)} player records")

    if not lineups.empty:
        starters = lineups[lineups["is_starter"] == 1]
        print(f"Starters: {len(starters)}")
        print("\\nSample starters:")
        print(starters.head(10)[["team_abbrev", "player_name", "position", "status"]])

        # Save lineups
        count = client.save_lineups(lineups)
        print(f"\\nSaved {count} records to database")
