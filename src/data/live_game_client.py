"""
Live Game Client

Fetch live NBA game data from stats.nba.com scoreboard API.
Tracks current game state: score, quarter, time remaining, etc.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import sqlite3
from loguru import logger

from src.data.live_betting_db import DB_PATH


class LiveGameClient:
    """Client for fetching live NBA game data."""

    BASE_URL = "https://stats.nba.com/stats"
    SCOREBOARD_ENDPOINT = "scoreboardv2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.nba.com/',
            'Origin': 'https://www.nba.com',
            'Accept': 'application/json',
        })

    def get_scoreboard(self, game_date: Optional[str] = None) -> Dict:
        """
        Fetch scoreboard data for a specific date.

        Args:
            game_date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            Raw scoreboard JSON response
        """
        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')

        url = f"{self.BASE_URL}/{self.SCOREBOARD_ENDPOINT}"
        params = {
            'GameDate': game_date,
            'LeagueID': '00',
            'DayOffset': '0'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.6)  # Rate limit: ~1 req/sec
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return {}

    def parse_scoreboard(self, scoreboard_data: Dict) -> pd.DataFrame:
        """
        Parse scoreboard JSON into structured DataFrame.

        Returns DataFrame with columns:
        - game_id, game_date, game_status_id, game_status_text
        - home_team_id, away_team_id, home_team, away_team
        - period (quarter), game_clock, home_score, away_score
        """
        if not scoreboard_data or 'resultSets' not in scoreboard_data:
            return pd.DataFrame()

        games = []

        # GameHeader result set (index 0)
        game_header = scoreboard_data['resultSets'][0]
        headers = game_header['headers']
        rows = game_header['rowSet']

        for row in rows:
            game_dict = dict(zip(headers, row))

            # Extract key fields
            games.append({
                'game_id': game_dict.get('GAME_ID'),
                'game_date': game_dict.get('GAME_DATE_EST'),
                'game_status_id': game_dict.get('GAME_STATUS_ID'),  # 1=not started, 2=live, 3=final
                'game_status_text': game_dict.get('GAME_STATUS_TEXT'),
                'home_team_id': game_dict.get('HOME_TEAM_ID'),
                'away_team_id': game_dict.get('VISITOR_TEAM_ID'),
                'period': game_dict.get('LIVE_PERIOD'),  # Current quarter (0 if not started)
                'game_clock': game_dict.get('LIVE_PC_TIME', ''),  # e.g., "7:23" or ""
                'home_score': game_dict.get('HOME_TEAM_SCORE', 0) or 0,
                'away_score': game_dict.get('VISITOR_TEAM_SCORE', 0) or 0,
            })

        df = pd.DataFrame(games)

        # Get team abbreviations from LineScore result set
        if len(scoreboard_data['resultSets']) > 1:
            line_score = scoreboard_data['resultSets'][1]
            line_headers = line_score['headers']
            line_rows = line_score['rowSet']

            team_map = {}
            for row in line_rows:
                line_dict = dict(zip(line_headers, row))
                team_id = line_dict.get('TEAM_ID')
                team_abbrev = line_dict.get('TEAM_ABBREVIATION')
                if team_id and team_abbrev:
                    team_map[team_id] = team_abbrev

            # Map team IDs to abbreviations
            df['home_team'] = df['home_team_id'].map(team_map)
            df['away_team'] = df['away_team_id'].map(team_map)

        return df

    def get_todays_games(self) -> pd.DataFrame:
        """Fetch today's games."""
        scoreboard = self.get_scoreboard()
        return self.parse_scoreboard(scoreboard)

    def get_live_games(self) -> pd.DataFrame:
        """Get only games that are currently live."""
        all_games = self.get_todays_games()

        if all_games.empty:
            return all_games

        # Filter to live games (game_status_id == 2)
        live_games = all_games[all_games['game_status_id'] == 2].copy()

        logger.info(f"Found {len(live_games)} live games")

        return live_games

    def get_game_details(self, game_id: str) -> Optional[Dict]:
        """
        Get detailed state for a specific game.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with game state or None if not found
        """
        games = self.get_todays_games()

        if games.empty:
            return None

        game = games[games['game_id'] == game_id]

        if game.empty:
            return None

        return game.iloc[0].to_dict()

    def save_game_state(self, game: Dict) -> bool:
        """
        Save game state snapshot to database.

        Args:
            game: Game state dict

        Returns:
            True if saved successfully
        """
        try:
            conn = sqlite3.connect(DB_PATH)

            # Calculate quarter and time remaining
            quarter = game.get('period', 0)
            game_clock = game.get('game_clock', '')
            time_remaining = game_clock if isinstance(game_clock, str) else ''

            # Parse time to seconds for calculations
            time_remaining_seconds = self._parse_time_to_seconds(time_remaining)

            conn.execute("""
                INSERT OR IGNORE INTO live_game_state
                (game_id, timestamp, game_date, quarter, time_remaining,
                 home_score, away_score, home_team, away_team, game_status,
                 game_clock, period_time_remaining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game['game_id'],
                datetime.now().isoformat(),
                game.get('game_date', datetime.now().strftime('%Y-%m-%d')),
                quarter,
                time_remaining,
                game.get('home_score', 0),
                game.get('away_score', 0),
                game.get('home_team', ''),
                game.get('away_team', ''),
                game.get('game_status_text', ''),
                time_remaining_seconds,
                time_remaining
            ))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            return False

    def _parse_time_to_seconds(self, time_str: str) -> int:
        """
        Parse time string like "7:23" to seconds.

        Args:
            time_str: Time in "M:SS" format

        Returns:
            Seconds remaining in quarter
        """
        if not time_str or time_str == '':
            return 0

        try:
            # Handle both "M:SS" and "MM:SS" formats
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except (ValueError, IndexError):
            logger.warning(f"Could not parse time: {time_str}")

        return 0

    def get_historical_game_states(
        self,
        game_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical game state snapshots for a game.

        Args:
            game_id: Game ID
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)

        Returns:
            DataFrame of game states
        """
        conn = sqlite3.connect(DB_PATH)

        query = "SELECT * FROM live_game_state WHERE game_id = ?"
        params = [game_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_game_state_at_time(self, game_id: str, timestamp: str) -> Optional[Dict]:
        """
        Get game state at specific time (or closest snapshot).

        Args:
            game_id: Game ID
            timestamp: Timestamp (ISO format)

        Returns:
            Dict with game state or None
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # Get closest snapshot before or at this time
        result = conn.execute("""
            SELECT * FROM live_game_state
            WHERE game_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (game_id, timestamp)).fetchone()

        conn.close()

        if result:
            return dict(result)

        return None


if __name__ == "__main__":
    # Test the client
    client = LiveGameClient()

    print("=== Fetching Today's Games ===")
    games = client.get_todays_games()

    if games.empty:
        print("No games today")
    else:
        print(f"\nFound {len(games)} games:\n")
        for _, game in games.iterrows():
            status = game['game_status_text']
            if game['game_status_id'] == 2:  # Live
                print(f"🔴 LIVE: {game['away_team']} @ {game['home_team']}")
                print(f"   Score: {game['away_score']}-{game['home_score']}")
                print(f"   Q{game['period']} {game['game_clock']}")
            elif game['game_status_id'] == 3:  # Final
                print(f"✓ FINAL: {game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")
            else:  # Not started
                print(f"⏰ {game['away_team']} @ {game['home_team']} - {status}")

        # Test saving live games
        live_games = client.get_live_games()
        if not live_games.empty:
            print(f"\n=== Saving {len(live_games)} live games ===")
            for _, game in live_games.iterrows():
                success = client.save_game_state(game.to_dict())
                if success:
                    print(f"✓ Saved: {game['game_id']}")
