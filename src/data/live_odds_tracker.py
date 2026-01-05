"""
Live Odds Tracker

Track live betting odds from The Odds API.
Fetches and stores odds movements during games.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

from src.data.odds_api import OddsAPIClient
from src.data.live_betting_db import DB_PATH
from src.utils.constants import ABBREV_TO_TEAM_NAME


class LiveOddsTracker:
    """Track live betting odds."""

    def __init__(self):
        self.odds_client = OddsAPIClient()

    def get_live_odds(self, game_ids: Optional[List[str]] = None, teams: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Fetch current odds for live games.

        Args:
            game_ids: Optional list of game IDs to filter (not used for team matching)
            teams: Optional list of (home_team, away_team) tuples to filter

        Returns:
            DataFrame with live odds
        """
        # Get all current odds
        odds = self.odds_client.get_current_odds(
            markets=["h2h", "spreads", "totals"]
        )

        if odds.empty:
            logger.warning("No odds available")
            return odds

        # Filter to live games (commence_time in past)
        # Use pandas Timestamp for timezone-aware comparison
        now = pd.Timestamp.now(tz='UTC')

        # Convert commence_time to datetime if it's not already
        if 'commence_time' in odds.columns:
            if not pd.api.types.is_datetime64_any_dtype(odds['commence_time']):
                odds['commence_datetime'] = pd.to_datetime(odds['commence_time'], utc=True)
            else:
                odds['commence_datetime'] = pd.to_datetime(odds['commence_time'], utc=True)

            # Games that have started (but likely not finished yet - within 4 hours)
            time_threshold = now - pd.Timedelta(hours=4)
            live_odds = odds[
                (odds['commence_datetime'] <= now) &
                (odds['commence_datetime'] >= time_threshold)
            ].copy()

            logger.info(f"Filtered to {len(live_odds)} live game odds")

            # Filter to specific teams if provided
            if teams:
                # Convert abbreviations to full names for matching
                # teams is a list of (home_abbrev, away_abbrev) tuples
                full_name_teams = []
                for home_abbrev, away_abbrev in teams:
                    home_full = ABBREV_TO_TEAM_NAME.get(home_abbrev, home_abbrev)
                    away_full = ABBREV_TO_TEAM_NAME.get(away_abbrev, away_abbrev)
                    full_name_teams.append((home_full, away_full))
                    logger.debug(f"Matching {home_abbrev} @ {away_abbrev} as {home_full} vs {away_full}")

                # Create team pair matching
                def matches_team_pair(row):
                    for home, away in full_name_teams:
                        if row['home_team'] == home and row['away_team'] == away:
                            return True
                    return False

                live_odds = live_odds[live_odds.apply(matches_team_pair, axis=1)].copy()
                logger.info(f"Filtered to {len(live_odds)} requested games")

            return live_odds

        return pd.DataFrame()

    def save_odds_snapshot(self, odds_df: pd.DataFrame) -> int:
        """
        Save odds snapshot to database.

        Args:
            odds_df: DataFrame of odds in long format (from odds_api)

        Returns:
            Number of rows saved
        """
        if odds_df.empty:
            return 0

        conn = sqlite3.connect(DB_PATH)
        timestamp = datetime.now().isoformat()
        saved_count = 0

        try:
            # Group by game_id, bookmaker, market to pivot from long to wide format
            for (game_id, bookmaker, market), group in odds_df.groupby(['game_id', 'bookmaker', 'market']):
                # Initialize all odds columns
                home_odds = None
                away_odds = None
                over_odds = None
                under_odds = None
                spread_value = None
                total_value = None

                # Extract values based on market type
                if market == 'moneyline':
                    # Moneyline - find home and away odds
                    for _, row in group.iterrows():
                        if row.get('team') == 'home':
                            home_odds = row.get('odds')
                        elif row.get('team') == 'away':
                            away_odds = row.get('odds')
                elif market == 'spread':
                    # Spread - find home and away lines + odds
                    for _, row in group.iterrows():
                        if row.get('team') == 'home':
                            spread_value = row.get('line')
                            home_odds = row.get('odds')
                        elif row.get('team') == 'away':
                            away_odds = row.get('odds')
                elif market == 'total':
                    # Total - find over/under lines + odds
                    for _, row in group.iterrows():
                        if row.get('team') == 'over':
                            total_value = row.get('line')
                            over_odds = row.get('odds')
                        elif row.get('team') == 'under':
                            under_odds = row.get('odds')

                conn.execute("""
                    INSERT OR IGNORE INTO live_odds_snapshot
                    (game_id, timestamp, bookmaker, market,
                     home_odds, away_odds, over_odds, under_odds,
                     spread_value, total_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    timestamp,
                    bookmaker,
                    market,
                    home_odds,
                    away_odds,
                    over_odds,
                    under_odds,
                    spread_value,
                    total_value
                ))
                saved_count += 1

            conn.commit()
            logger.info(f"Saved {saved_count} odds snapshots")

        except Exception as e:
            logger.error(f"Error saving odds snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()

        return saved_count

    def get_odds_history(
        self,
        game_id: str,
        market: Optional[str] = None,
        bookmaker: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical odds snapshots for a game.

        Args:
            game_id: Game ID
            market: Optional market filter ('h2h', 'spreads', 'totals')
            bookmaker: Optional bookmaker filter
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)

        Returns:
            DataFrame of odds snapshots
        """
        conn = sqlite3.connect(DB_PATH)

        query = "SELECT * FROM live_odds_snapshot WHERE game_id = ?"
        params = [game_id]

        if market:
            query += " AND market = ?"
            params.append(market)

        if bookmaker:
            query += " AND bookmaker = ?"
            params.append(bookmaker)

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

    def get_latest_odds(self, game_id: str) -> Dict[str, Dict]:
        """
        Get latest odds snapshot for a game.

        Args:
            game_id: Game ID

        Returns:
            Dict with format:
            {
                'spread': {'spread_value': -5.5, 'home_odds': -110, ...},
                'moneyline': {'home_odds': -200, 'away_odds': +170, ...},
                'total': {'total_value': 215.5, 'over_odds': -110, ...}
            }
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        result = {}

        # Get latest spread
        spread = conn.execute("""
            SELECT * FROM live_odds_snapshot
            WHERE game_id = ? AND market = 'spread'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (game_id,)).fetchone()

        if spread:
            result['spread'] = {
                'spread_value': spread['spread_value'],
                'home_odds': spread['home_odds'],
                'away_odds': spread['away_odds'],
                'bookmaker': spread['bookmaker'],
                'timestamp': spread['timestamp']
            }

        # Get latest moneyline
        moneyline = conn.execute("""
            SELECT * FROM live_odds_snapshot
            WHERE game_id = ? AND market = 'moneyline'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (game_id,)).fetchone()

        if moneyline:
            result['moneyline'] = {
                'home_odds': moneyline['home_odds'],
                'away_odds': moneyline['away_odds'],
                'bookmaker': moneyline['bookmaker'],
                'timestamp': moneyline['timestamp']
            }

        # Get latest total
        total = conn.execute("""
            SELECT * FROM live_odds_snapshot
            WHERE game_id = ? AND market = 'total'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (game_id,)).fetchone()

        if total:
            result['total'] = {
                'total_value': total['total_value'],
                'over_odds': total['over_odds'],
                'under_odds': total['under_odds'],
                'bookmaker': total['bookmaker'],
                'timestamp': total['timestamp']
            }

        conn.close()

        return result

    def get_odds_movement(
        self,
        game_id: str,
        market: str = 'spreads'
    ) -> pd.DataFrame:
        """
        Get odds movement over time for a game.

        Args:
            game_id: Game ID
            market: Market type ('spreads', 'h2h', 'totals')

        Returns:
            DataFrame with timestamp and odds values
        """
        df = self.get_odds_history(game_id, market=market)

        if df.empty:
            return df

        # Simplify to key columns based on market
        if market == 'spreads':
            df = df[['timestamp', 'spread_value', 'home_odds', 'away_odds', 'bookmaker']]
        elif market == 'h2h':
            df = df[['timestamp', 'home_odds', 'away_odds', 'bookmaker']]
        elif market == 'totals':
            df = df[['timestamp', 'total_value', 'over_odds', 'under_odds', 'bookmaker']]

        return df

    def detect_line_movement(
        self,
        game_id: str,
        market: str = 'spreads',
        min_movement: float = 1.0
    ) -> Optional[Dict]:
        """
        Detect significant line movements.

        Args:
            game_id: Game ID
            market: Market type
            min_movement: Minimum movement to flag (in points for spreads/totals)

        Returns:
            Dict with movement info or None if no significant movement
        """
        history = self.get_odds_movement(game_id, market)

        if len(history) < 2:
            return None

        # Compare first and last snapshot
        first = history.iloc[0]
        last = history.iloc[-1]

        movement = None

        if market == 'spreads':
            first_spread = first['spread_value']
            last_spread = last['spread_value']
            change = last_spread - first_spread

            if abs(change) >= min_movement:
                movement = {
                    'market': market,
                    'original_spread': first_spread,
                    'current_spread': last_spread,
                    'movement': change,
                    'direction': 'toward_home' if change < 0 else 'toward_away',
                    'snapshots': len(history)
                }

        elif market == 'totals':
            first_total = first['total_value']
            last_total = last['total_value']
            change = last_total - first_total

            if abs(change) >= min_movement:
                movement = {
                    'market': market,
                    'original_total': first_total,
                    'current_total': last_total,
                    'movement': change,
                    'direction': 'up' if change > 0 else 'down',
                    'snapshots': len(history)
                }

        return movement


if __name__ == "__main__":
    # Test the tracker
    tracker = LiveOddsTracker()

    print("=== Fetching Live Odds ===")
    live_odds = tracker.get_live_odds()

    if live_odds.empty:
        print("No live games with odds available")
        print("(Games haven't started yet or odds API has no live data)")
    else:
        print(f"\nFound odds for {live_odds['game_id'].nunique()} games")

        # Show sample
        print("\nSample odds:")
        print(live_odds[['game_id', 'home_team', 'away_team', 'market', 'bookmaker']].head(10))

        # Save to database
        print("\n=== Saving Odds Snapshot ===")
        saved = tracker.save_odds_snapshot(live_odds)
        print(f"Saved {saved} odds records")

        # Test retrieval
        if not live_odds.empty:
            sample_game = live_odds.iloc[0]['game_id']
            print(f"\n=== Latest Odds for Game {sample_game} ===")
            latest = tracker.get_latest_odds(sample_game)
            for market, odds in latest.items():
                print(f"\n{market.upper()}:")
                for key, value in odds.items():
                    print(f"  {key}: {value}")
