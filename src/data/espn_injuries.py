"""
ESPN NBA Injuries and Player Impact Calculator

Fetches injury reports from ESPN and calculates player impact using NBA API stats.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import requests
from loguru import logger


class ESPNClient:
    """Client for fetching NBA injury data from ESPN's API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    # Team name to abbreviation mapping
    TEAM_NAME_TO_ABBREV = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "LA Clippers": "LAC",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "LA Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

    def _make_request(self, url: str, params: dict = None) -> dict:
        """Make a request with error handling."""
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.5)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ESPN API request failed: {e}")
            return {}

    def get_injuries(self) -> pd.DataFrame:
        """
        Fetch current NBA injury report.

        Returns:
            DataFrame with: player_name, team, status, injury_type, details
        """
        url = f"{self.BASE_URL}/injuries"
        data = self._make_request(url)

        records = []
        for team_data in data.get("injuries", []):
            # Get team abbreviation from displayName
            team_full_name = team_data.get("displayName", "")
            team_abbrev = self.TEAM_NAME_TO_ABBREV.get(team_full_name, "UNK")

            for injury in team_data.get("injuries", []):
                athlete = injury.get("athlete", {})

                # Parse status - normalize to standard categories
                status = injury.get("status", "Unknown")
                if status.lower() in ["day-to-day", "day to day"]:
                    status = "Day-To-Day"

                records.append({
                    "player_id": athlete.get("id"),
                    "player_name": athlete.get("displayName"),
                    "team": team_abbrev,
                    "position": athlete.get("position", {}).get("abbreviation") if isinstance(athlete.get("position"), dict) else None,
                    "status": status,
                    "injury_type": injury.get("type", {}).get("description") if isinstance(injury.get("type"), dict) else injury.get("type"),
                    "details": injury.get("shortComment", ""),
                    "long_details": injury.get("longComment", ""),
                    "injury_date": injury.get("date"),
                })

        df = pd.DataFrame(records)
        if not df.empty:
            logger.info(f"Fetched {len(df)} injury records for {df['team'].nunique()} teams")
        return df

    def get_scoreboard(self, date: str = None) -> pd.DataFrame:
        """Get today's games."""
        url = f"{self.BASE_URL}/scoreboard"
        params = {"dates": date} if date else {}
        data = self._make_request(url, params)

        records = []
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            records.append({
                "game_id": event.get("id"),
                "date": event.get("date"),
                "status": event.get("status", {}).get("type", {}).get("name"),
                "home_team": home.get("team", {}).get("abbreviation"),
                "away_team": away.get("team", {}).get("abbreviation"),
            })

        return pd.DataFrame(records)


class PlayerStatsCache:
    """Cache for player statistics from NBA API."""

    # Fallback stats for star players who might be missing from current season
    # Format: {name_lower: {ppg, rpg, apg, mpg, plus_minus}}
    STAR_PLAYER_FALLBACK = {
        # MVPs and All-NBA caliber
        "nikola jokic": {"ppg": 26.5, "rpg": 12.5, "apg": 9.0, "mpg": 35.0, "plus_minus": 8.0},
        "giannis antetokounmpo": {"ppg": 31.0, "rpg": 12.0, "apg": 6.0, "mpg": 35.0, "plus_minus": 6.0},
        "luka doncic": {"ppg": 28.5, "rpg": 9.0, "apg": 8.5, "mpg": 36.0, "plus_minus": 4.0},
        "shai gilgeous-alexander": {"ppg": 31.0, "rpg": 5.5, "apg": 6.0, "mpg": 34.0, "plus_minus": 9.0},
        "jayson tatum": {"ppg": 27.0, "rpg": 8.0, "apg": 5.0, "mpg": 36.0, "plus_minus": 7.0},
        "joel embiid": {"ppg": 34.0, "rpg": 11.0, "apg": 6.0, "mpg": 34.0, "plus_minus": 8.0},
        "kevin durant": {"ppg": 27.0, "rpg": 6.5, "apg": 5.0, "mpg": 35.0, "plus_minus": 4.0},
        "lebron james": {"ppg": 25.5, "rpg": 7.5, "apg": 8.0, "mpg": 35.0, "plus_minus": 3.0},
        "stephen curry": {"ppg": 27.0, "rpg": 4.5, "apg": 5.0, "mpg": 32.0, "plus_minus": 5.0},
        "anthony davis": {"ppg": 24.5, "rpg": 12.5, "apg": 3.5, "mpg": 35.0, "plus_minus": 5.0},
        "anthony edwards": {"ppg": 26.0, "rpg": 5.5, "apg": 5.0, "mpg": 35.0, "plus_minus": 3.0},
        "donovan mitchell": {"ppg": 27.0, "rpg": 5.0, "apg": 5.5, "mpg": 34.0, "plus_minus": 5.0},
        "devin booker": {"ppg": 27.0, "rpg": 4.5, "apg": 6.5, "mpg": 35.0, "plus_minus": 3.0},
        "trae young": {"ppg": 26.0, "rpg": 3.0, "apg": 10.5, "mpg": 35.0, "plus_minus": -1.0},
        "ja morant": {"ppg": 25.0, "rpg": 5.5, "apg": 8.0, "mpg": 32.0, "plus_minus": 2.0},
        "damian lillard": {"ppg": 25.0, "rpg": 4.5, "apg": 7.0, "mpg": 35.0, "plus_minus": 2.0},
        "kyrie irving": {"ppg": 25.5, "rpg": 5.0, "apg": 5.5, "mpg": 34.0, "plus_minus": 3.0},
        "jimmy butler": {"ppg": 21.0, "rpg": 6.0, "apg": 5.5, "mpg": 33.0, "plus_minus": 4.0},
        "kawhi leonard": {"ppg": 24.0, "rpg": 6.5, "apg": 4.0, "mpg": 34.0, "plus_minus": 5.0},
        "paul george": {"ppg": 23.0, "rpg": 5.5, "apg": 4.5, "mpg": 33.0, "plus_minus": 3.0},
        "zion williamson": {"ppg": 23.0, "rpg": 5.5, "apg": 4.5, "mpg": 30.0, "plus_minus": 1.0},
        "chet holmgren": {"ppg": 17.0, "rpg": 8.0, "apg": 2.5, "mpg": 30.0, "plus_minus": 6.0},
        "victor wembanyama": {"ppg": 24.0, "rpg": 10.5, "apg": 4.0, "mpg": 32.0, "plus_minus": 4.0},
        "tyrese haliburton": {"ppg": 20.0, "rpg": 4.0, "apg": 10.5, "mpg": 33.0, "plus_minus": 4.0},
        "paolo banchero": {"ppg": 23.0, "rpg": 7.0, "apg": 5.0, "mpg": 34.0, "plus_minus": 2.0},
        "franz wagner": {"ppg": 24.0, "rpg": 5.5, "apg": 5.5, "mpg": 34.0, "plus_minus": 3.0},
        "jalen brunson": {"ppg": 28.0, "rpg": 3.5, "apg": 6.5, "mpg": 35.0, "plus_minus": 5.0},
        "domantas sabonis": {"ppg": 19.5, "rpg": 14.0, "apg": 8.0, "mpg": 35.0, "plus_minus": 3.0},
        "tyler herro": {"ppg": 24.0, "rpg": 5.0, "apg": 5.0, "mpg": 33.0, "plus_minus": 2.0},
        "jamal murray": {"ppg": 21.0, "rpg": 4.0, "apg": 6.5, "mpg": 33.0, "plus_minus": 4.0},
        "de'aaron fox": {"ppg": 26.5, "rpg": 4.5, "apg": 6.0, "mpg": 35.0, "plus_minus": 2.0},
        "lauri markkanen": {"ppg": 23.5, "rpg": 8.0, "apg": 2.0, "mpg": 33.0, "plus_minus": 1.0},
        "fred vanvleet": {"ppg": 14.5, "rpg": 3.5, "apg": 7.0, "mpg": 33.0, "plus_minus": 2.0},
        "alperen sengun": {"ppg": 21.0, "rpg": 10.0, "apg": 5.0, "mpg": 32.0, "plus_minus": 4.0},
        "jalen green": {"ppg": 21.0, "rpg": 5.0, "apg": 3.5, "mpg": 32.0, "plus_minus": 1.0},
        "draymond green": {"ppg": 9.0, "rpg": 7.0, "apg": 6.0, "mpg": 28.0, "plus_minus": 5.0},
        # Key role players
        "jonas valanciunas": {"ppg": 12.0, "rpg": 9.0, "apg": 2.0, "mpg": 23.0, "plus_minus": 1.0},
        "austin reaves": {"ppg": 20.0, "rpg": 4.5, "apg": 6.5, "mpg": 35.0, "plus_minus": 3.0},
        "cameron johnson": {"ppg": 18.5, "rpg": 4.5, "apg": 3.0, "mpg": 31.0, "plus_minus": 2.0},
        "christian braun": {"ppg": 15.0, "rpg": 5.0, "apg": 3.0, "mpg": 32.0, "plus_minus": 4.0},
        "aaron gordon": {"ppg": 14.5, "rpg": 6.5, "apg": 3.5, "mpg": 30.0, "plus_minus": 4.0},
        "ivica zubac": {"ppg": 17.0, "rpg": 12.0, "apg": 2.5, "mpg": 30.0, "plus_minus": 3.0},
        "bradley beal": {"ppg": 17.0, "rpg": 4.0, "apg": 5.0, "mpg": 32.0, "plus_minus": 0.0},
        "isaiah hartenstein": {"ppg": 11.0, "rpg": 10.0, "apg": 3.5, "mpg": 28.0, "plus_minus": 5.0},
        "zach lavine": {"ppg": 23.0, "rpg": 5.0, "apg": 4.5, "mpg": 34.0, "plus_minus": 0.0},
        "coby white": {"ppg": 20.0, "rpg": 4.0, "apg": 5.5, "mpg": 34.0, "plus_minus": 1.0},
        "devin vassell": {"ppg": 16.5, "rpg": 4.0, "apg": 3.5, "mpg": 30.0, "plus_minus": 2.0},
        "dejounte murray": {"ppg": 17.5, "rpg": 5.5, "apg": 6.5, "mpg": 33.0, "plus_minus": 1.0},
        "josh hart": {"ppg": 13.5, "rpg": 8.5, "apg": 5.0, "mpg": 33.0, "plus_minus": 3.0},
    }

    def __init__(self):
        self._stats_df = None
        self._last_refresh = None
        self._refresh_interval = timedelta(hours=12)

    def refresh(self, force: bool = False) -> pd.DataFrame:
        """Refresh player stats from NBA API."""
        now = datetime.now()
        if not force and self._stats_df is not None:
            if (now - self._last_refresh) < self._refresh_interval:
                return self._stats_df

        try:
            from nba_api.stats.endpoints import leaguedashplayerstats

            logger.info("Fetching player stats from NBA API...")
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season='2024-25',
                per_mode_detailed='PerGame'
            )
            df = stats.get_data_frames()[0]

            # Calculate per-game averages (already per-game but ensure)
            self._stats_df = df[[
                'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION',
                'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS'
            ]].copy()

            # Rename columns
            self._stats_df.columns = [
                'player_id', 'player_name', 'team',
                'gp', 'mpg', 'ppg', 'rpg', 'apg', 'spg', 'bpg', 'tov',
                'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'plus_minus'
            ]

            self._last_refresh = now
            logger.info(f"Cached stats for {len(self._stats_df)} players")

        except Exception as e:
            logger.error(f"Failed to fetch NBA stats: {e}")
            if self._stats_df is None:
                self._stats_df = pd.DataFrame()

        return self._stats_df

    def get_player_stats(self, player_name: str, team: str = None) -> Optional[dict]:
        """
        Get stats for a specific player.

        Args:
            player_name: Player name to look up
            team: Team abbreviation to verify (optional but recommended)
        """
        df = self.refresh()

        name_lower = player_name.lower().strip()

        # Try to find in NBA API data first
        if not df.empty:
            # Try exact match first
            match = df[df['player_name'].str.lower() == name_lower]

            # Try partial match if no exact match
            if match.empty:
                last_name = name_lower.split()[-1] if name_lower else ""
                if last_name:
                    match = df[df['player_name'].str.lower().str.contains(last_name, na=False)]

            if not match.empty:
                result = match.iloc[0].to_dict()

                # If team mismatch, trust NBA API and correct the team
                if team and result.get('team') != team:
                    logger.info(
                        f"Team correction for {player_name}: ESPN says {team}, "
                        f"NBA API says {result.get('team')} - using NBA API team"
                    )
                    # Return with corrected team (NBA API is source of truth)
                    result['espn_team'] = team  # Keep original for reference
                    result['team_corrected'] = True

                return result

        # Check star player fallback dictionary
        if name_lower in self.STAR_PLAYER_FALLBACK:
            logger.debug(f"Using fallback stats for {player_name}")
            return self.STAR_PLAYER_FALLBACK[name_lower]

        # Try partial match on fallback (for variations like "Nikola Jokić")
        for fallback_name, stats in self.STAR_PLAYER_FALLBACK.items():
            # Check if last names match
            if name_lower.split()[-1] in fallback_name or fallback_name.split()[-1] in name_lower:
                logger.debug(f"Using fallback stats for {player_name} (matched {fallback_name})")
                return stats

        return None


class PlayerImpactCalculator:
    """
    Calculate player impact scores for injury adjustment.

    Uses a simplified approach based on:
    - Points/Rebounds/Assists production
    - Minutes played (proxy for role importance)
    - Plus/minus (net impact on court)
    """

    # Status to probability of missing the game
    STATUS_MISS_PROB = {
        "Out": 1.0,
        "Doubtful": 0.9,
        "Questionable": 0.6,
        "Day-To-Day": 0.5,
        "Probable": 0.15,
        "Available": 0.0,
        "Unknown": 0.5,
    }

    def __init__(self):
        self.stats_cache = PlayerStatsCache()

    def calculate_player_impact(self, player_stats: dict) -> float:
        """
        Calculate a player's point impact score.

        This estimates how many points per game the team loses when this player is out.
        Based on research: replacement players produce ~20-30% less than starters.

        Args:
            player_stats: Dict with player stats (ppg, rpg, apg, mpg, etc.)

        Returns:
            Estimated point swing when player is out
        """
        if not player_stats:
            return 2.0  # Default for unknown players

        ppg = player_stats.get('ppg', 0) or 0
        rpg = player_stats.get('rpg', 0) or 0
        apg = player_stats.get('apg', 0) or 0
        mpg = player_stats.get('mpg', 0) or 0
        plus_minus = player_stats.get('plus_minus', 0) or 0

        # Minutes factor: scale impact by playing time (30-35 min = max impact)
        minutes_factor = min(mpg / 32, 1.0) if mpg > 0 else 0.3

        # Scoring impact: ~25-30% of scoring not replaced
        scoring_impact = ppg * 0.28

        # Assist impact: each assist ~1.5 points, ~35% not replaced
        assist_impact = apg * 1.5 * 0.35

        # Rebound impact: each rebound ~0.4 points in possession value
        rebound_impact = rpg * 0.4 * 0.25

        # Plus/minus bonus: adds context for defensive impact
        pm_impact = max(0, plus_minus * 0.1)

        base_impact = (scoring_impact + assist_impact + rebound_impact + pm_impact) * minutes_factor

        # Star player multiplier (stars are harder to replace)
        if ppg >= 25:
            base_impact *= 1.4  # MVP-level
        elif ppg >= 20:
            base_impact *= 1.25  # All-Star level
        elif ppg >= 15:
            base_impact *= 1.1  # Quality starter

        # Floor of 1.0 for any rotation player
        return max(round(base_impact, 2), 1.0)

    def get_team_injury_impact(
        self,
        team_abbrev: str,
        injuries_df: pd.DataFrame,
    ) -> dict:
        """
        Calculate total injury impact for a team.

        Uses NBA API as source of truth for team assignments, since ESPN
        sometimes lists players under incorrect teams.

        Returns:
            Dictionary with impact metrics and player details
        """
        if injuries_df.empty:
            return {
                "team": team_abbrev,
                "total_impact": 0.0,
                "expected_impact": 0.0,
                "num_injuries": 0,
                "injured_players": [],
                "star_out": False,
                "key_players_out": [],
            }

        total_impact = 0.0
        expected_impact = 0.0
        injured_players = []
        star_out = False
        key_players_out = []

        # Check ALL injuries - use NBA API to determine actual team
        for _, injury in injuries_df.iterrows():
            player_name = injury["player_name"]
            status = injury["status"]
            espn_team = injury["team"]

            # Get player stats - this returns actual team from NBA API
            player_stats = self.stats_cache.get_player_stats(player_name, team=espn_team)

            # Skip if player not found
            if player_stats is None:
                continue

            # Check if player actually belongs to this team (per NBA API)
            actual_team = player_stats.get('team', espn_team)
            if actual_team != team_abbrev:
                # Player doesn't belong to this team - skip
                continue

            # Calculate impact
            impact = self.calculate_player_impact(player_stats)

            # Get probability of missing game
            miss_prob = self.STATUS_MISS_PROB.get(status, 0.5)

            # Expected impact = impact * probability of missing
            exp_impact = impact * miss_prob

            total_impact += impact
            expected_impact += exp_impact

            # Track stars and key players
            if impact >= 7:
                star_out = True
                key_players_out.append(player_name)
            elif impact >= 4:
                key_players_out.append(player_name)

            injured_players.append({
                "name": player_name,
                "status": status,
                "injury": injury.get("injury_type"),
                "ppg": player_stats.get('ppg') if player_stats else None,
                "mpg": player_stats.get('mpg') if player_stats else None,
                "impact": impact,
                "miss_prob": miss_prob,
                "expected_impact": exp_impact,
            })

        # Sort by expected impact
        injured_players.sort(key=lambda x: x['expected_impact'], reverse=True)

        return {
            "team": team_abbrev,
            "total_impact": round(total_impact, 2),
            "expected_impact": round(expected_impact, 2),
            "num_injuries": len(injured_players),
            "injured_players": injured_players,
            "star_out": star_out,
            "key_players_out": key_players_out,
        }


class InjuryFeatureBuilder:
    """Build injury-related features for the prediction model."""

    def __init__(self):
        self.espn = ESPNClient()
        self.impact_calc = PlayerImpactCalculator()
        self._injuries_cache = None
        self._cache_time = None
        self._cache_duration = timedelta(minutes=30)

    def refresh_data(self, force: bool = False):
        """Refresh injury data."""
        now = datetime.now()
        if not force and self._cache_time:
            if (now - self._cache_time) < self._cache_duration:
                return

        logger.info("Fetching latest injury data...")
        self._injuries_cache = self.espn.get_injuries()

        # Pre-warm the stats cache
        self.impact_calc.stats_cache.refresh()
        self._cache_time = now

    def get_injured_player_ids(self, team: str) -> List[int]:
        """
        Get list of NBA API player IDs for injured players on a team.

        Args:
            team: Team abbreviation (e.g., 'LAL')

        Returns:
            List of NBA API player IDs for injured players
        """
        self.refresh_data()

        if self._injuries_cache is None or self._injuries_cache.empty:
            return []

        # Get injured players for this team (Out or Doubtful)
        team_injuries = self._injuries_cache[
            (self._injuries_cache['team'] == team) &
            (self._injuries_cache['status'].isin(['Out', 'Doubtful']))
        ]

        if team_injuries.empty:
            return []

        # Get NBA API player IDs by matching names
        stats_df = self.impact_calc.stats_cache.refresh()
        if stats_df is None or stats_df.empty:
            return []

        player_ids = []
        for _, injury in team_injuries.iterrows():
            player_name = injury['player_name']
            # Match by player name (case-insensitive)
            matched = stats_df[
                stats_df['player_name'].str.lower() == player_name.lower()
            ]
            if not matched.empty:
                player_ids.append(int(matched.iloc[0]['player_id']))

        return player_ids

    def get_game_injury_features(
        self,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Get injury-related features for a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary of injury features
        """
        self.refresh_data()

        if self._injuries_cache is None or self._injuries_cache.empty:
            return self._empty_features()

        # Get impact for each team
        home_impact = self.impact_calc.get_team_injury_impact(
            home_team, self._injuries_cache
        )
        away_impact = self.impact_calc.get_team_injury_impact(
            away_team, self._injuries_cache
        )

        # Injury advantage: positive = away team more hurt = good for home
        injury_diff = away_impact["expected_impact"] - home_impact["expected_impact"]

        return {
            "home_injury_impact": home_impact["expected_impact"],
            "away_injury_impact": away_impact["expected_impact"],
            "injury_diff": round(injury_diff, 2),
            "home_star_out": int(home_impact["star_out"]),
            "away_star_out": int(away_impact["star_out"]),
            "home_num_injuries": home_impact["num_injuries"],
            "away_num_injuries": away_impact["num_injuries"],
            "home_key_players_out": home_impact["key_players_out"],
            "away_key_players_out": away_impact["key_players_out"],
        }

    def _empty_features(self) -> dict:
        """Return empty injury features."""
        return {
            "home_injury_impact": 0.0,
            "away_injury_impact": 0.0,
            "injury_diff": 0.0,
            "home_star_out": 0,
            "away_star_out": 0,
            "home_num_injuries": 0,
            "away_num_injuries": 0,
            "home_key_players_out": [],
            "away_key_players_out": [],
        }

    def get_full_injury_report(self) -> pd.DataFrame:
        """Get formatted injury report for all teams with impact scores."""
        self.refresh_data()

        if self._injuries_cache is None or self._injuries_cache.empty:
            return pd.DataFrame()

        # Get all unique teams
        teams = self._injuries_cache['team'].unique()

        records = []
        for team in teams:
            impact = self.impact_calc.get_team_injury_impact(
                team, self._injuries_cache
            )

            for player in impact["injured_players"]:
                records.append({
                    "team": team,
                    "player": player["name"],
                    "status": player["status"],
                    "injury": player.get("injury"),
                    "ppg": player.get("ppg"),
                    "mpg": player.get("mpg"),
                    "impact": player["impact"],
                    "miss_prob": f"{player['miss_prob']:.0%}",
                    "expected_impact": player["expected_impact"],
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("expected_impact", ascending=False)

        return df

    def get_team_summary(self) -> pd.DataFrame:
        """Get injury summary by team."""
        self.refresh_data()

        if self._injuries_cache is None or self._injuries_cache.empty:
            return pd.DataFrame()

        teams = self._injuries_cache['team'].unique()

        records = []
        for team in teams:
            impact = self.impact_calc.get_team_injury_impact(
                team, self._injuries_cache
            )
            records.append({
                "team": team,
                "num_injuries": impact["num_injuries"],
                "expected_impact": impact["expected_impact"],
                "star_out": impact["star_out"],
                "key_players": ", ".join(impact["key_players_out"][:3]) if impact["key_players_out"] else "-",
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("expected_impact", ascending=False)

        return df


def test_injury_tracking():
    """Test the injury tracking system."""
    print("=" * 60)
    print("Testing ESPN Injury Tracking with Player Impact")
    print("=" * 60)

    builder = InjuryFeatureBuilder()

    # Refresh data
    print("\n1. Fetching injury and player data...")
    builder.refresh_data(force=True)

    # Get full injury report
    print("\n2. Full Injury Report (Top 15 by Impact):")
    report = builder.get_full_injury_report()
    if not report.empty:
        display_cols = ["team", "player", "status", "ppg", "impact", "expected_impact"]
        print(report[display_cols].head(15).to_string(index=False))

    # Get team summary
    print("\n3. Team Impact Summary:")
    summary = builder.get_team_summary()
    if not summary.empty:
        print(summary.head(10).to_string(index=False))

    # Test specific game
    print("\n4. Game Features Example (LAL vs BOS):")
    features = builder.get_game_injury_features("LAL", "BOS")
    for k, v in features.items():
        if not k.endswith("_out") or v:  # Skip empty lists
            print(f"   {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_injury_tracking()
