"""
Odds API Client

Fetches current and historical betting odds from The Odds API.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import requests
from loguru import logger


class OddsAPIClient:
    """Client for fetching NBA betting odds."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"

    # Common US bookmakers
    BOOKMAKERS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "pointsbet",
        "betrivers",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY") or os.getenv("ODDS_API")
        if not self.api_key:
            logger.warning("No ODDS_API_KEY found. Set it in .env")
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a request to the API."""
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        response = self.session.get(url, params=params)

        # Log remaining requests
        remaining = response.headers.get("x-requests-remaining", "?")
        logger.debug(f"API requests remaining: {remaining}")

        response.raise_for_status()
        return response.json()

    def get_current_odds(
        self,
        markets: List[str] = None,
        bookmakers: List[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch current odds for upcoming NBA games.

        Args:
            markets: List of markets (h2h, spreads, totals)
            bookmakers: List of bookmakers to include

        Returns:
            DataFrame with current odds
        """
        markets = markets or ["h2h", "spreads", "totals"]
        bookmakers = bookmakers or self.BOOKMAKERS

        params = {
            "regions": "us",
            "markets": ",".join(markets),
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "american",
        }

        data = self._make_request(f"sports/{self.SPORT}/odds", params)
        return self._parse_odds(data)

    def _parse_odds(self, games: list) -> pd.DataFrame:
        """Parse raw odds data into a structured DataFrame."""
        records = []

        for game in games:
            game_id = game["id"]
            commence_time = game["commence_time"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker["key"]
                last_update = bookmaker["last_update"]

                for market in bookmaker.get("markets", []):
                    market_key = market["key"]

                    if market_key == "h2h":
                        # Moneyline
                        for outcome in market["outcomes"]:
                            is_home = outcome["name"] == home_team
                            records.append({
                                "game_id": game_id,
                                "commence_time": commence_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": book_name,
                                "market": "moneyline",
                                "team": "home" if is_home else "away",
                                "line": None,
                                "odds": outcome["price"],
                                "last_update": last_update,
                            })

                    elif market_key == "spreads":
                        # Point spread
                        for outcome in market["outcomes"]:
                            is_home = outcome["name"] == home_team
                            records.append({
                                "game_id": game_id,
                                "commence_time": commence_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": book_name,
                                "market": "spread",
                                "team": "home" if is_home else "away",
                                "line": outcome.get("point"),
                                "odds": outcome["price"],
                                "last_update": last_update,
                            })

                    elif market_key == "totals":
                        # Over/under
                        for outcome in market["outcomes"]:
                            records.append({
                                "game_id": game_id,
                                "commence_time": commence_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": book_name,
                                "market": "total",
                                "team": outcome["name"].lower(),  # "Over" or "Under"
                                "line": outcome.get("point"),
                                "odds": outcome["price"],
                                "last_update": last_update,
                            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["last_update"] = pd.to_datetime(df["last_update"])
            df["implied_prob"] = df["odds"].apply(self._american_to_implied_prob)

        return df

    @staticmethod
    def _american_to_implied_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def _american_to_decimal(odds: int) -> float:
        """Convert American odds to decimal odds."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    def get_historical_odds(
        self,
        date: str,
        markets: List[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical odds for a specific date.

        Note: Requires paid plan for The Odds API.

        Args:
            date: Date in YYYY-MM-DD format
            markets: List of markets to fetch

        Returns:
            DataFrame with historical odds
        """
        markets = markets or ["h2h", "spreads", "totals"]

        # Convert date to ISO format
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_iso = dt.strftime("%Y-%m-%dT12:00:00Z")

        params = {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": date_iso,
        }

        try:
            data = self._make_request(
                f"historical/sports/{self.SPORT}/odds",
                params
            )
            return self._parse_odds(data.get("data", []))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Historical odds require a paid API plan")
            raise

    def get_best_odds(self, current_odds: pd.DataFrame) -> pd.DataFrame:
        """
        Find the best available odds across all bookmakers.

        Args:
            current_odds: DataFrame from get_current_odds()

        Returns:
            DataFrame with best odds for each game/market/team
        """
        if current_odds.empty:
            return current_odds

        # For spreads and totals, group by line as well
        best_odds = (
            current_odds
            .sort_values("odds", ascending=False)
            .groupby(["game_id", "market", "team", "line"])
            .first()
            .reset_index()
        )

        return best_odds

    def calculate_no_vig_odds(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate no-vig (fair) probabilities by removing bookmaker margin.

        Args:
            odds_df: DataFrame with odds data

        Returns:
            DataFrame with added no_vig_prob column
        """
        if odds_df.empty:
            return odds_df

        df = odds_df.copy()

        # For each game/market/bookmaker, calculate total implied prob
        # and normalize to remove vig
        grouped = df.groupby(["game_id", "market", "bookmaker"])

        results = []
        for name, group in grouped:
            total_implied = group["implied_prob"].sum()
            group = group.copy()
            group["no_vig_prob"] = group["implied_prob"] / total_implied
            group["vig"] = total_implied - 1
            results.append(group)

        return pd.concat(results, ignore_index=True)


def fetch_and_save_current_odds(output_path: str = "data/raw/odds_current.parquet"):
    """Fetch current odds and save to file."""
    client = OddsAPIClient()
    odds = client.get_current_odds()

    if not odds.empty:
        odds = client.calculate_no_vig_odds(odds)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        odds.to_parquet(output_path)
        logger.info(f"Saved {len(odds)} odds records to {output_path}")

    return odds


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("config/.env")

    client = OddsAPIClient()

    # Fetch current odds
    odds = client.get_current_odds()
    print(f"Fetched odds for {odds['game_id'].nunique()} games")

    if not odds.empty:
        # Calculate no-vig odds
        odds = client.calculate_no_vig_odds(odds)
        print("\nSample odds with no-vig probabilities:")
        print(odds[["home_team", "away_team", "market", "team", "odds", "implied_prob", "no_vig_prob"]].head(10))

        # Find best odds
        best = client.get_best_odds(odds)
        print(f"\nBest odds available:")
        print(best[["home_team", "away_team", "market", "team", "odds", "bookmaker"]].head(10))
