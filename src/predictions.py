"""
NBA Game Predictions

Generate predictions for upcoming games using the trained model and real odds.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.data import NBAStatsClient, OddsAPIClient, InjuryFeatureBuilder
from src.features import GameFeatureBuilder
from src.features.elo import EloRatingSystem
from src.betting.kelly import KellyBetSizer
from src.betting.edge_strategy import EdgeStrategy, BetSignal, STRATEGY_PERFORMANCE
from src.models.injury_adjustment import InjuryAdjuster
from src.models.dual_model import DualPredictionModel


class NBAPredictor:
    """Generate predictions for NBA games."""

    def __init__(
        self,
        model_path: str = "models/spread_model.pkl",
        min_edge: float = 0.03,
        kelly_fraction: float = 0.2,
        use_injuries: bool = True,
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.use_injuries = use_injuries
        self.sizer = KellyBetSizer(fraction=kelly_fraction)

        # Load model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.feature_cols = model_data["feature_cols"]
        logger.info(f"Loaded model with {len(self.feature_cols)} features")

        # Initialize clients
        self.stats_client = NBAStatsClient()
        self.odds_client = OddsAPIClient()
        self.feature_builder = GameFeatureBuilder()

        # Initialize injury tracking and adjustment
        self.injury_builder = InjuryFeatureBuilder() if use_injuries else None
        self.injury_adjuster = InjuryAdjuster() if use_injuries else None

    def get_recent_games(self, days: int = 60) -> pd.DataFrame:
        """Fetch recent games for feature building."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        games = self.stats_client.get_games(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Fetched {len(games)} recent games")
        return games

    def get_current_odds(self) -> pd.DataFrame:
        """Fetch current odds for upcoming games."""
        odds = self.odds_client.get_current_odds()
        if not odds.empty:
            odds = self.odds_client.calculate_no_vig_odds(odds)
        logger.info(f"Fetched odds for {odds['game_id'].nunique()} upcoming games")
        return odds

    def build_features_for_upcoming(
        self,
        recent_games: pd.DataFrame,
        upcoming_odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for upcoming games based on recent performance."""
        # Build team features from recent games
        team_features = self.feature_builder.team_builder.build_all_features(recent_games)

        # Get most recent features for each team
        latest_features = (
            team_features
            .sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )

        # Create feature rows for upcoming games
        upcoming_games = upcoming_odds[["game_id", "home_team", "away_team", "commence_time"]].drop_duplicates()

        records = []
        for _, game in upcoming_games.iterrows():
            home_team = self._normalize_team_name(game["home_team"])
            away_team = self._normalize_team_name(game["away_team"])

            home_stats = latest_features[latest_features["team"] == home_team]
            away_stats = latest_features[latest_features["team"] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {away_team} @ {home_team}")
                continue

            # Build feature row
            record = {
                "game_id": game["game_id"],
                "commence_time": game["commence_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
            }

            # Add home team features
            for col in home_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"home_{col}"] = home_stats[col].values[0]

            # Add away team features
            for col in away_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"away_{col}"] = away_stats[col].values[0]

            records.append(record)

        df = pd.DataFrame(records)

        # Add differential features
        if not df.empty:
            df = self._add_differentials(df)

        return df

    def _normalize_team_name(self, name: str) -> str:
        """Convert full team name to abbreviation."""
        name_map = {
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
            "Los Angeles Clippers": "LAC",
            "Los Angeles Lakers": "LAL",
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
        return name_map.get(name, name)

    def _add_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add differential features."""
        windows = [5, 10, 20]

        for window in windows:
            suffix = f"_{window}g"
            for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate"]:
                home_col = f"home_{stat}{suffix}"
                away_col = f"away_{stat}{suffix}"
                if home_col in df.columns and away_col in df.columns:
                    df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

        if "home_travel_distance" in df.columns and "away_travel_distance" in df.columns:
            df["travel_diff"] = df["home_travel_distance"] - df["away_travel_distance"]

        return df

    def generate_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate predictions and identify +EV bets."""
        if features.empty:
            return pd.DataFrame()

        # Ensure all required features exist
        missing_cols = [c for c in self.feature_cols if c not in features.columns]
        for col in missing_cols:
            features[col] = 0  # Fill with 0 for missing features

        # Get predictions
        X = features[self.feature_cols].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]

        # Build predictions dataframe
        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions["model_home_prob"] = probs
        predictions["model_away_prob"] = 1 - probs

        # Add injury data and apply probability adjustment
        if self.use_injuries and self.injury_builder:
            predictions = self._add_injury_data(predictions)

            # Apply injury-based probability adjustment
            if self.injury_adjuster:
                predictions = self.injury_adjuster.adjust_predictions(predictions)

        # Merge with market odds (use average no-vig across books)
        market_probs = self._get_market_consensus(odds)
        predictions = predictions.merge(market_probs, on="game_id", how="left")

        # Calculate edge
        predictions["home_edge"] = predictions["model_home_prob"] - predictions["market_home_prob"]
        predictions["away_edge"] = predictions["model_away_prob"] - predictions["market_away_prob"]

        # Get best available odds
        best_odds = self._get_best_odds(odds)
        predictions = predictions.merge(best_odds, on="game_id", how="left")

        # Calculate Kelly bet sizing
        predictions["home_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_home_prob"], r["best_home_odds"])
            if pd.notna(r["best_home_odds"]) else 0,
            axis=1
        )
        predictions["away_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_away_prob"], r["best_away_odds"])
            if pd.notna(r["best_away_odds"]) else 0,
            axis=1
        )

        # Flag recommended bets
        predictions["bet_home"] = predictions["home_edge"] >= self.min_edge
        predictions["bet_away"] = predictions["away_edge"] >= self.min_edge

        return predictions

    def _add_injury_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add injury information to predictions."""
        injury_data = []

        for _, row in predictions.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            injury_features = self.injury_builder.get_game_injury_features(
                home_team, away_team
            )

            injury_data.append({
                "game_id": row["game_id"],
                "home_injury_impact": injury_features["home_injury_impact"],
                "away_injury_impact": injury_features["away_injury_impact"],
                "injury_diff": injury_features["injury_diff"],
                "home_star_out": injury_features["home_star_out"],
                "away_star_out": injury_features["away_star_out"],
                "home_injuries": injury_features["home_num_injuries"],
                "away_injuries": injury_features["away_num_injuries"],
                "home_key_out": ", ".join(injury_features["home_key_players_out"][:2]) if injury_features["home_key_players_out"] else "",
                "away_key_out": ", ".join(injury_features["away_key_players_out"][:2]) if injury_features["away_key_players_out"] else "",
            })

        injury_df = pd.DataFrame(injury_data)
        return predictions.merge(injury_df, on="game_id", how="left")

    def _get_market_consensus(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get average market probability across bookmakers."""
        ml_odds = odds[odds["market"] == "moneyline"].copy()

        home_probs = (
            ml_odds[ml_odds["team"] == "home"]
            .groupby("game_id")["no_vig_prob"]
            .mean()
            .reset_index()
            .rename(columns={"no_vig_prob": "market_home_prob"})
        )

        away_probs = (
            ml_odds[ml_odds["team"] == "away"]
            .groupby("game_id")["no_vig_prob"]
            .mean()
            .reset_index()
            .rename(columns={"no_vig_prob": "market_away_prob"})
        )

        return home_probs.merge(away_probs, on="game_id")

    def _get_best_odds(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get best available odds for each game."""
        ml_odds = odds[odds["market"] == "moneyline"].copy()

        best_home = (
            ml_odds[ml_odds["team"] == "home"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_home_odds", "bookmaker": "home_book"})
        )

        best_away = (
            ml_odds[ml_odds["team"] == "away"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_away_odds", "bookmaker": "away_book"})
        )

        return best_home.merge(best_away, on="game_id")

    def print_predictions(self, predictions: pd.DataFrame):
        """Print formatted predictions."""
        print("\n" + "=" * 70)
        print("NBA GAME PREDICTIONS")
        print("=" * 70)

        for _, row in predictions.iterrows():
            print(f"\n{row['away_team']} @ {row['home_team']}")
            print(f"  Game time: {row['commence_time']}")

            # Show base vs adjusted probability if injury adjustment was applied
            if "base_home_prob" in row and "injury_adjustment" in row:
                adj = row['injury_adjustment']
                if abs(adj) > 0.001:
                    print(f"  Base:   Home {row['base_home_prob']:.1%} | Away {1-row['base_home_prob']:.1%}")
                    print(f"  Injury Adj: {adj:+.1%}")

            print(f"  Model:  Home {row['model_home_prob']:.1%} | Away {row['model_away_prob']:.1%}")
            print(f"  Market: Home {row['market_home_prob']:.1%} | Away {row['market_away_prob']:.1%}")
            print(f"  Edge:   Home {row['home_edge']:+.1%} | Away {row['away_edge']:+.1%}")

            # Show injury information if available
            if "home_injury_impact" in row and row.get("home_injury_impact", 0) > 0:
                print(f"  Injuries: {row['home_team']} -{row['home_injury_impact']:.1f}pts", end="")
                if row.get("home_key_out"):
                    print(f" ({row['home_key_out']})", end="")
                print()
            if "away_injury_impact" in row and row.get("away_injury_impact", 0) > 0:
                print(f"  Injuries: {row['away_team']} -{row['away_injury_impact']:.1f}pts", end="")
                if row.get("away_key_out"):
                    print(f" ({row['away_key_out']})", end="")
                print()

            if row["bet_home"]:
                print(f"  >>> BET HOME at {row['best_home_odds']:+.0f} ({row['home_book']})")
                print(f"      Kelly: {row['home_kelly']:.1%} of bankroll")
            elif row["bet_away"]:
                print(f"  >>> BET AWAY at {row['best_away_odds']:+.0f} ({row['away_book']})")
                print(f"      Kelly: {row['away_kelly']:.1%} of bankroll")
            else:
                print(f"  No edge (min required: {self.min_edge:.0%})")

        print("\n" + "=" * 70)

        # Summary
        home_bets = predictions["bet_home"].sum()
        away_bets = predictions["bet_away"].sum()
        print(f"Total recommendations: {home_bets + away_bets} bets")
        print(f"  Home bets: {home_bets}")
        print(f"  Away bets: {away_bets}")


def run_daily_predictions():
    """Run predictions for today's games."""
    predictor = NBAPredictor(min_edge=0.03, kelly_fraction=0.2)

    # Get recent games for features
    recent_games = predictor.get_recent_games(days=60)

    # Get current odds
    odds = predictor.get_current_odds()

    if odds.empty:
        print("No upcoming games with odds available")
        return None

    # Build features
    features = predictor.build_features_for_upcoming(recent_games, odds)

    if features.empty:
        print("Could not build features for upcoming games")
        return None

    # Generate predictions
    predictions = predictor.generate_predictions(features, odds)

    # Print results
    predictor.print_predictions(predictions)

    # Save predictions
    predictions.to_csv(f"data/predictions_{datetime.now().strftime('%Y%m%d')}.csv", index=False)

    return predictions


class ATSPredictor:
    """Generate ATS predictions using dual model disagreement signal."""

    def __init__(
        self,
        model_path: str = "models/dual_model.pkl",
        min_disagreement: float = 3.0,
        use_injuries: bool = True,
    ):
        self.min_disagreement = min_disagreement
        self.use_injuries = use_injuries

        # Load dual model
        self.model = DualPredictionModel.load(model_path)
        logger.info(f"Loaded dual model with {len(self.model.feature_columns)} features")

        # Initialize clients
        self.stats_client = NBAStatsClient()
        self.odds_client = OddsAPIClient()
        self.feature_builder = GameFeatureBuilder()

        # Elo system for live predictions
        self.elo_system = EloRatingSystem()
        self._load_current_elo()

        # Injury tracking
        self.injury_builder = InjuryFeatureBuilder() if use_injuries else None
        self.injury_adjuster = InjuryAdjuster() if use_injuries else None

    def _load_current_elo(self):
        """Load current Elo ratings from saved file or calculate from history."""
        try:
            elo_df = pd.read_parquet("data/features/current_elo.parquet")
            for _, row in elo_df.iterrows():
                self.elo_system.ratings[row["team"]] = row["elo"]
            logger.info(f"Loaded Elo ratings for {len(self.elo_system.ratings)} teams")
        except FileNotFoundError:
            # Calculate from historical games
            games = pd.read_parquet("data/raw/games.parquet")
            self.elo_system.calculate_historical_elo(games)
            logger.info("Calculated Elo from historical games")

    def get_recent_games(self, days: int = 60) -> pd.DataFrame:
        """Fetch recent games for feature building."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        games = self.stats_client.get_games(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Fetched {len(games)} recent games")
        return games

    def get_current_odds(self) -> pd.DataFrame:
        """Fetch current odds for upcoming games."""
        odds = self.odds_client.get_current_odds()
        if not odds.empty:
            odds = self.odds_client.calculate_no_vig_odds(odds)
        logger.info(f"Fetched odds for {odds['game_id'].nunique()} upcoming games")
        return odds

    def _normalize_team_name(self, name: str) -> str:
        """Convert full team name to abbreviation."""
        name_map = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
        }
        return name_map.get(name, name)

    def build_features_for_upcoming(
        self,
        recent_games: pd.DataFrame,
        upcoming_odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for upcoming games."""
        # Build team features from recent games
        team_features = self.feature_builder.team_builder.build_all_features(recent_games)

        # Get most recent features for each team
        latest_features = (
            team_features
            .sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )

        # Create feature rows for upcoming games
        upcoming_games = upcoming_odds[["game_id", "home_team", "away_team", "commence_time"]].drop_duplicates()

        records = []
        for _, game in upcoming_games.iterrows():
            home_team = self._normalize_team_name(game["home_team"])
            away_team = self._normalize_team_name(game["away_team"])

            home_stats = latest_features[latest_features["team"] == home_team]
            away_stats = latest_features[latest_features["team"] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {away_team} @ {home_team}")
                continue

            # Build feature row
            record = {
                "game_id": game["game_id"],
                "commence_time": game["commence_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_team_abbr": home_team,
                "away_team_abbr": away_team,
            }

            # Add home team features
            for col in home_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"home_{col}"] = home_stats[col].values[0]

            # Add away team features
            for col in away_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"away_{col}"] = away_stats[col].values[0]

            records.append(record)

        df = pd.DataFrame(records)

        # Add differential features
        if not df.empty:
            df = self._add_differentials(df)
            df = self._add_elo_features(df)

        return df

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Elo features to upcoming games."""
        home_elos = []
        away_elos = []

        for _, row in df.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)

            home_elos.append(home_elo)
            away_elos.append(away_elo)

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_prob"] = df.apply(
            lambda r: self.elo_system.expected_home_win_prob(
                self._normalize_team_name(r["home_team"]),
                self._normalize_team_name(r["away_team"])
            ),
            axis=1
        )

        return df

    def _add_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add differential features."""
        for window in [5, 10, 20]:
            suffix = f"_{window}g"
            for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate"]:
                home_col = f"home_{stat}{suffix}"
                away_col = f"away_{stat}{suffix}"
                if home_col in df.columns and away_col in df.columns:
                    df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

        return df

    def _get_market_spreads(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get market spreads from odds data."""
        spread_odds = odds[odds["market"] == "spread"].copy()

        if spread_odds.empty:
            return pd.DataFrame()

        market_spreads = (
            spread_odds[spread_odds["team"] == "home"]
            .groupby("game_id")
            .agg({"line": "mean"})
            .reset_index()
            .rename(columns={"line": "market_spread"})
        )

        return market_spreads

    def generate_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate ATS predictions using dual model."""
        if features.empty:
            return pd.DataFrame()

        # Ensure all required features exist
        missing_cols = [c for c in self.model.feature_columns if c not in features.columns]
        for col in missing_cols:
            features[col] = 0

        # Get market spreads first (needed for prediction)
        market_spreads = self._get_market_spreads(odds)

        # Get dual model predictions with Vegas spreads
        X = features[self.model.feature_columns].fillna(0)

        # Merge spreads for prediction
        vegas_spread = None
        if not market_spreads.empty:
            features_with_spread = features.merge(market_spreads, on="game_id", how="left")
            vegas_spread = features_with_spread["market_spread"].values

        preds = self.model.get_predictions(X, vegas_spread=vegas_spread)

        # Build predictions dataframe
        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions["mlp_prob"] = preds["mlp_prob"].values
        predictions["xgb_prob"] = preds["xgb_prob"].values
        predictions["mlp_spread"] = preds["mlp_spread"].values
        predictions["xgb_spread"] = preds["xgb_spread"].values
        predictions["disagreement"] = preds["disagreement"].values
        predictions["bet_side"] = preds["bet_side"].values
        predictions["confidence"] = preds["confidence"].values

        # Add market spread info
        if not market_spreads.empty:
            predictions = predictions.merge(market_spreads, on="game_id", how="left")
            predictions["edge_vs_market"] = predictions["mlp_spread"] - predictions["market_spread"]
        else:
            predictions["market_spread"] = np.nan
            predictions["edge_vs_market"] = np.nan

        # Add injury info if available
        if self.use_injuries and self.injury_builder:
            predictions = self._add_injury_data(predictions)

        return predictions

    def _add_injury_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add injury information to predictions."""
        injury_data = []

        for _, row in predictions.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            injury_features = self.injury_builder.get_game_injury_features(home_team, away_team)

            injury_data.append({
                "game_id": row["game_id"],
                "home_injury_impact": injury_features["home_injury_impact"],
                "away_injury_impact": injury_features["away_injury_impact"],
                "home_key_out": ", ".join(injury_features["home_key_players_out"][:2]) if injury_features["home_key_players_out"] else "",
                "away_key_out": ", ".join(injury_features["away_key_players_out"][:2]) if injury_features["away_key_players_out"] else "",
            })

        injury_df = pd.DataFrame(injury_data)
        return predictions.merge(injury_df, on="game_id", how="left")

    def print_predictions(self, predictions: pd.DataFrame):
        """Print formatted ATS predictions."""
        print("\n" + "=" * 70)
        print("NBA ATS PREDICTIONS (DUAL MODEL)")
        print("=" * 70)
        print("Validated thresholds: disagreement >= 5 AND edge vs Vegas >= 5")
        print("Backtest: 58.2% ATS, +11.1% ROI on 67 bets (2024 season)")

        # Filter to actionable bets
        action_mask = predictions["bet_side"] != "PASS"
        actionable = predictions[action_mask].copy()

        if actionable.empty:
            print("\nNo high-confidence bets found today.")
            print("Games analyzed:")
            for _, row in predictions.iterrows():
                print(f"  {row['away_team']} @ {row['home_team']}: disagreement {row['disagreement']:.1f} pts")
            return

        print(f"\nFound {len(actionable)} actionable bets:\n")

        for _, row in actionable.sort_values("confidence", ascending=False).iterrows():
            print(f"{row['away_team']} @ {row['home_team']}")
            print(f"  Game time: {row['commence_time']}")
            print(f"  MLP spread:    {row['mlp_spread']:+.1f}")
            print(f"  XGBoost spread: {row['xgb_spread']:+.1f}")
            print(f"  Disagreement:  {row['disagreement']:+.1f} pts")

            if pd.notna(row.get("market_spread")):
                print(f"  Market spread: {row['market_spread']:+.1f}")
                print(f"  Edge vs market: {row['edge_vs_market']:+.1f} pts")

            # Show injury info
            if row.get("home_injury_impact", 0) > 0:
                print(f"  Injuries: {row['home_team']} -{row['home_injury_impact']:.1f}pts", end="")
                if row.get("home_key_out"):
                    print(f" ({row['home_key_out']})", end="")
                print()
            if row.get("away_injury_impact", 0) > 0:
                print(f"  Injuries: {row['away_team']} -{row['away_injury_impact']:.1f}pts", end="")
                if row.get("away_key_out"):
                    print(f" ({row['away_key_out']})", end="")
                print()

            print(f"  >>> BET {row['bet_side']} (confidence: {row['confidence']:.0%})")
            print()

        print("=" * 70)
        home_bets = (actionable["bet_side"] == "HOME").sum()
        away_bets = (actionable["bet_side"] == "AWAY").sum()
        print(f"Summary: {home_bets} HOME bets, {away_bets} AWAY bets")


class EdgeATSPredictor:
    """
    Generate ATS predictions using validated edge strategy.

    Based on backtesting (2022-2025):
    - Edge 5+ & No B2B: 54.8% win rate, +4.6% ROI, p=0.0006
    - Current season (2024-25): 57.2% win rate, +9.2% ROI
    """

    def __init__(
        self,
        model_path: str = "models/spread_model.pkl",
        strategy: str = "primary",
        use_injuries: bool = True,
    ):
        self.use_injuries = use_injuries

        # Load spread prediction model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.feature_cols = model_data["feature_cols"]
        logger.info(f"Loaded spread model with {len(self.feature_cols)} features")

        # Initialize strategy
        if strategy == "primary":
            self.strategy = EdgeStrategy.primary_strategy()
        elif strategy == "enhanced":
            self.strategy = EdgeStrategy.enhanced_strategy()
        elif strategy == "aggressive":
            self.strategy = EdgeStrategy.aggressive_strategy()
        elif strategy == "team_filtered":
            self.strategy = EdgeStrategy.team_filtered_strategy()
        elif strategy == "optimal":
            self.strategy = EdgeStrategy.optimal_strategy()
        else:
            # Default to team_filtered (recommended)
            self.strategy = EdgeStrategy.team_filtered_strategy()

        # Initialize clients
        self.stats_client = NBAStatsClient()
        self.odds_client = OddsAPIClient()
        self.feature_builder = GameFeatureBuilder()

        # Injury tracking
        self.injury_builder = InjuryFeatureBuilder() if use_injuries else None

    def _normalize_team_name(self, name: str) -> str:
        """Convert full team name to abbreviation."""
        name_map = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
        }
        return name_map.get(name, name)

    def get_recent_games(self, days: int = 60) -> pd.DataFrame:
        """Fetch recent games for feature building."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        games = self.stats_client.get_games(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Fetched {len(games)} recent games")
        return games

    def get_current_odds(self) -> pd.DataFrame:
        """Fetch current odds for upcoming games."""
        odds = self.odds_client.get_current_odds()
        if not odds.empty:
            odds = self.odds_client.calculate_no_vig_odds(odds)
        logger.info(f"Fetched odds for {odds['game_id'].nunique()} upcoming games")
        return odds

    def build_features_for_upcoming(
        self,
        recent_games: pd.DataFrame,
        upcoming_odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for upcoming games."""
        # Build team features from recent games
        team_features = self.feature_builder.team_builder.build_all_features(recent_games)

        # Get most recent features for each team
        latest_features = (
            team_features
            .sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )

        # Create feature rows for upcoming games
        upcoming_games = upcoming_odds[["game_id", "home_team", "away_team", "commence_time"]].drop_duplicates()

        records = []
        for _, game in upcoming_games.iterrows():
            home_team = self._normalize_team_name(game["home_team"])
            away_team = self._normalize_team_name(game["away_team"])

            home_stats = latest_features[latest_features["team"] == home_team]
            away_stats = latest_features[latest_features["team"] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {away_team} @ {home_team}")
                continue

            # Build feature row
            record = {
                "game_id": game["game_id"],
                "commence_time": game["commence_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_team_abbr": home_team,
                "away_team_abbr": away_team,
            }

            # Add home team features
            for col in home_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"home_{col}"] = home_stats[col].values[0]

            # Add away team features
            for col in away_stats.columns:
                if col not in ["team", "game_id", "date", "season", "opponent"]:
                    record[f"away_{col}"] = away_stats[col].values[0]

            records.append(record)

        df = pd.DataFrame(records)

        # Add differential features
        if not df.empty:
            df = self._add_differentials(df)
            # Add B2B detection (would need last game dates)
            df['home_b2b'] = False  # Placeholder - would need actual B2B detection
            df['away_b2b'] = False
            df['rest_advantage'] = df.get('rest_diff', 0)

        return df

    def _add_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add differential features."""
        for window in [5, 10, 20]:
            suffix = f"_{window}g"
            for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate"]:
                home_col = f"home_{stat}{suffix}"
                away_col = f"away_{stat}{suffix}"
                if home_col in df.columns and away_col in df.columns:
                    df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

        return df

    def _get_market_spreads(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get market spreads from odds data."""
        spread_odds = odds[odds["market"] == "spread"].copy()

        if spread_odds.empty:
            return pd.DataFrame()

        market_spreads = (
            spread_odds[spread_odds["team"] == "home"]
            .groupby("game_id")
            .agg({"line": "mean"})
            .reset_index()
            .rename(columns={"line": "market_spread"})
        )

        return market_spreads

    def generate_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate predictions using edge strategy."""
        if features.empty:
            return pd.DataFrame()

        # Ensure all required features exist
        missing_cols = [c for c in self.feature_cols if c not in features.columns]
        for col in missing_cols:
            features[col] = 0

        # Get model predictions (point spread)
        X = features[self.feature_cols].fillna(0)
        features['pred_diff'] = self.model.predict(X)

        # Get market spreads
        market_spreads = self._get_market_spreads(odds)
        if not market_spreads.empty:
            features = features.merge(market_spreads, on="game_id", how="left")
        else:
            features['market_spread'] = 0

        # Rename for strategy compatibility
        features['home_spread'] = features['market_spread']

        # Apply edge strategy
        bets = self.strategy.get_actionable_bets(features)

        # Build predictions dataframe
        predictions = features[["game_id", "commence_time", "home_team", "away_team",
                               "pred_diff", "market_spread"]].copy()
        predictions['model_edge'] = predictions['pred_diff'] + predictions['market_spread']

        # Merge bet recommendations
        if not bets.empty:
            predictions = predictions.merge(
                bets[['game_id', 'bet_side', 'confidence', 'filters_passed']],
                on='game_id',
                how='left'
            )
        else:
            predictions['bet_side'] = 'PASS'
            predictions['confidence'] = 'LOW'
            predictions['filters_passed'] = ''

        predictions['bet_side'] = predictions['bet_side'].fillna('PASS')

        # Add injury info if available
        if self.use_injuries and self.injury_builder:
            predictions = self._add_injury_data(predictions)

        return predictions

    def _add_injury_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add injury information to predictions."""
        injury_data = []

        for _, row in predictions.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            injury_features = self.injury_builder.get_game_injury_features(home_team, away_team)

            injury_data.append({
                "game_id": row["game_id"],
                "home_injury_impact": injury_features["home_injury_impact"],
                "away_injury_impact": injury_features["away_injury_impact"],
                "home_key_out": ", ".join(injury_features["home_key_players_out"][:2]) if injury_features["home_key_players_out"] else "",
                "away_key_out": ", ".join(injury_features["away_key_players_out"][:2]) if injury_features["away_key_players_out"] else "",
            })

        injury_df = pd.DataFrame(injury_data)
        return predictions.merge(injury_df, on="game_id", how="left")

    def print_predictions(self, predictions: pd.DataFrame):
        """Print formatted edge predictions."""
        print("\n" + "=" * 70)
        print("NBA ATS PREDICTIONS (EDGE STRATEGY)")
        print("=" * 70)
        print("Strategy: Edge 5+ & No B2B")
        print("Backtest: 54.8% ATS, +4.6% ROI, p=0.0006 (2022-2025)")
        print("Current Season: 57.2% ATS, +9.2% ROI")

        # Filter to actionable bets
        actionable = predictions[predictions["bet_side"] != "PASS"].copy()

        if actionable.empty:
            print("\nNo high-confidence bets found today.")
            print("\nAll games analyzed:")
            for _, row in predictions.iterrows():
                print(f"  {row['away_team']} @ {row['home_team']}: "
                      f"edge {row['model_edge']:+.1f} pts")
            return

        print(f"\nFound {len(actionable)} actionable bet(s):\n")

        for _, row in actionable.iterrows():
            print(f"{row['away_team']} @ {row['home_team']}")
            print(f"  Game time: {row['commence_time']}")
            print(f"  Model prediction: Home {row['pred_diff']:+.1f}")
            print(f"  Market spread:    Home {row['market_spread']:+.1f}")
            print(f"  Model edge:       {row['model_edge']:+.1f} pts")

            # Show injury info
            if row.get("home_injury_impact", 0) > 0:
                print(f"  Injuries: {row['home_team']} -{row['home_injury_impact']:.1f}pts", end="")
                if row.get("home_key_out"):
                    print(f" ({row['home_key_out']})", end="")
                print()
            if row.get("away_injury_impact", 0) > 0:
                print(f"  Injuries: {row['away_team']} -{row['away_injury_impact']:.1f}pts", end="")
                if row.get("away_key_out"):
                    print(f" ({row['away_key_out']})", end="")
                print()

            print(f"  >>> BET {row['bet_side']} ATS ({row['confidence']} confidence)")
            print(f"  Filters: {row['filters_passed']}")
            print()

        print("=" * 70)
        home_bets = (actionable["bet_side"] == "HOME").sum()
        away_bets = (actionable["bet_side"] == "AWAY").sum()
        print(f"Summary: {home_bets} HOME bet(s), {away_bets} AWAY bet(s)")
        print("\nBankroll recommendation: Flat bet 1-2% per game")


def run_edge_predictions():
    """Run edge-based ATS predictions for today's games."""
    try:
        predictor = EdgeATSPredictor(strategy="primary")
    except FileNotFoundError:
        print("Model file not found. Run training first.")
        return None

    # Get recent games for features
    recent_games = predictor.get_recent_games(days=60)

    # Get current odds
    odds = predictor.get_current_odds()

    if odds.empty:
        print("No upcoming games with odds available")
        return None

    # Build features
    features = predictor.build_features_for_upcoming(recent_games, odds)

    if features.empty:
        print("Could not build features for upcoming games")
        return None

    # Generate predictions
    predictions = predictor.generate_predictions(features, odds)

    # Print results
    predictor.print_predictions(predictions)

    # Save predictions
    predictions.to_csv(f"data/edge_predictions_{datetime.now().strftime('%Y%m%d')}.csv", index=False)

    return predictions


def run_ats_predictions():
    """Run ATS predictions for today's games."""
    predictor = ATSPredictor(min_disagreement=3.0)

    # Get recent games for features
    recent_games = predictor.get_recent_games(days=60)

    # Get current odds
    odds = predictor.get_current_odds()

    if odds.empty:
        print("No upcoming games with odds available")
        return None

    # Build features
    features = predictor.build_features_for_upcoming(recent_games, odds)

    if features.empty:
        print("Could not build features for upcoming games")
        return None

    # Generate predictions
    predictions = predictor.generate_predictions(features, odds)

    # Print results
    predictor.print_predictions(predictions)

    # Save predictions
    predictions.to_csv(f"data/ats_predictions_{datetime.now().strftime('%Y%m%d')}.csv", index=False)

    return predictions


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")

    # Run edge strategy predictions (primary recommended strategy)
    print("\n" + "="*70)
    print("ATS PREDICTIONS (EDGE STRATEGY) - RECOMMENDED")
    print("="*70)
    run_edge_predictions()

    # Run dual model predictions
    print("\n" + "="*70)
    print("ATS PREDICTIONS (DUAL MODEL)")
    print("="*70)
    run_ats_predictions()

    # Run moneyline predictions
    print("\n" + "="*70)
    print("MONEYLINE PREDICTIONS")
    print("="*70)
    run_daily_predictions()
