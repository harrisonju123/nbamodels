"""
NBA Multi-Bet Predictions

Generate predictions for moneyline, spread, and totals using trained models.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
from loguru import logger

from src.data import NBAStatsClient, OddsAPIClient, InjuryFeatureBuilder
from src.features import GameFeatureBuilder
from src.betting.kelly import KellyBetSizer
from src.betting.edge_strategy import EdgeStrategy, TEAMS_TO_EXCLUDE
from src.models.injury_adjustment import InjuryAdjuster
from src.models.point_spread import PointSpreadModel
from src.models.totals import TotalsModel
from src.models.dual_model import DualPredictionModel

# Optional player impact features
try:
    from src.features.player_impact import PlayerImpactModel
    HAS_PLAYER_IMPACT = True
except ImportError:
    PlayerImpactModel = None
    HAS_PLAYER_IMPACT = False

# Optional stacked ensemble model
try:
    from src.models.stacking import StackedEnsembleModel
    HAS_STACKING = True
except ImportError:
    StackedEnsembleModel = None
    HAS_STACKING = False

# Optional matchup analysis
try:
    from src.features.matchup_features import MatchupFeatureBuilder, MatchupAdjuster
    HAS_MATCHUP = True
except ImportError:
    MatchupFeatureBuilder = None
    MatchupAdjuster = None
    HAS_MATCHUP = False


class MultiPredictor:
    """Generate predictions for all bet types: moneyline, spread, and totals."""

    def __init__(
        self,
        moneyline_model_path: str = "models/spread_model_tuned.pkl",
        spread_model_path: str = "models/point_spread_model_tuned.pkl",
        totals_model_path: str = "models/totals_model_tuned.pkl",
        dual_model_path: str = "models/dual_model.pkl",
        stacking_model_path: str = "models/stacked_ensemble.pkl",
        min_edge: float = 0.03,
        kelly_fraction: float = 0.2,
        use_injuries: bool = True,
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.use_injuries = use_injuries
        self.sizer = KellyBetSizer(fraction=kelly_fraction)

        # Load moneyline model
        self.moneyline_model = None
        self.moneyline_feature_cols = None
        if os.path.exists(moneyline_model_path):
            with open(moneyline_model_path, "rb") as f:
                model_data = pickle.load(f)
            self.moneyline_model = model_data["model"]
            self.moneyline_feature_cols = model_data.get("feature_cols") or model_data.get("feature_columns")
            logger.info(f"Loaded moneyline model with {len(self.moneyline_feature_cols)} features")

        # Load spread model
        self.spread_model = None
        if os.path.exists(spread_model_path):
            self.spread_model = PointSpreadModel.load(spread_model_path)
            logger.info("Loaded point spread model")

        # Load totals model
        self.totals_model = None
        if os.path.exists(totals_model_path):
            self.totals_model = TotalsModel.load(totals_model_path)
            logger.info("Loaded totals model")

        # Load dual model for ATS predictions
        self.dual_model = None
        if os.path.exists(dual_model_path):
            try:
                self.dual_model = DualPredictionModel.load(dual_model_path)
                logger.info("Loaded dual model for ATS predictions")
            except Exception as e:
                logger.warning(f"Could not load dual model: {e}")

        # Initialize clients
        self.stats_client = NBAStatsClient()
        self.odds_client = OddsAPIClient()
        self.feature_builder = GameFeatureBuilder()

        # Initialize injury tracking
        self.injury_builder = InjuryFeatureBuilder() if use_injuries else None
        self.injury_adjuster = InjuryAdjuster() if use_injuries else None

        # Initialize edge strategy (validated ATS strategy with team filter)
        self.edge_strategy = EdgeStrategy.team_filtered_strategy()
        logger.info("Initialized edge strategy (Edge 5+ & No B2B & Team Filter)")

        # Initialize player impact model if available
        self.player_impact_model = None
        self.team_impact_cache = {}  # Cache team -> total impact for top 8 players
        impact_model_path = "data/cache/player_impact/player_impact_model.parquet"

        if HAS_PLAYER_IMPACT and os.path.exists(impact_model_path):
            try:
                self.player_impact_model = PlayerImpactModel()
                self.player_impact_model.load(impact_model_path)

                # Build team impact cache (sum of top 8 players by minutes)
                impacts_df = self.player_impact_model.get_impacts_df()
                for team in impacts_df['team_abbrev'].unique():
                    team_players = impacts_df[impacts_df['team_abbrev'] == team]
                    top_8 = team_players.nlargest(8, 'minutes')
                    self.team_impact_cache[team] = top_8['impact'].sum()

                logger.info(f"Loaded player impact model: {len(impacts_df)} players, {len(self.team_impact_cache)} teams")
            except Exception as e:
                logger.warning(f"Could not load player impact model: {e}")

        # Initialize stacked ensemble model if available
        self.stacking_model = None
        self.stacking_feature_cols = None
        if HAS_STACKING and os.path.exists(stacking_model_path):
            try:
                with open(stacking_model_path, 'rb') as f:
                    self.stacking_model = pickle.load(f)
                if self.stacking_model.is_fitted:
                    # Stacking model uses same features as base models (moneyline features)
                    self.stacking_feature_cols = self.moneyline_feature_cols
                    logger.info(f"Loaded stacked ensemble model: {len(self.stacking_model.base_models)} base models")
                else:
                    logger.warning("Stacking model loaded but not fitted")
                    self.stacking_model = None
            except Exception as e:
                logger.warning(f"Could not load stacking model: {e}")

        # Initialize Bayesian meta-learner for true posterior uncertainty
        self.bayesian_model = None
        bayesian_model_path = "models/bayesian_meta.pkl"
        if os.path.exists(bayesian_model_path):
            try:
                from src.models.bayesian import BayesianLinearModel
                self.bayesian_model = BayesianLinearModel.load(bayesian_model_path)
                logger.info("Loaded Bayesian meta-learner for posterior uncertainty")
            except Exception as e:
                logger.warning(f"Could not load Bayesian model: {e}")

        # Initialize matchup adjuster for H2H and rivalry analysis
        self.matchup_adjuster = None
        self.matchup_builder = None
        self.historical_games = None
        if HAS_MATCHUP:
            try:
                self.matchup_builder = MatchupFeatureBuilder(
                    h2h_window_games=10,
                    h2h_window_seasons=3,
                    recency_decay=0.9,
                )
                self.matchup_adjuster = MatchupAdjuster(
                    matchup_builder=self.matchup_builder,
                    adjustment_strength=0.15,  # 15% adjustment weight
                )
                # Load historical games for H2H lookup
                games_path = "data/raw/games.parquet"
                if os.path.exists(games_path):
                    self.historical_games = pd.read_parquet(games_path)
                    logger.info(f"Loaded {len(self.historical_games)} historical games for matchup analysis")
                else:
                    logger.warning("No historical games found for matchup analysis")
            except Exception as e:
                logger.warning(f"Could not initialize matchup adjuster: {e}")

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
        """Fetch current odds for all markets."""
        odds = self.odds_client.get_current_odds(
            markets=["h2h", "spreads", "totals"]
        )
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

        if not df.empty:
            df = self._add_differentials(df)
            df = self._add_elo_features(df)
            df = self._add_schedule_features(df, recent_games)
            df = self._add_season_record(df, recent_games)
            df = self._add_lineup_impact_features(df)

        return df

    def _add_season_record(
        self,
        df: pd.DataFrame,
        recent_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add current season win percentage for playoff implications."""
        # Calculate win pct from recent games
        team_records = {}

        for _, game in recent_games.iterrows():
            ht, at = game["home_team"], game["away_team"]
            home_won = game["home_score"] > game["away_score"]

            for team in [ht, at]:
                if team not in team_records:
                    team_records[team] = [0, 0]

            if home_won:
                team_records[ht][0] += 1
                team_records[at][1] += 1
            else:
                team_records[ht][1] += 1
                team_records[at][0] += 1

        home_pcts = []
        away_pcts = []
        for _, row in df.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            h_rec = team_records.get(home_team, [0, 0])
            a_rec = team_records.get(away_team, [0, 0])

            home_pcts.append(h_rec[0] / max(sum(h_rec), 1))
            away_pcts.append(a_rec[0] / max(sum(a_rec), 1))

        df["home_season_win_pct"] = home_pcts
        df["away_season_win_pct"] = away_pcts
        df["season_win_pct_diff"] = df["home_season_win_pct"] - df["away_season_win_pct"]

        return df

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Elo ratings for upcoming games."""
        try:
            # Load current Elo ratings
            elo_df = pd.read_parquet("data/features/current_elo.parquet")
            elo_dict = dict(zip(elo_df["team"], elo_df["elo"]))

            home_elos = []
            away_elos = []
            elo_diffs = []
            elo_probs = []

            home_advantage = 100.0  # Standard NBA home court advantage in Elo points

            for _, row in df.iterrows():
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])

                home_elo = elo_dict.get(home_team, 1500.0)
                away_elo = elo_dict.get(away_team, 1500.0)

                elo_diff = home_elo - away_elo
                # Elo expected probability (with home advantage)
                elo_prob = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))

                home_elos.append(home_elo)
                away_elos.append(away_elo)
                elo_diffs.append(elo_diff)
                elo_probs.append(elo_prob)

            df["home_elo"] = home_elos
            df["away_elo"] = away_elos
            df["elo_diff"] = elo_diffs
            df["elo_prob"] = elo_probs

            logger.info(f"Added Elo features for {len(df)} games")
        except Exception as e:
            logger.warning(f"Could not add Elo features: {e}")
            # Add default values
            df["home_elo"] = 1500.0
            df["away_elo"] = 1500.0
            df["elo_diff"] = 0.0
            df["elo_prob"] = 0.5

        return df

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

    def _add_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add differential features."""
        windows = [5, 10, 20]
        for window in windows:
            suffix = f"_{window}g"
            for stat in ["pts_for", "pts_against", "net_rating", "pace", "win_rate", "win_streak"]:
                home_col = f"home_{stat}{suffix}"
                away_col = f"away_{stat}{suffix}"
                if home_col in df.columns and away_col in df.columns:
                    df[f"diff_{stat}{suffix}"] = df[home_col] - df[away_col]

        # Add rest and travel differentials
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        if "home_travel_distance" in df.columns and "away_travel_distance" in df.columns:
            df["travel_diff"] = df["home_travel_distance"] - df["away_travel_distance"]

        return df

    def _add_schedule_features(
        self,
        df: pd.DataFrame,
        recent_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add B2B and rest features for upcoming games."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        # Use EST for all date comparisons (NBA games are scheduled in local time)
        est = ZoneInfo('America/New_York')
        today = datetime.now(est).date()

        # Prepare recent games for lookup
        recent_games = recent_games.copy()
        recent_games["date"] = pd.to_datetime(recent_games["date"]).dt.date

        # Only include COMPLETED games (before today) - today's games haven't been played yet!
        completed_games = recent_games[recent_games["date"] < today]

        # Build a dict of which teams played on which dates
        games_by_date = {}
        for _, game in completed_games.iterrows():
            game_date = game["date"]
            if game_date not in games_by_date:
                games_by_date[game_date] = set()
            games_by_date[game_date].add(game["home_team"])
            games_by_date[game_date].add(game["away_team"])

        logger.info(f"Today (EST): {today}, completed games through: {max(games_by_date.keys()) if games_by_date else 'none'}")

        def get_est_date(commence_time):
            """Convert commence_time to EST date."""
            try:
                dt = pd.to_datetime(commence_time)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                return dt.tz_convert(est).date()
            except Exception:
                return today + timedelta(days=1)

        # Also add upcoming games to lookup (for multi-day B2B detection)
        for _, row in df.iterrows():
            try:
                game_date = get_est_date(row.get("commence_time", ""))
                if game_date not in games_by_date:
                    games_by_date[game_date] = set()
                games_by_date[game_date].add(self._normalize_team_name(row["home_team"]))
                games_by_date[game_date].add(self._normalize_team_name(row["away_team"]))
            except Exception:
                pass

        logger.info(f"Games by date (EST): {[(d, len(t)) for d, t in sorted(games_by_date.items())[-5:]]}")

        # Calculate B2B for each upcoming game based on that game's EST date
        home_b2b = []
        away_b2b = []

        for _, row in df.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            # Get this game's date in EST
            game_date = get_est_date(row.get("commence_time", ""))

            # B2B = team played the day before this game
            day_before = game_date - timedelta(days=1)
            teams_played_day_before = games_by_date.get(day_before, set())

            home_b2b.append(1 if home_team in teams_played_day_before else 0)
            away_b2b.append(1 if away_team in teams_played_day_before else 0)

        df["home_b2b"] = home_b2b
        df["away_b2b"] = away_b2b
        df["b2b_advantage"] = df["away_b2b"] - df["home_b2b"]  # Positive = away more tired

        # Also add rest days if we have the data
        # Calculate days since last game for each team
        last_game_dates = {}
        for team in recent_games["home_team"].unique():
            team_games = recent_games[
                (recent_games["home_team"] == team) | (recent_games["away_team"] == team)
            ]
            if not team_games.empty:
                last_game_dates[team] = team_games["date"].max()

        home_rest = []
        away_rest = []

        for _, row in df.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            home_last = last_game_dates.get(home_team)
            away_last = last_game_dates.get(away_team)

            home_rest.append((today - home_last).days if home_last else 2)
            away_rest.append((today - away_last).days if away_last else 2)

        df["home_rest"] = home_rest
        df["away_rest"] = away_rest
        df["rest_advantage"] = df["home_rest"] - df["away_rest"]

        logger.info(f"Added schedule features: home_b2b={sum(home_b2b)}, away_b2b={sum(away_b2b)}")
        return df

    def _add_lineup_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add injury-adjusted player impact features for lineup strength.

        Uses LineupFeatureBuilder with real-time injury data to compute
        RAPM-based lineup strength adjusted for injured players.
        """
        # Try to use LineupFeatureBuilder with injury adjustment
        try:
            from src.features.lineup_features import LineupFeatureBuilder
            impact_model_path = "data/cache/player_impact/player_impact_model.parquet"

            if os.path.exists(impact_model_path):
                lineup_builder = LineupFeatureBuilder(impact_model_path=impact_model_path)

                home_impacts = []
                away_impacts = []
                home_missing = []
                away_missing = []

                for _, row in df.iterrows():
                    home_team = self._normalize_team_name(row["home_team"])
                    away_team = self._normalize_team_name(row["away_team"])

                    # Get injured player IDs if injury builder is available
                    home_injured_ids = []
                    away_injured_ids = []
                    if self.use_injuries and self.injury_builder:
                        try:
                            home_injured_ids = self.injury_builder.get_injured_player_ids(home_team)
                            away_injured_ids = self.injury_builder.get_injured_player_ids(away_team)
                        except Exception as e:
                            logger.warning(f"Could not get injured player IDs for {home_team} vs {away_team}: {e}")
                            home_injured_ids = []
                            away_injured_ids = []

                    # Build lineup features with injury adjustment
                    features = lineup_builder.build_game_features(
                        home_team, away_team,
                        home_injured_ids, away_injured_ids
                    )

                    home_impacts.append(features.get('home_lineup_impact', 0.0))
                    away_impacts.append(features.get('away_lineup_impact', 0.0))
                    home_missing.append(features.get('home_missing_impact', 0.0))
                    away_missing.append(features.get('away_missing_impact', 0.0))

                df["home_lineup_impact"] = home_impacts
                df["away_lineup_impact"] = away_impacts
                df["lineup_impact_diff"] = df["home_lineup_impact"] - df["away_lineup_impact"]
                df["home_missing_impact"] = home_missing
                df["away_missing_impact"] = away_missing

                logger.info(f"Added injury-adjusted RAPM features: {len([i for i in home_impacts if i != 0])} teams with data")
                return df

        except Exception as e:
            logger.warning(f"Could not use LineupFeatureBuilder: {e}")

        # Fallback to static team totals if lineup builder not available
        if not self.team_impact_cache:
            df["home_lineup_impact"] = 0.0
            df["away_lineup_impact"] = 0.0
            df["lineup_impact_diff"] = 0.0
            df["home_missing_impact"] = 0.0
            df["away_missing_impact"] = 0.0
            return df

        home_impacts = []
        away_impacts = []

        for _, row in df.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            home_impact = self.team_impact_cache.get(home_team, 0.0)
            away_impact = self.team_impact_cache.get(away_team, 0.0)

            home_impacts.append(home_impact)
            away_impacts.append(away_impact)

        df["home_lineup_impact"] = home_impacts
        df["away_lineup_impact"] = away_impacts
        df["lineup_impact_diff"] = df["home_lineup_impact"] - df["away_lineup_impact"]
        df["home_missing_impact"] = 0.0
        df["away_missing_impact"] = 0.0

        logger.info(f"Added lineup impact features (fallback): {len([i for i in home_impacts if i != 0])} teams with data")
        return df

    def generate_all_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions for all bet types.

        Returns:
            Dictionary with keys: 'moneyline', 'spread', 'totals', 'ats'
        """
        results = {}

        # Moneyline predictions
        if self.moneyline_model is not None:
            results["moneyline"] = self._generate_moneyline_predictions(features, odds)

        # Spread predictions
        if self.spread_model is not None:
            results["spread"] = self._generate_spread_predictions(features, odds)

        # Totals predictions
        if self.totals_model is not None:
            results["totals"] = self._generate_totals_predictions(features, odds)

        # ATS predictions using dual model
        if self.dual_model is not None:
            results["ats"] = self._generate_ats_predictions(features, odds)

        # Edge strategy predictions (validated ATS strategy)
        if self.spread_model is not None:
            results["edge"] = self._generate_edge_predictions(features, odds)

        # Consensus predictions (both EDGE and ATS agree)
        if "edge" in results and "ats" in results:
            results["consensus"] = self._generate_consensus_predictions(
                results["edge"], results["ats"], features, odds
            )

        # Stacking ensemble predictions (combines XGB + LightGBM with meta-learner)
        if self.stacking_model is not None:
            results["stacking"] = self._generate_stacking_predictions(features, odds)

        # Matchup predictions (H2H, rivalries, style matchups)
        if self.matchup_adjuster is not None and self.historical_games is not None:
            results["matchup"] = self._generate_matchup_predictions(features, odds)

        return results

    def _generate_moneyline_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate moneyline predictions."""
        if features.empty:
            return pd.DataFrame()

        # Ensure all required features exist
        missing_cols = [c for c in self.moneyline_feature_cols if c not in features.columns]
        for col in missing_cols:
            features[col] = 0

        # Get predictions
        X = features[self.moneyline_feature_cols].fillna(0)
        probs = self.moneyline_model.predict_proba(X)[:, 1]

        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions["model_home_prob"] = probs
        predictions["model_away_prob"] = 1 - probs

        # Add injury adjustments
        if self.use_injuries and self.injury_builder:
            predictions = self._add_injury_data(predictions)
            if self.injury_adjuster:
                predictions = self.injury_adjuster.adjust_predictions(predictions)

        # Merge with market odds
        ml_odds = odds[odds["market"] == "moneyline"]
        market_probs = self._get_market_consensus(ml_odds)
        predictions = predictions.merge(market_probs, on="game_id", how="left")

        # Calculate edge
        predictions["home_edge"] = predictions["model_home_prob"] - predictions["market_home_prob"]
        predictions["away_edge"] = predictions["model_away_prob"] - predictions["market_away_prob"]

        # Get best odds
        best_odds = self._get_best_moneyline_odds(ml_odds)
        predictions = predictions.merge(best_odds, on="game_id", how="left")

        # Kelly sizing
        predictions["home_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_home_prob"], r["best_home_odds"])
            if pd.notna(r.get("best_home_odds")) else 0, axis=1
        )
        predictions["away_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_away_prob"], r["best_away_odds"])
            if pd.notna(r.get("best_away_odds")) else 0, axis=1
        )

        # Bet recommendations
        predictions["bet_home"] = predictions["home_edge"] >= self.min_edge
        predictions["bet_away"] = predictions["away_edge"] >= self.min_edge
        predictions["bet_type"] = "moneyline"

        return predictions

    def _generate_spread_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate spread predictions with injury adjustments."""
        if features.empty:
            return pd.DataFrame()

        # Get spread odds
        spread_odds = odds[odds["market"] == "spread"].copy()
        if spread_odds.empty:
            return pd.DataFrame()

        # Get consensus spread line for each game
        home_spreads = (
            spread_odds[spread_odds["team"] == "home"]
            .groupby("game_id")
            .agg({"line": "median", "no_vig_prob": "mean"})
            .reset_index()
            .rename(columns={"line": "spread_line", "no_vig_prob": "market_cover_prob"})
        )

        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions = predictions.merge(home_spreads, on="game_id", how="left")

        if predictions["spread_line"].isna().all():
            return pd.DataFrame()

        # Predict expected point differential
        spread_features = features.copy()
        missing_cols = [c for c in self.spread_model.feature_columns if c not in spread_features.columns]
        for col in missing_cols:
            spread_features[col] = 0

        predictions["base_expected_diff"] = self.spread_model.predict(spread_features)

        # Apply injury adjustments to spread predictions
        if self.use_injuries and self.injury_builder and self.injury_adjuster:
            injury_adjustments = []
            for _, row in predictions.iterrows():
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])
                inj = self.injury_builder.get_game_injury_features(home_team, away_team)

                adj_diff, adjustment = self.injury_adjuster.adjust_spread_prediction(
                    expected_diff=row["base_expected_diff"],
                    home_injury_impact=inj["home_injury_impact"],
                    away_injury_impact=inj["away_injury_impact"],
                    home_star_out=bool(inj["home_star_out"]),
                    away_star_out=bool(inj["away_star_out"]),
                )
                injury_adjustments.append({
                    "game_id": row["game_id"],
                    "expected_diff": adj_diff,
                    "injury_spread_adj": adjustment,
                    "home_injury_impact": inj["home_injury_impact"],
                    "away_injury_impact": inj["away_injury_impact"],
                    "home_star_out": inj["home_star_out"],
                    "away_star_out": inj["away_star_out"],
                })
            injury_df = pd.DataFrame(injury_adjustments)
            predictions = predictions.merge(injury_df, on="game_id", how="left")
            logger.info(f"Applied injury spread adjustments (avg: {injury_df['injury_spread_adj'].abs().mean():.1f} pts)")
        else:
            predictions["expected_diff"] = predictions["base_expected_diff"]
            predictions["injury_spread_adj"] = 0.0

        # Apply RAPM-based lineup adjustment to spread predictions
        # Each point of RAPM missing impact ≈ 0.8 points on spread (calibrated constant)
        RAPM_TO_SPREAD = 0.8

        # Validate lineup features are available
        required_cols = ["game_id", "home_missing_impact", "away_missing_impact"]
        missing_cols = [c for c in required_cols if c not in features.columns]

        if missing_cols:
            logger.warning(f"Missing lineup features: {missing_cols}, skipping RAPM adjustment")
            predictions["rapm_adjustment"] = 0.0
            predictions["home_missing_impact"] = 0.0
            predictions["away_missing_impact"] = 0.0
        else:
            # Merge lineup impact features from features DataFrame
            lineup_features = features[required_cols]

            # Validate game_id uniqueness
            if lineup_features["game_id"].duplicated().any():
                logger.error("Duplicate game_ids in lineup features, skipping RAPM adjustment")
                predictions["rapm_adjustment"] = 0.0
                predictions["home_missing_impact"] = 0.0
                predictions["away_missing_impact"] = 0.0
            else:
                n_before = len(predictions)
                predictions = predictions.merge(lineup_features, on="game_id", how="left")

                # Validate merge didn't create duplicates
                if len(predictions) != n_before:
                    logger.error(
                        f"Merge created duplicates: {n_before} -> {len(predictions)} rows, "
                        "rolling back RAPM adjustment"
                    )
                    # This shouldn't happen, but if it does, we need to handle it
                    predictions = predictions.drop_duplicates(subset=["game_id"], keep="first")
                    predictions["rapm_adjustment"] = 0.0
                else:
                    # Fill NaN with 0 if lineup features not available for some games
                    predictions["home_missing_impact"] = predictions["home_missing_impact"].fillna(0)
                    predictions["away_missing_impact"] = predictions["away_missing_impact"].fillna(0)

                    # Calculate RAPM adjustment
                    # Positive adjustment hurts home team (their players are out)
                    predictions["rapm_adjustment"] = (
                        (predictions["home_missing_impact"] - predictions["away_missing_impact"]) * RAPM_TO_SPREAD
                    ).clip(-10.0, 10.0)  # Cap at ±10 points

                    # Apply RAPM adjustment to expected_diff
                    predictions["expected_diff"] = predictions["expected_diff"] - predictions["rapm_adjustment"]

                    # Log RAPM adjustments
                    non_zero_rapm = predictions[predictions["rapm_adjustment"] != 0]
                    if not non_zero_rapm.empty:
                        logger.info(
                            f"Applied RAPM spread adjustments: "
                            f"avg={predictions['rapm_adjustment'].mean():.2f}, "
                            f"games with adjustment={len(non_zero_rapm)}"
                        )

                        # Check for potential double-counting with injury adjustments
                        if self.use_injuries and self.injury_builder and self.injury_adjuster:
                            both_adjusted = (
                                (predictions["injury_spread_adj"].abs() > 0.1) &
                                (predictions["rapm_adjustment"].abs() > 0.1)
                            )
                            if both_adjusted.any():
                                logger.warning(
                                    f"{both_adjusted.sum()} games have both injury ({predictions.loc[both_adjusted, 'injury_spread_adj'].abs().mean():.1f} pts) "
                                    f"and RAPM ({predictions.loc[both_adjusted, 'rapm_adjustment'].abs().mean():.1f} pts) adjustments. "
                                    "Verify these are complementary (injury uses ESPN data, RAPM uses player impact), not duplicative."
                                )
                    else:
                        logger.info(
                            "No RAPM adjustments applied (all missing impact values are zero). "
                            "Verify lineup features are being generated correctly."
                        )

        # Calculate cover probability for each game's spread
        predictions["model_cover_prob"] = predictions.apply(
            lambda r: self.spread_model.predict_spread_prob(
                spread_features[spread_features.index == r.name],
                r["spread_line"]
            )[0] if pd.notna(r["spread_line"]) else 0.5,
            axis=1
        )

        # Calculate edge
        predictions["cover_edge"] = predictions["model_cover_prob"] - predictions["market_cover_prob"]

        # Get best spread odds
        best_spread = self._get_best_spread_odds(spread_odds)
        predictions = predictions.merge(best_spread, on="game_id", how="left")

        # Kelly sizing
        predictions["kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_cover_prob"], r["best_spread_odds"])
            if pd.notna(r.get("best_spread_odds")) else 0, axis=1
        )

        # Add B2B info from features
        predictions["home_b2b"] = features.get("home_b2b", pd.Series([0] * len(features))).values
        predictions["away_b2b"] = features.get("away_b2b", pd.Series([0] * len(features))).values

        # Calculate point-based edge (for display)
        predictions["point_edge"] = predictions["expected_diff"] + predictions["spread_line"]

        # Apply validated filters for bet recommendation
        # Strategy: Edge 5+ & No B2B & Team Filter (57.9% win rate, +10.5% ROI)
        edge_threshold = 5.0  # Points of edge required

        # Filter 1: Point edge threshold
        predictions["has_edge"] = predictions["point_edge"].abs() >= edge_threshold

        # Filter 2: Team exclusion (historically poor ATS performers)
        predictions["team_excluded"] = predictions["home_team"].isin(TEAMS_TO_EXCLUDE)

        # Filter 3: B2B check (don't bet on team that's on back-to-back)
        # If betting home to cover (point_edge > 0), check home_b2b
        # If betting away to cover (point_edge < 0), check away_b2b
        predictions["betting_on_b2b"] = (
            ((predictions["point_edge"] > 0) & (predictions["home_b2b"] == 1)) |
            ((predictions["point_edge"] < 0) & (predictions["away_b2b"] == 1))
        )

        # Build filters_passed string for UI
        def get_filters_passed(row):
            filters = []
            if abs(row["point_edge"]) >= edge_threshold:
                filters.append(f"edge {abs(row['point_edge']):.1f}")
            if row["point_edge"] > 0 and row["home_b2b"] == 0:
                filters.append("no B2B")
            elif row["point_edge"] < 0 and row["away_b2b"] == 0:
                filters.append("no B2B")
            if row["point_edge"] > 0 and row["home_team"] not in TEAMS_TO_EXCLUDE:
                filters.append("team OK")
            elif row["point_edge"] < 0 and row["away_team"] not in TEAMS_TO_EXCLUDE:
                filters.append("team OK")
            return ", ".join(filters) if filters else ""

        predictions["filters_passed"] = predictions.apply(get_filters_passed, axis=1)

        # Bet side: HOME if positive edge, AWAY if negative edge
        predictions["bet_side"] = predictions["point_edge"].apply(
            lambda x: "HOME" if x > 0 else ("AWAY" if x < 0 else "PASS")
        )

        # Final bet recommendation - all filters must pass
        predictions["bet_cover"] = (
            predictions["has_edge"] &
            (~predictions["team_excluded"] | (predictions["point_edge"] < 0)) &  # Team filter for home bets
            (~predictions["betting_on_b2b"])  # B2B filter
        )

        # Also exclude away bets on excluded teams
        predictions.loc[
            (predictions["point_edge"] < 0) &
            (predictions["away_team"].isin(TEAMS_TO_EXCLUDE)),
            "bet_cover"
        ] = False

        predictions["bet_type"] = "spread"

        return predictions

    def _generate_totals_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate totals predictions."""
        if features.empty:
            return pd.DataFrame()

        # Get totals odds
        totals_odds = odds[odds["market"] == "total"].copy()
        if totals_odds.empty:
            return pd.DataFrame()

        # Get consensus total line for each game
        over_odds = (
            totals_odds[totals_odds["team"] == "over"]
            .groupby("game_id")
            .agg({"line": "median", "no_vig_prob": "mean"})
            .reset_index()
            .rename(columns={"line": "total_line", "no_vig_prob": "market_over_prob"})
        )

        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions = predictions.merge(over_odds, on="game_id", how="left")

        if predictions["total_line"].isna().all():
            return pd.DataFrame()

        # Predict expected total
        totals_features = features.copy()
        missing_cols = [c for c in self.totals_model.feature_columns if c not in totals_features.columns]
        for col in missing_cols:
            totals_features[col] = 0

        predictions["expected_total"] = self.totals_model.predict(totals_features)

        # Calculate over probability for each game's line
        predictions["model_over_prob"] = predictions.apply(
            lambda r: self.totals_model.predict_over_prob(
                totals_features[totals_features.index == r.name],
                r["total_line"]
            )[0] if pd.notna(r["total_line"]) else 0.5,
            axis=1
        )
        predictions["model_under_prob"] = 1 - predictions["model_over_prob"]
        predictions["market_under_prob"] = 1 - predictions["market_over_prob"]

        # Calculate edge
        predictions["over_edge"] = predictions["model_over_prob"] - predictions["market_over_prob"]
        predictions["under_edge"] = predictions["model_under_prob"] - predictions["market_under_prob"]

        # Get best totals odds
        best_totals = self._get_best_totals_odds(totals_odds)
        predictions = predictions.merge(best_totals, on="game_id", how="left")

        # Kelly sizing
        predictions["over_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_over_prob"], r["best_over_odds"])
            if pd.notna(r.get("best_over_odds")) else 0, axis=1
        )
        predictions["under_kelly"] = predictions.apply(
            lambda r: self.sizer.calculate_kelly(r["model_under_prob"], r["best_under_odds"])
            if pd.notna(r.get("best_under_odds")) else 0, axis=1
        )

        # Bet recommendations
        predictions["bet_over"] = predictions["over_edge"] >= self.min_edge
        predictions["bet_under"] = predictions["under_edge"] >= self.min_edge
        predictions["bet_type"] = "totals"

        return predictions

    def _generate_ats_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate ATS predictions using dual model disagreement signal.

        VALIDATED STRATEGY (5 seasons, 2020-2024):
        - Moderate HOME: min_disagreement=4.0, min_edge=2.0, home_only=True
        - 280 bets, 63.2% ATS, +19.6% ROI, profitable 4/5 seasons
        """
        # Validated strategy parameters
        MIN_DISAGREEMENT = 4.0  # MLP/XGB must disagree by 4+ points
        MIN_EDGE_VS_MARKET = 2.0  # Must have 2+ point edge vs market
        HOME_ONLY = True  # Only bet on home teams (away signal is unprofitable)
        if features.empty:
            return pd.DataFrame()

        # Get spread odds for market lines
        spread_odds = odds[odds["market"] == "spread"].copy()
        if spread_odds.empty:
            logger.warning("No spread odds available for ATS predictions")
            return pd.DataFrame()

        # Get consensus market spread for each game
        market_spreads = (
            spread_odds[spread_odds["team"] == "home"]
            .groupby("game_id")
            .agg({"line": "median"})
            .reset_index()
            .rename(columns={"line": "market_spread"})
        )

        # Ensure all required features exist for dual model (work on copy to avoid mutation)
        features_copy = features.copy()
        missing_cols = [c for c in self.dual_model.feature_columns if c not in features_copy.columns]
        for col in missing_cols:
            features_copy[col] = 0

        # Merge market spreads with features
        features_with_spread = features_copy.merge(market_spreads, on="game_id", how="left")
        vegas_spread = features_with_spread["market_spread"].values

        # Get dual model predictions with Vegas spreads
        X = features_copy[self.dual_model.feature_columns].fillna(0)
        preds = self.dual_model.get_predictions(X, vegas_spread=vegas_spread)

        # Build predictions dataframe
        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        predictions["mlp_prob"] = preds["mlp_prob"].values
        predictions["xgb_prob"] = preds["xgb_prob"].values
        predictions["mlp_spread"] = preds["mlp_spread"].values
        predictions["xgb_spread"] = preds["xgb_spread"].values
        predictions["disagreement"] = preds["disagreement"].values
        predictions["bet_side"] = preds["bet_side"].values
        predictions["confidence"] = preds["confidence"].values

        # Add market spread
        predictions = predictions.merge(market_spreads, on="game_id", how="left")
        predictions["edge_vs_market"] = predictions["mlp_spread"] - predictions["market_spread"]

        # Add best spread odds for bet sizing
        best_spread = self._get_best_spread_odds(spread_odds)
        predictions = predictions.merge(best_spread, on="game_id", how="left")

        # Add injury info if available
        if self.use_injuries and self.injury_builder:
            injury_data = []
            for _, row in predictions.iterrows():
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])
                inj = self.injury_builder.get_game_injury_features(home_team, away_team)
                injury_data.append({
                    "game_id": row["game_id"],
                    "home_injury_impact": inj["home_injury_impact"],
                    "away_injury_impact": inj["away_injury_impact"],
                    "home_key_out": ", ".join(inj["home_key_players_out"][:2]) if inj["home_key_players_out"] else "",
                    "away_key_out": ", ".join(inj["away_key_players_out"][:2]) if inj["away_key_players_out"] else "",
                })
            predictions = predictions.merge(pd.DataFrame(injury_data), on="game_id", how="left")

        # Betting flags using VALIDATED strategy thresholds
        # Only bet when: disagreement >= threshold AND edge vs market >= threshold
        predictions["bet_home"] = (
            (predictions["disagreement"] >= MIN_DISAGREEMENT) &
            (predictions["edge_vs_market"] >= MIN_EDGE_VS_MARKET)
        )
        # Away bets are filtered out if HOME_ONLY is True (validated as unprofitable)
        if HOME_ONLY:
            predictions["bet_away"] = False
        else:
            predictions["bet_away"] = (
                (predictions["disagreement"] <= -MIN_DISAGREEMENT) &
                (predictions["edge_vs_market"] <= -MIN_EDGE_VS_MARKET)
            )
        predictions["bet_type"] = "ats"

        # Kelly sizing based on confidence and spread odds
        # ATS bets logged as "spread" type, so use spread odds for Kelly calc
        # Convert confidence (0-1) to implied probability for Kelly
        # Higher confidence = higher implied edge
        def calc_ats_kelly(row):
            if not row["bet_home"] and not row["bet_away"]:
                return 0.0
            # Use confidence as proxy for win probability (baseline 52.4% + confidence boost)
            base_prob = 0.524  # Break-even at -110
            confidence_boost = row["confidence"] * 0.10  # Up to 10% boost at max confidence
            implied_prob = min(base_prob + confidence_boost, 0.70)  # Cap at 70%
            odds = row.get("best_spread_odds", -110)
            if pd.isna(odds):
                odds = -110
            return self.sizer.calculate_kelly(implied_prob, odds)

        predictions["kelly"] = predictions.apply(calc_ats_kelly, axis=1)

        logger.info(f"Generated ATS predictions: {predictions['bet_home'].sum()} home, {predictions['bet_away'].sum()} away")
        return predictions

    def _generate_edge_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate ATS predictions using validated edge strategy.

        VALIDATED STRATEGY (2022-2025):
        - Edge 5+ & No B2B: 54.8% win rate, +4.6% ROI, p=0.0006
        - Current season (2024-25): 57.2% win rate, +9.2% ROI
        """
        if features.empty:
            return pd.DataFrame()

        # Get spread odds for market lines
        spread_odds = odds[odds["market"] == "spread"].copy()
        if spread_odds.empty:
            logger.warning("No spread odds available for edge predictions")
            return pd.DataFrame()

        # Get consensus market spread for each game
        market_spreads = (
            spread_odds[spread_odds["team"] == "home"]
            .groupby("game_id")
            .agg({"line": "median"})
            .reset_index()
            .rename(columns={"line": "market_spread"})
        )

        # Get model predictions using spread model
        X = features[self.spread_model.feature_columns].fillna(0)
        pred_diff = self.spread_model.predict(X)

        # Build features dataframe for strategy
        features_copy = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        features_copy["pred_diff"] = pred_diff

        # Add B2B flags if available
        features_copy["home_b2b"] = features.get("home_b2b", pd.Series([False] * len(features))).values
        features_copy["away_b2b"] = features.get("away_b2b", pd.Series([False] * len(features))).values
        features_copy["rest_advantage"] = features.get("rest_advantage", pd.Series([0] * len(features))).values

        # Merge with market spreads
        features_copy = features_copy.merge(market_spreads, on="game_id", how="left")
        features_copy["home_spread"] = features_copy["market_spread"]

        # Apply edge strategy to get betting signals
        actionable = self.edge_strategy.get_actionable_bets(features_copy)

        # Build predictions dataframe
        predictions = features_copy[["game_id", "commence_time", "home_team", "away_team",
                                     "pred_diff", "market_spread", "home_b2b", "away_b2b",
                                     "rest_advantage"]].copy()
        predictions["model_edge"] = predictions["pred_diff"] + predictions["market_spread"]

        # Merge bet signals
        if not actionable.empty:
            predictions = predictions.merge(
                actionable[["game_id", "bet_side", "confidence", "filters_passed"]],
                on="game_id",
                how="left"
            )
        else:
            predictions["bet_side"] = "PASS"
            predictions["confidence"] = "LOW"
            predictions["filters_passed"] = ""

        predictions["bet_side"] = predictions["bet_side"].fillna("PASS")
        predictions["confidence"] = predictions["confidence"].fillna("LOW")
        predictions["filters_passed"] = predictions["filters_passed"].fillna("")

        # Add betting flags
        predictions["bet_home"] = predictions["bet_side"] == "HOME"
        predictions["bet_away"] = predictions["bet_side"] == "AWAY"
        predictions["bet_type"] = "edge"

        # Add best spread odds for bet sizing
        best_spread = self._get_best_spread_odds(spread_odds)
        predictions = predictions.merge(best_spread, on="game_id", how="left")

        # Kelly sizing - using validated win rate
        def calc_edge_kelly(row):
            if not row["bet_home"] and not row["bet_away"]:
                return 0.0
            # Use validated win rate based on confidence level
            if row["confidence"] == "HIGH":
                win_prob = 0.57  # High confidence edge
            elif row["confidence"] == "MEDIUM":
                win_prob = 0.55
            else:
                win_prob = 0.53
            odds = row.get("best_spread_odds", -110)
            if pd.isna(odds):
                odds = -110
            return self.sizer.calculate_kelly(win_prob, odds)

        predictions["kelly"] = predictions.apply(calc_edge_kelly, axis=1)

        # Add injury info if available
        if self.use_injuries and self.injury_builder:
            injury_data = []
            for _, row in predictions.iterrows():
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])
                inj = self.injury_builder.get_game_injury_features(home_team, away_team)
                injury_data.append({
                    "game_id": row["game_id"],
                    "home_injury_impact": inj["home_injury_impact"],
                    "away_injury_impact": inj["away_injury_impact"],
                    "home_key_out": ", ".join(inj["home_key_players_out"][:2]) if inj["home_key_players_out"] else "",
                    "away_key_out": ", ".join(inj["away_key_players_out"][:2]) if inj["away_key_players_out"] else "",
                })
            predictions = predictions.merge(pd.DataFrame(injury_data), on="game_id", how="left")

        logger.info(f"Generated edge predictions: {predictions['bet_home'].sum()} home, {predictions['bet_away'].sum()} away")
        return predictions

    def _generate_consensus_predictions(
        self,
        edge_preds: pd.DataFrame,
        ats_preds: pd.DataFrame,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate CONSENSUS predictions when both SPREAD (EdgeStrategy) and ATS (Dual Model) agree.

        CONSENSUS STRATEGY:
        - Bet only when BOTH strategies recommend the same side
        - Higher confidence due to independent model agreement
        - Expected to have higher win rate but fewer bets
        """
        if edge_preds.empty or ats_preds.empty:
            logger.warning("Cannot generate consensus - missing edge or ats predictions")
            return pd.DataFrame()

        # Start with edge predictions as base
        predictions = edge_preds[["game_id", "commence_time", "home_team", "away_team",
                                   "pred_diff", "market_spread", "model_edge",
                                   "home_b2b", "away_b2b", "rest_advantage"]].copy()

        # Merge edge strategy bet signals
        predictions["edge_bet_home"] = edge_preds["bet_home"].values
        predictions["edge_bet_away"] = edge_preds["bet_away"].values
        predictions["edge_confidence"] = edge_preds["confidence"].values
        predictions["edge_filters"] = edge_preds["filters_passed"].values

        # Merge ATS (dual model) bet signals
        ats_cols = ats_preds[["game_id", "bet_home", "disagreement", "mlp_spread",
                              "xgb_spread", "edge_vs_market", "confidence"]].copy()
        ats_cols = ats_cols.rename(columns={
            "bet_home": "ats_bet_home",
            "disagreement": "ats_disagreement",
            "confidence": "ats_confidence",
        })
        predictions = predictions.merge(ats_cols, on="game_id", how="left")

        # Fill NaN values for games where ATS didn't have predictions
        predictions["ats_bet_home"] = predictions["ats_bet_home"].fillna(False)
        predictions["ats_disagreement"] = predictions["ats_disagreement"].fillna(0)
        predictions["ats_confidence"] = predictions["ats_confidence"].fillna(0)

        # CONSENSUS: Both strategies must agree on HOME
        # (ATS is home-only, so we only check home agreement)
        predictions["consensus_home"] = (
            predictions["edge_bet_home"] & predictions["ats_bet_home"]
        )

        # For away bets, edge strategy can bet away but ATS is home-only
        # So consensus away = edge_away AND ats says DON'T bet home (i.e., no signal)
        # Actually, let's be stricter: consensus only when both actively agree
        predictions["consensus_away"] = False  # ATS doesn't bet away, so no consensus possible

        # Build consensus explanation
        def build_consensus_reason(row):
            reasons = []
            if row["consensus_home"]:
                reasons.append(f"EDGE: {row['edge_filters']}")
                reasons.append(f"ATS: disagree {row['ats_disagreement']:.1f}pt")
                return " + ".join(reasons)
            return ""

        predictions["consensus_reason"] = predictions.apply(build_consensus_reason, axis=1)

        # Confidence level based on combined signals
        def calc_consensus_confidence(row):
            if not row["consensus_home"] and not row["consensus_away"]:
                return "NONE"
            # Both models agree = HIGH confidence
            edge_high = row["edge_confidence"] == "HIGH"
            ats_high = row["ats_confidence"] >= 0.6
            if edge_high and ats_high:
                return "HIGH"
            elif edge_high or ats_high:
                return "MEDIUM"
            return "LOW"

        predictions["confidence"] = predictions.apply(calc_consensus_confidence, axis=1)

        # Final bet flags
        predictions["bet_home"] = predictions["consensus_home"]
        predictions["bet_away"] = predictions["consensus_away"]
        predictions["bet_side"] = predictions.apply(
            lambda r: "HOME" if r["bet_home"] else ("AWAY" if r["bet_away"] else "PASS"),
            axis=1
        )
        predictions["bet_type"] = "consensus"

        # Add spread odds for Kelly sizing
        spread_odds = odds[odds["market"] == "spread"].copy()
        if not spread_odds.empty:
            best_spread = self._get_best_spread_odds(spread_odds)
            predictions = predictions.merge(best_spread, on="game_id", how="left")

        # Kelly sizing - higher confidence for consensus bets
        def calc_consensus_kelly(row):
            if not row["bet_home"] and not row["bet_away"]:
                return 0.0
            # Consensus bets should have higher win probability
            if row["confidence"] == "HIGH":
                win_prob = 0.62  # Both models high confidence
            elif row["confidence"] == "MEDIUM":
                win_prob = 0.58
            else:
                win_prob = 0.55
            odds_val = row.get("best_spread_odds", -110)
            if pd.isna(odds_val):
                odds_val = -110
            return self.sizer.calculate_kelly(win_prob, odds_val)

        predictions["kelly"] = predictions.apply(calc_consensus_kelly, axis=1)

        # Add injury info if available
        if self.use_injuries and self.injury_builder:
            injury_data = []
            for _, row in predictions.iterrows():
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])
                inj = self.injury_builder.get_game_injury_features(home_team, away_team)
                injury_data.append({
                    "game_id": row["game_id"],
                    "home_injury_impact": inj["home_injury_impact"],
                    "away_injury_impact": inj["away_injury_impact"],
                    "home_key_out": ", ".join(inj["home_key_players_out"][:2]) if inj["home_key_players_out"] else "",
                    "away_key_out": ", ".join(inj["away_key_players_out"][:2]) if inj["away_key_players_out"] else "",
                })
            predictions = predictions.merge(pd.DataFrame(injury_data), on="game_id", how="left")

        consensus_bets = predictions["bet_home"].sum() + predictions["bet_away"].sum()
        logger.info(f"Generated consensus predictions: {consensus_bets} bets (both EDGE + ATS agree)")
        return predictions

    def _generate_stacking_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions using stacked ensemble model.

        The stacking model combines XGBoost and LightGBM base models with a
        logistic regression meta-learner for improved probability calibration.
        Also provides uncertainty estimates from base model disagreement.
        """
        if features.empty:
            return pd.DataFrame()

        if self.stacking_feature_cols is None:
            logger.warning("No feature columns for stacking model")
            return pd.DataFrame()

        # Ensure all required features exist
        features_copy = features.copy()
        missing_cols = [c for c in self.stacking_feature_cols if c not in features_copy.columns]
        for col in missing_cols:
            features_copy[col] = 0

        # Get stacking predictions
        X = features_copy[self.stacking_feature_cols].fillna(0)
        probs = self.stacking_model.predict_proba(X)
        uncertainty = self.stacking_model.get_uncertainty(X)

        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()
        # Note: stacking model returns P(away_win), so invert for home_prob
        predictions["stack_home_prob"] = 1 - probs
        predictions["stack_away_prob"] = probs
        predictions["stack_uncertainty"] = uncertainty

        # Add 90% credible intervals
        # Try to use true Bayesian posterior if available, otherwise use normal approximation
        if self.bayesian_model is not None:
            try:
                # Validate stacking model has required method
                if not hasattr(self.stacking_model, 'get_base_predictions'):
                    raise AttributeError("Stacking model missing get_base_predictions method")

                # Use Bayesian model on base model predictions to get true posterior CI
                # Get base model predictions as features for Bayesian model
                base_preds = self.stacking_model.get_base_predictions(X)

                # Validate shape before creating DataFrame
                expected_n_models = len(self.stacking_model.base_models)
                if base_preds.shape[1] != expected_n_models:
                    raise ValueError(
                        f"Expected {expected_n_models} base predictions, got {base_preds.shape[1]}"
                    )

                base_preds_df = pd.DataFrame(
                    base_preds,
                    columns=[f"{m.name}_pred" for m in self.stacking_model.base_models]
                )

                # Get true Bayesian credible intervals
                ci_lower, ci_upper = self.bayesian_model.get_credible_intervals(base_preds_df, alpha=0.10)
                predictions["ci_lower"] = ci_lower
                predictions["ci_upper"] = ci_upper

                # Also get Bayesian uncertainty for Kelly sizing
                bayesian_uncertainty = self.bayesian_model.get_uncertainty(base_preds_df)
                predictions["bayesian_uncertainty"] = bayesian_uncertainty

                logger.info("Using true Bayesian credible intervals")
            except (AttributeError, ValueError) as e:
                logger.warning(f"Stacking model incompatible with Bayesian CI: {e}")
                # Fall back to normal approximation (inverted for home_prob)
                z_90 = 1.645
                home_probs = 1 - probs
                predictions["ci_lower"] = np.clip(home_probs - z_90 * uncertainty, 0, 1)
                predictions["ci_upper"] = np.clip(home_probs + z_90 * uncertainty, 0, 1)
            except Exception as e:
                logger.warning(f"Bayesian CI failed, using approximation: {e}")
                # Fall back to normal approximation (inverted for home_prob)
                z_90 = 1.645
                home_probs = 1 - probs
                predictions["ci_lower"] = np.clip(home_probs - z_90 * uncertainty, 0, 1)
                predictions["ci_upper"] = np.clip(home_probs + z_90 * uncertainty, 0, 1)
        else:
            # Fall back to normal approximation if no Bayesian model (inverted for home_prob)
            z_90 = 1.645
            home_probs = 1 - probs
            predictions["ci_lower"] = np.clip(home_probs - z_90 * uncertainty, 0, 1)
            predictions["ci_upper"] = np.clip(home_probs + z_90 * uncertainty, 0, 1)

        # Add injury adjustments
        if self.use_injuries and self.injury_builder:
            predictions = self._add_injury_data(predictions)
            if self.injury_adjuster:
                # Adjust probabilities based on injuries
                adj_probs = []
                for _, row in predictions.iterrows():
                    base_prob = row["stack_home_prob"]
                    home_impact = row.get("home_injury_impact", 0)
                    away_impact = row.get("away_injury_impact", 0)
                    # Net impact: positive means home team healthier
                    net_impact = away_impact - home_impact
                    # Adjust probability (up to +/- 5% for major injuries)
                    adj = min(max(net_impact * 0.02, -0.05), 0.05)
                    adj_probs.append(min(max(base_prob + adj, 0.05), 0.95))
                predictions["stack_home_prob_adj"] = adj_probs
                predictions["stack_away_prob_adj"] = 1 - predictions["stack_home_prob_adj"]
            else:
                predictions["stack_home_prob_adj"] = predictions["stack_home_prob"]
                predictions["stack_away_prob_adj"] = predictions["stack_away_prob"]
        else:
            predictions["stack_home_prob_adj"] = predictions["stack_home_prob"]
            predictions["stack_away_prob_adj"] = predictions["stack_away_prob"]

        # Merge with market odds
        ml_odds = odds[odds["market"] == "moneyline"]
        market_probs = self._get_market_consensus(ml_odds)
        predictions = predictions.merge(market_probs, on="game_id", how="left")

        # Calculate edge vs market
        predictions["home_edge"] = predictions["stack_home_prob_adj"] - predictions["market_home_prob"]
        predictions["away_edge"] = predictions["stack_away_prob_adj"] - predictions["market_away_prob"]

        # Get best odds
        best_odds = self._get_best_moneyline_odds(ml_odds)
        predictions = predictions.merge(best_odds, on="game_id", how="left")

        # Kelly sizing (reduce bet size when uncertainty is high)
        def calc_stacking_kelly(row):
            prob = row["stack_home_prob_adj"]
            edge = row["home_edge"]
            odds_val = row.get("best_home_odds")

            # Skip if no edge or no odds
            if edge < self.min_edge or pd.isna(odds_val):
                return 0.0

            # Base Kelly calculation
            base_kelly = self.sizer.calculate_kelly(prob, odds_val)

            # Use Bayesian uncertainty if available, otherwise use model disagreement
            uncertainty = row.get("bayesian_uncertainty", row.get("stack_uncertainty", 0))

            # More principled uncertainty penalty using posterior variance
            if uncertainty > 0.05:
                # Scale bet inversely with uncertainty squared (variance)
                confidence = max(0.1, 1.0 - (uncertainty ** 2) * 10)
                return base_kelly * confidence

            return base_kelly

        def calc_away_kelly(row):
            prob = row["stack_away_prob_adj"]
            edge = row["away_edge"]
            odds_val = row.get("best_away_odds")

            # Skip if no edge or no odds
            if edge < self.min_edge or pd.isna(odds_val):
                return 0.0

            # Base Kelly calculation
            base_kelly = self.sizer.calculate_kelly(prob, odds_val)

            # Use Bayesian uncertainty if available, otherwise use model disagreement
            uncertainty = row.get("bayesian_uncertainty", row.get("stack_uncertainty", 0))

            # More principled uncertainty penalty using posterior variance
            if uncertainty > 0.05:
                # Scale bet inversely with uncertainty squared (variance)
                confidence = max(0.1, 1.0 - (uncertainty ** 2) * 10)
                return base_kelly * confidence

            return base_kelly

        predictions["home_kelly"] = predictions.apply(calc_stacking_kelly, axis=1)
        predictions["away_kelly"] = predictions.apply(calc_away_kelly, axis=1)

        # Bet recommendations (with uncertainty filter)
        # Higher bar for uncertain predictions
        predictions["bet_home"] = (
            (predictions["home_edge"] >= self.min_edge) &
            (predictions["stack_uncertainty"] < 0.15)  # Skip highly uncertain games
        )
        predictions["bet_away"] = (
            (predictions["away_edge"] >= self.min_edge) &
            (predictions["stack_uncertainty"] < 0.15)
        )
        predictions["bet_type"] = "stacking"

        # Confidence level based on edge and uncertainty
        def get_confidence(row):
            if not row["bet_home"] and not row["bet_away"]:
                return "NONE"
            edge = max(row["home_edge"], row["away_edge"])
            uncertainty = row["stack_uncertainty"]
            if edge >= 0.08 and uncertainty < 0.05:
                return "HIGH"
            elif edge >= 0.05 and uncertainty < 0.10:
                return "MEDIUM"
            return "LOW"

        predictions["confidence"] = predictions.apply(get_confidence, axis=1)

        home_bets = predictions["bet_home"].sum()
        away_bets = predictions["bet_away"].sum()
        logger.info(f"Generated stacking predictions: {home_bets} home, {away_bets} away")

        return predictions

    def _generate_matchup_predictions(
        self,
        features: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions using matchup analysis.

        Uses historical head-to-head performance, rivalry intensity,
        and style matchup factors to adjust base predictions.
        """
        if features.empty:
            return pd.DataFrame()

        predictions = features[["game_id", "commence_time", "home_team", "away_team"]].copy()

        # Get matchup stats for each game
        matchup_data = []
        for _, row in features.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            game_date = pd.to_datetime(row["commence_time"])

            try:
                # Get matchup statistics
                stats = self.matchup_builder.get_matchup_stats(
                    home_team, away_team, game_date, self.historical_games
                )

                matchup_data.append({
                    "game_id": row["game_id"],
                    "h2h_home_wins": stats.h2h_record[0],
                    "h2h_away_wins": stats.h2h_record[1],
                    "h2h_margin": stats.avg_home_margin,
                    "h2h_total": stats.avg_total_points,
                    "h2h_sample": stats.sample_size,
                    "rivalry_factor": stats.rivalry_factor,
                    "is_rivalry": 1 if stats.rivalry_factor > 0.3 else 0,
                })
            except Exception as e:
                logger.debug(f"Error getting matchup stats for {home_team} vs {away_team}: {e}")
                matchup_data.append({
                    "game_id": row["game_id"],
                    "h2h_home_wins": 0,
                    "h2h_away_wins": 0,
                    "h2h_margin": 0.0,
                    "h2h_total": 215.0,
                    "h2h_sample": 0,
                    "rivalry_factor": 0.0,
                    "is_rivalry": 0,
                })

        # Merge matchup stats
        predictions = predictions.merge(pd.DataFrame(matchup_data), on="game_id", how="left")

        # Calculate H2H win rate (with bayesian prior toward 50%)
        prior_games = 4  # Equivalent of 4 games at 50%
        predictions["h2h_home_winrate"] = (
            (predictions["h2h_home_wins"] + prior_games / 2) /
            (predictions["h2h_home_wins"] + predictions["h2h_away_wins"] + prior_games)
        )

        # Calculate matchup edge factor (-1 to +1 scale)
        # Positive = home team has H2H advantage
        predictions["matchup_edge"] = (predictions["h2h_home_winrate"] - 0.5) * 2

        # Scale by sample size confidence (more games = more confident)
        predictions["sample_confidence"] = predictions["h2h_sample"].clip(0, 10) / 10
        predictions["matchup_edge_adj"] = (
            predictions["matchup_edge"] * predictions["sample_confidence"]
        )

        # Rivalry adjustment (rivalries tend to be closer games - reduce confidence)
        predictions["rivalry_penalty"] = predictions["rivalry_factor"] * 0.1

        # Merge with market odds
        ml_odds = odds[odds["market"] == "moneyline"]
        market_probs = self._get_market_consensus(ml_odds)
        predictions = predictions.merge(market_probs, on="game_id", how="left")

        # Best odds
        best_ml = self._get_best_moneyline_odds(ml_odds)
        predictions = predictions.merge(best_ml, on="game_id", how="left")

        # Compute matchup-adjusted probability
        # Start with market implied, then adjust based on matchup
        predictions["matchup_home_prob"] = predictions["market_home_prob"].fillna(0.5)
        predictions["matchup_away_prob"] = predictions["market_away_prob"].fillna(0.5)

        # Apply matchup adjustment (up to 5% shift based on H2H)
        max_shift = 0.05
        shift = predictions["matchup_edge_adj"] * max_shift
        predictions["matchup_home_prob"] = (predictions["matchup_home_prob"] + shift).clip(0.1, 0.9)
        predictions["matchup_away_prob"] = 1 - predictions["matchup_home_prob"]

        # Calculate edge vs market
        predictions["home_edge"] = predictions["matchup_home_prob"] - predictions["market_home_prob"].fillna(0.5)
        predictions["away_edge"] = predictions["matchup_away_prob"] - predictions["market_away_prob"].fillna(0.5)

        # Determine bet recommendations
        predictions["bet_home"] = (predictions["home_edge"] >= self.min_edge).astype(int)
        predictions["bet_away"] = (predictions["away_edge"] >= self.min_edge).astype(int)

        # Kelly sizing (reduced due to matchup-only signal)
        def calc_matchup_kelly(row, is_away=False):
            prob = row["matchup_away_prob"] if is_away else row["matchup_home_prob"]
            edge = row["away_edge"] if is_away else row["home_edge"]
            odds_val = row.get("best_away_odds") if is_away else row.get("best_home_odds")

            if edge < self.min_edge or pd.isna(odds_val):
                return 0.0

            base_kelly = self.sizer.calculate_kelly(prob, odds_val)

            # Reduce bet size for low sample sizes and rivalries
            sample_penalty = 1.0 - (1.0 - row["sample_confidence"]) * 0.3
            rivalry_penalty = 1.0 - row["rivalry_penalty"]

            return base_kelly * sample_penalty * rivalry_penalty

        predictions["home_kelly"] = predictions.apply(lambda r: calc_matchup_kelly(r, False), axis=1)
        predictions["away_kelly"] = predictions.apply(lambda r: calc_matchup_kelly(r, True), axis=1)

        # Confidence level based on H2H sample and rivalry
        def get_confidence(row):
            sample = row["h2h_sample"]
            rivalry = row["is_rivalry"]
            edge = max(abs(row["home_edge"]), abs(row["away_edge"]))

            if sample >= 6 and edge >= 0.03 and not rivalry:
                return "HIGH"
            elif sample >= 3 and edge >= 0.02:
                return "MEDIUM"
            elif sample >= 1:
                return "LOW"
            return "NONE"

        predictions["confidence"] = predictions.apply(get_confidence, axis=1)

        home_bets = predictions["bet_home"].sum()
        away_bets = predictions["bet_away"].sum()
        logger.info(f"Generated matchup predictions: {home_bets} home, {away_bets} away")

        return predictions

    def _add_injury_data(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add injury information to predictions."""
        injury_data = []
        for _, row in predictions.iterrows():
            home_team = self._normalize_team_name(row["home_team"])
            away_team = self._normalize_team_name(row["away_team"])

            features = self.injury_builder.get_game_injury_features(home_team, away_team)
            injury_data.append({
                "game_id": row["game_id"],
                "home_injury_impact": features["home_injury_impact"],
                "away_injury_impact": features["away_injury_impact"],
                "home_star_out": features["home_star_out"],
                "away_star_out": features["away_star_out"],
            })

        return predictions.merge(pd.DataFrame(injury_data), on="game_id", how="left")

    def _get_market_consensus(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get average market probability across bookmakers."""
        home_probs = (
            odds[odds["team"] == "home"]
            .groupby("game_id")["no_vig_prob"]
            .mean()
            .reset_index()
            .rename(columns={"no_vig_prob": "market_home_prob"})
        )
        away_probs = (
            odds[odds["team"] == "away"]
            .groupby("game_id")["no_vig_prob"]
            .mean()
            .reset_index()
            .rename(columns={"no_vig_prob": "market_away_prob"})
        )
        return home_probs.merge(away_probs, on="game_id")

    def _get_best_moneyline_odds(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get best available moneyline odds."""
        best_home = (
            odds[odds["team"] == "home"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_home_odds", "bookmaker": "home_book"})
        )
        best_away = (
            odds[odds["team"] == "away"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_away_odds", "bookmaker": "away_book"})
        )
        return best_home.merge(best_away, on="game_id")

    def _get_best_spread_odds(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get best spread odds for home team."""
        best = (
            odds[odds["team"] == "home"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "line", "bookmaker"]]
            .rename(columns={"odds": "best_spread_odds", "bookmaker": "spread_book"})
        )
        return best

    def _get_best_totals_odds(self, odds: pd.DataFrame) -> pd.DataFrame:
        """Get best over/under odds."""
        best_over = (
            odds[odds["team"] == "over"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_over_odds", "bookmaker": "over_book"})
        )
        best_under = (
            odds[odds["team"] == "under"]
            .sort_values("odds", ascending=False)
            .groupby("game_id")
            .first()
            .reset_index()[["game_id", "odds", "bookmaker"]]
            .rename(columns={"odds": "best_under_odds", "bookmaker": "under_book"})
        )
        return best_over.merge(best_under, on="game_id")

    def print_all_predictions(self, predictions: Dict[str, pd.DataFrame]):
        """Print formatted predictions for all bet types."""
        print("\n" + "=" * 80)
        print("NBA PREDICTIONS - ALL MARKETS")
        print("=" * 80)

        # Get unique games
        all_games = set()
        for df in predictions.values():
            if not df.empty:
                all_games.update(df["game_id"].unique())

        for game_id in all_games:
            game_printed = False

            # Moneyline
            if "moneyline" in predictions and not predictions["moneyline"].empty:
                ml = predictions["moneyline"]
                game = ml[ml["game_id"] == game_id]
                if not game.empty:
                    row = game.iloc[0]
                    if not game_printed:
                        print(f"\n{row['away_team']} @ {row['home_team']}")
                        print(f"  Time: {row['commence_time']}")
                        game_printed = True

                    print(f"\n  MONEYLINE:")
                    print(f"    Model:  Home {row['model_home_prob']:.1%} | Away {row['model_away_prob']:.1%}")
                    print(f"    Market: Home {row['market_home_prob']:.1%} | Away {row['market_away_prob']:.1%}")
                    print(f"    Edge:   Home {row['home_edge']:+.1%} | Away {row['away_edge']:+.1%}")

                    if row["bet_home"]:
                        print(f"    >>> BET HOME at {row['best_home_odds']:+.0f} ({row['home_book']}) - Kelly: {row['home_kelly']:.1%}")
                    elif row["bet_away"]:
                        print(f"    >>> BET AWAY at {row['best_away_odds']:+.0f} ({row['away_book']}) - Kelly: {row['away_kelly']:.1%}")

            # Spread
            if "spread" in predictions and not predictions["spread"].empty:
                sp = predictions["spread"]
                game = sp[sp["game_id"] == game_id]
                if not game.empty:
                    row = game.iloc[0]
                    if not game_printed:
                        print(f"\n{row['away_team']} @ {row['home_team']}")
                        print(f"  Time: {row['commence_time']}")
                        game_printed = True

                    print(f"\n  SPREAD ({row['spread_line']:+.1f}):")
                    print(f"    Expected diff: {row['expected_diff']:+.1f} pts")
                    print(f"    Model cover:   {row['model_cover_prob']:.1%}")
                    print(f"    Market cover:  {row['market_cover_prob']:.1%}")
                    print(f"    Edge:          {row['cover_edge']:+.1%}")

                    if row["bet_cover"]:
                        print(f"    >>> BET HOME {row['spread_line']:+.1f} at {row['best_spread_odds']:+.0f} - Kelly: {row['kelly']:.1%}")

            # Totals
            if "totals" in predictions and not predictions["totals"].empty:
                tot = predictions["totals"]
                game = tot[tot["game_id"] == game_id]
                if not game.empty:
                    row = game.iloc[0]
                    if not game_printed:
                        print(f"\n{row['away_team']} @ {row['home_team']}")
                        print(f"  Time: {row['commence_time']}")
                        game_printed = True

                    print(f"\n  TOTALS (O/U {row['total_line']:.1f}):")
                    print(f"    Expected total: {row['expected_total']:.1f} pts")
                    print(f"    Model:  Over {row['model_over_prob']:.1%} | Under {row['model_under_prob']:.1%}")
                    print(f"    Market: Over {row['market_over_prob']:.1%} | Under {row['market_under_prob']:.1%}")
                    print(f"    Edge:   Over {row['over_edge']:+.1%} | Under {row['under_edge']:+.1%}")

                    if row["bet_over"]:
                        print(f"    >>> BET OVER at {row['best_over_odds']:+.0f} ({row['over_book']}) - Kelly: {row['over_kelly']:.1%}")
                    elif row["bet_under"]:
                        print(f"    >>> BET UNDER at {row['best_under_odds']:+.0f} ({row['under_book']}) - Kelly: {row['under_kelly']:.1%}")

            # Stacking ensemble
            if "stacking" in predictions and not predictions["stacking"].empty:
                stk = predictions["stacking"]
                game = stk[stk["game_id"] == game_id]
                if not game.empty:
                    row = game.iloc[0]
                    if not game_printed:
                        print(f"\n{row['away_team']} @ {row['home_team']}")
                        print(f"  Time: {row['commence_time']}")
                        game_printed = True

                    print(f"\n  STACKING ENSEMBLE:")
                    print(f"    Model:  Home {row['stack_home_prob_adj']:.1%} | Away {row['stack_away_prob_adj']:.1%}")
                    if 'market_home_prob' in row and pd.notna(row['market_home_prob']):
                        print(f"    Market: Home {row['market_home_prob']:.1%} | Away {row['market_away_prob']:.1%}")
                    print(f"    Edge:   Home {row['home_edge']:+.1%} | Away {row['away_edge']:+.1%}")
                    print(f"    Uncertainty: {row['stack_uncertainty']:.3f} | Confidence: {row['confidence']}")

                    if row["bet_home"]:
                        book = row.get('home_book', 'N/A')
                        print(f"    >>> BET HOME at {row['best_home_odds']:+.0f} ({book}) - Kelly: {row['home_kelly']:.1%}")
                    elif row["bet_away"]:
                        book = row.get('away_book', 'N/A')
                        print(f"    >>> BET AWAY at {row['best_away_odds']:+.0f} ({book}) - Kelly: {row['away_kelly']:.1%}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("-" * 80)

        total_bets = 0
        for bet_type, df in predictions.items():
            if df.empty:
                continue

            if bet_type == "moneyline":
                home_bets = df["bet_home"].sum()
                away_bets = df["bet_away"].sum()
                count = home_bets + away_bets
                print(f"  Moneyline: {count} bets ({home_bets} home, {away_bets} away)")
            elif bet_type == "spread":
                count = df["bet_cover"].sum()
                print(f"  Spread: {count} bets")
            elif bet_type == "totals":
                over_bets = df["bet_over"].sum()
                under_bets = df["bet_under"].sum()
                count = over_bets + under_bets
                print(f"  Totals: {count} bets ({over_bets} over, {under_bets} under)")
            elif bet_type == "stacking":
                home_bets = df["bet_home"].sum()
                away_bets = df["bet_away"].sum()
                count = home_bets + away_bets
                print(f"  Stacking: {count} bets ({home_bets} home, {away_bets} away)")
            else:
                count = 0

            total_bets += count

        print(f"\n  TOTAL: {total_bets} recommended bets")
        print("=" * 80)


def run_multi_predictions():
    """Run predictions for all bet types."""
    predictor = MultiPredictor(min_edge=0.03, kelly_fraction=0.2)

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

    # Generate all predictions
    predictions = predictor.generate_all_predictions(features, odds)

    # Print results
    predictor.print_all_predictions(predictions)

    # Save predictions
    date_str = datetime.now().strftime("%Y%m%d")
    for bet_type, df in predictions.items():
        if not df.empty:
            df.to_csv(f"data/predictions_{bet_type}_{date_str}.csv", index=False)

    return predictions


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")

    run_multi_predictions()
