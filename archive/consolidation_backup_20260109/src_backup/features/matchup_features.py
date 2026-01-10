"""
Matchup Feature Engineering

Builds team-vs-team specific features for game prediction including:
- Head-to-head historical performance
- Style/pace matchup interactions
- Division and rivalry effects
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MatchupStats:
    """Historical matchup statistics between two teams."""
    home_team: str
    away_team: str
    h2h_record: Tuple[int, int]  # (home_wins, away_wins)
    avg_home_margin: float
    avg_total_points: float
    last_n_games: List[Dict]
    home_ats_record: Tuple[int, int]  # Against the spread
    rivalry_factor: float  # 0-1, intensity of rivalry
    sample_size: int


class MatchupFeatureBuilder:
    """
    Builds matchup-specific features for game prediction.

    Categories:
    1. Historical H2H performance
    2. Style matchup (pace, efficiency interactions)
    3. Division/Conference effects
    4. Rivalry intensity
    """

    # NBA Division structure
    DIVISIONS = {
        "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
        "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
        "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
        "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
        "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
        "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
    }

    # Conference mapping
    CONFERENCES = {
        "East": ["Atlantic", "Central", "Southeast"],
        "West": ["Northwest", "Pacific", "Southwest"],
    }

    # Historic rivalries with extra intensity (unordered pairs)
    HISTORIC_RIVALRIES = [
        frozenset(["LAL", "BOS"]),   # Lakers-Celtics
        frozenset(["LAL", "LAC"]),   # Battle of LA
        frozenset(["NYK", "BKN"]),   # New York
        frozenset(["CHI", "DET"]),   # 80s-90s rivalry
        frozenset(["MIA", "BOS"]),   # 2010s playoff rivalry
        frozenset(["GSW", "CLE"]),   # Finals rivalry
        frozenset(["GSW", "LAL"]),   # California rivalry
        frozenset(["PHI", "BOS"]),   # Historic Eastern rivalry
        frozenset(["DAL", "SAS"]),   # Texas rivalry
        frozenset(["DAL", "HOU"]),   # Texas rivalry
        frozenset(["LAL", "SAS"]),   # 2000s playoff rivalry
    ]

    def __init__(
        self,
        h2h_window_games: int = 10,
        h2h_window_seasons: int = 3,
        recency_decay: float = 0.9,
    ):
        """
        Initialize matchup feature builder.

        Args:
            h2h_window_games: Maximum H2H games to consider
            h2h_window_seasons: Only use games from recent N seasons
            recency_decay: Exponential decay for recency weighting (0.9 = 10% decay per game)
        """
        # Validate parameters
        if h2h_window_games < 1:
            raise ValueError(f"h2h_window_games must be >= 1, got {h2h_window_games}")
        if h2h_window_seasons < 1:
            raise ValueError(f"h2h_window_seasons must be >= 1, got {h2h_window_seasons}")
        if not (0.0 < recency_decay <= 1.0):
            raise ValueError(
                f"recency_decay must be in range (0, 1], got {recency_decay}. "
                "Use values like 0.9 for 10% decay per game."
            )

        self.h2h_window_games = h2h_window_games
        self.h2h_window_seasons = h2h_window_seasons
        self.recency_decay = recency_decay

        # Build reverse lookups
        self._team_to_division = {}
        self._team_to_conference = {}
        for div_name, teams in self.DIVISIONS.items():
            for team in teams:
                self._team_to_division[team] = div_name
        for conf_name, divisions in self.CONFERENCES.items():
            for div_name in divisions:
                for team in self.DIVISIONS[div_name]:
                    self._team_to_conference[team] = conf_name

    def build_h2h_features(
        self,
        games_df: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Build head-to-head features for each game.

        Features:
        - h2h_home_win_rate: Win rate for home team vs this opponent
        - h2h_home_margin: Average margin when home team plays this opponent
        - h2h_total_points: Average total points in H2H matchups
        - h2h_sample_size: Number of H2H games (for uncertainty)
        - h2h_recency_weighted_margin: More recent games weighted higher
        """
        # Validate input
        if games_df.empty:
            logger.warning("Empty games_df provided to build_h2h_features")
            return pd.DataFrame(columns=[
                "game_id", "h2h_home_win_rate", "h2h_home_margin",
                "h2h_total_points", "h2h_sample_size", "h2h_recency_weighted_margin"
            ])

        # Validate required columns
        required_cols = ["date", "home_team", "away_team", "season", "game_id", "home_score", "away_score"]
        missing = [col for col in required_cols if col not in games_df.columns]
        if missing:
            raise ValueError(f"games_df missing required columns: {missing}")

        games_df = games_df.copy()
        games_df["date"] = pd.to_datetime(games_df["date"])
        games_df = games_df.sort_values("date").reset_index(drop=True)

        h2h_features = []

        for idx, game in games_df.iterrows():
            game_date = game["date"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            season = game["season"]

            # Validate season is numeric for H2H window calculation
            if not isinstance(season, (int, np.integer, float)):
                raise ValueError(
                    f"Season must be numeric for H2H window calculation, "
                    f"got {type(season).__name__}: {season}"
                )

            # Get historical H2H games before this game
            min_season = int(season) - self.h2h_window_seasons
            h2h_mask = (
                (games_df["date"] < game_date) &
                (games_df["season"] >= min_season) &
                (
                    ((games_df["home_team"] == home_team) & (games_df["away_team"] == away_team)) |
                    ((games_df["home_team"] == away_team) & (games_df["away_team"] == home_team))
                )
            )
            h2h_games = games_df[h2h_mask].tail(self.h2h_window_games)

            # Data quality check: all games have zero scores
            if len(h2h_games) > 0:
                total_scores = h2h_games[["home_score", "away_score"]].sum().sum()
                if total_scores == 0:
                    logger.warning(
                        f"All H2H games for {home_team} vs {away_team} have zero scores. "
                        "Data quality issue detected, treating as no data."
                    )
                    h2h_games = pd.DataFrame()  # Treat as no data

            features = self._compute_h2h_stats(home_team, away_team, h2h_games)
            features["game_id"] = game["game_id"]
            h2h_features.append(features)

        return pd.DataFrame(h2h_features)

    def _compute_h2h_stats(
        self,
        home_team: str,
        away_team: str,
        h2h_games: pd.DataFrame,
    ) -> Dict:
        """Compute H2H statistics for a specific matchup."""
        n_games = len(h2h_games)

        if n_games == 0:
            return {
                "h2h_home_win_rate": 0.5,  # Prior: 50%
                "h2h_home_margin": 0.0,
                "h2h_total_points": 215.0,  # League average
                "h2h_sample_size": 0,
                "h2h_recency_weighted_margin": 0.0,
            }

        # Calculate home team's performance against this opponent
        home_wins = 0
        home_margins = []
        totals = []
        recency_weights = []

        # Pre-check for potential underflow in recency weights
        # log(min_weight) = (n_games - 1) * log(decay)
        # For float64, min normal value ~2.2e-308, so we check if exponent < -700
        max_exponent = n_games - 1
        use_recency_weights = True
        if max_exponent > 0 and max_exponent * np.log(self.recency_decay) < -700:
            logger.warning(
                f"Recency decay will underflow (decay={self.recency_decay}, "
                f"n_games={n_games}). Using uniform weights."
            )
            use_recency_weights = False

        for i, (_, game) in enumerate(h2h_games.iterrows()):
            # Weight more recent games higher (or uniform if underflow risk)
            if use_recency_weights:
                weight = self.recency_decay ** (n_games - i - 1)
            else:
                weight = 1.0
            recency_weights.append(weight)

            if game["home_team"] == home_team:
                # Home team was home in this game
                margin = game["home_score"] - game["away_score"]
                won = game["home_score"] > game["away_score"]
            else:
                # Home team was away in this game
                margin = game["away_score"] - game["home_score"]
                won = game["away_score"] > game["home_score"]

            if won:
                home_wins += 1
            home_margins.append(margin)
            totals.append(game["home_score"] + game["away_score"])

        # Compute weighted margin (handle zero weights edge case)
        weights = np.array(recency_weights)
        weight_sum = weights.sum()

        # Check for underflow - use exact zero check first
        if weight_sum == 0.0:
            logger.warning(
                f"Recency weights underflowed to exactly zero (decay={self.recency_decay}, "
                f"n_games={n_games}). Using uniform average. Consider increasing decay parameter."
            )
            recency_weighted_margin = np.mean(home_margins) if home_margins else 0.0
        elif weight_sum < 1e-100:
            logger.warning(
                f"Recency weights very small (sum={weight_sum:.6e}). Using uniform average."
            )
            recency_weighted_margin = np.mean(home_margins) if home_margins else 0.0
        else:
            weights = weights / weight_sum
            recency_weighted_margin = np.average(home_margins, weights=weights)

        return {
            "h2h_home_win_rate": home_wins / n_games if n_games > 0 else 0.5,
            "h2h_home_margin": np.mean(home_margins) if home_margins else 0.0,
            "h2h_total_points": np.mean(totals) if totals else 215.0,
            "h2h_sample_size": n_games,
            "h2h_recency_weighted_margin": recency_weighted_margin,
        }

    def build_style_matchup_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build style/pace matchup features.

        Features:
        - pace_matchup: Expected game pace based on both teams
        - tempo_mismatch: Absolute difference in team tempos
        - offensive_style_clash: Difference in 3PT attempt rates

        Args:
            games_df: Game data with home_team, away_team
            team_stats_df: Team rolling stats with pace, 3pt_rate, etc.
                           Expected columns: team, date, pace_Xg, off_rating_Xg, def_rating_Xg
        """
        games_df = games_df.copy()
        games_df["date"] = pd.to_datetime(games_df["date"])

        if "pace_5g" not in team_stats_df.columns:
            logger.warning("No pace columns in team_stats_df, skipping style features")
            return pd.DataFrame({"game_id": games_df["game_id"]})

        team_stats_df = team_stats_df.copy()
        team_stats_df["date"] = pd.to_datetime(team_stats_df["date"])

        style_features = []

        # League averages for normalization
        if "pace_5g" in team_stats_df.columns:
            league_avg_pace = team_stats_df["pace_5g"].mean()
            # Validate computed average - NBA pace typically 95-110
            if not np.isfinite(league_avg_pace):
                logger.warning(
                    f"Computed league_avg_pace is NaN/Inf, using default 100.0"
                )
                league_avg_pace = 100.0
            elif league_avg_pace < 80.0 or league_avg_pace > 120.0:
                logger.warning(
                    f"Computed league_avg_pace ({league_avg_pace:.1f}) outside "
                    "reasonable range [80, 120], using default 100.0"
                )
                league_avg_pace = 100.0
        else:
            league_avg_pace = 100.0

        for _, game in games_df.iterrows():
            game_date = game["date"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Get most recent stats before game date
            home_stats = self._get_team_stats_before_date(
                team_stats_df, home_team, game_date
            )
            away_stats = self._get_team_stats_before_date(
                team_stats_df, away_team, game_date
            )

            features = self._compute_style_matchup(
                home_stats, away_stats, league_avg_pace
            )
            features["game_id"] = game["game_id"]
            style_features.append(features)

        return pd.DataFrame(style_features)

    def _get_team_stats_before_date(
        self,
        team_stats_df: pd.DataFrame,
        team: str,
        date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """Get most recent team stats before a given date."""
        mask = (team_stats_df["team"] == team) & (team_stats_df["date"] < date)
        team_data = team_stats_df[mask]

        if len(team_data) == 0:
            return None

        return team_data.iloc[-1]

    def _compute_style_matchup(
        self,
        home_stats: Optional[pd.Series],
        away_stats: Optional[pd.Series],
        league_avg_pace: float,
    ) -> Dict:
        """Compute style matchup features from team stats."""
        if home_stats is None or away_stats is None:
            missing = []
            if home_stats is None:
                missing.append("home")
            if away_stats is None:
                missing.append("away")
            logger.debug(f"Missing stats for {', '.join(missing)} team(s), using defaults")
            return {
                "pace_matchup": 1.0,
                "tempo_mismatch": 0.0,
                "off_rating_matchup": 0.0,
                "def_rating_matchup": 0.0,
            }

        # Get pace values
        home_pace = home_stats.get("pace_5g", league_avg_pace)
        away_pace = away_stats.get("pace_5g", league_avg_pace)

        # Guard against invalid league average (zero, negative, or very small)
        if league_avg_pace < 1e-6:
            logger.warning(f"Invalid league_avg_pace ({league_avg_pace}), using defaults")
            pace_matchup = 1.0
            tempo_mismatch = 0.0
        else:
            # Expected game pace (geometric mean normalized)
            pace_matchup = (home_pace * away_pace) / (league_avg_pace ** 2)
            # Tempo mismatch (one team likes fast, other likes slow)
            tempo_mismatch = abs(home_pace - away_pace) / league_avg_pace

        # Offensive/Defensive rating matchup
        home_off = home_stats.get("pts_for_5g", 110.0)
        away_off = away_stats.get("pts_for_5g", 110.0)
        home_def = home_stats.get("pts_against_5g", 110.0)
        away_def = away_stats.get("pts_against_5g", 110.0)

        # How well does home offense match up vs away defense?
        off_rating_matchup = (home_off - away_def) / 10.0  # Normalize

        # How well does home defense match up vs away offense?
        def_rating_matchup = (away_off - home_def) / 10.0  # Inverse: positive = bad for home

        return {
            "pace_matchup": pace_matchup,
            "tempo_mismatch": tempo_mismatch,
            "off_rating_matchup": off_rating_matchup,
            "def_rating_matchup": def_rating_matchup,
        }

    def build_division_conference_features(
        self,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Division and conference game indicators.

        Features:
        - is_division_game: Play 4x/season, higher intensity
        - is_conference_game: Playoff implications
        - is_rivalry_game: Historic rivalry (Lakers-Celtics, etc.)
        - rivalry_intensity: 0-1 score based on matchup history
        """
        features = []

        for _, game in games_df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Division game check
            home_div = self._team_to_division.get(home_team)
            away_div = self._team_to_division.get(away_team)
            is_division = 1 if (home_div and home_div == away_div) else 0

            # Conference game check
            home_conf = self._team_to_conference.get(home_team)
            away_conf = self._team_to_conference.get(away_team)
            is_conference = 1 if (home_conf and home_conf == away_conf) else 0

            # Historic rivalry check
            matchup = frozenset([home_team, away_team])
            is_rivalry = 1 if matchup in self.HISTORIC_RIVALRIES else 0

            # Rivalry intensity score (0-1)
            # Division games = 0.3, Conference = 0.1, Historic = 0.5
            rivalry_intensity = is_division * 0.3 + is_rivalry * 0.5 + is_conference * 0.1

            features.append({
                "game_id": game["game_id"],
                "is_division_game": is_division,
                "is_conference_game": is_conference,
                "is_rivalry_game": is_rivalry,
                "rivalry_intensity": rivalry_intensity,
            })

        return pd.DataFrame(features)

    def build_all_matchup_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build all matchup features.

        Args:
            games_df: Game data with home_team, away_team, home_score, away_score, date, season
            team_stats_df: Optional team rolling stats for style matchup features

        Returns:
            DataFrame with all matchup features indexed by game_id
        """
        logger.info("Building matchup features...")

        # Start with H2H features
        result = self.build_h2h_features(games_df)

        # Add division/conference features
        div_features = self.build_division_conference_features(games_df)
        result = result.merge(div_features, on="game_id", how="left")

        # Add style matchup features if team stats available
        if team_stats_df is not None and not team_stats_df.empty:
            style_features = self.build_style_matchup_features(games_df, team_stats_df)
            result = result.merge(style_features, on="game_id", how="left")

        logger.info(f"Built {len(result.columns) - 1} matchup features for {len(result)} games")

        return result

    def get_matchup_stats(
        self,
        home_team: str,
        away_team: str,
        as_of_date: pd.Timestamp,
        games_df: pd.DataFrame,
    ) -> MatchupStats:
        """Get detailed matchup statistics for a specific pairing."""
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            raise ValueError("home_team and away_team must be strings")
        if home_team == away_team:
            raise ValueError("home_team and away_team cannot be the same")

        games_df = games_df.copy()
        games_df["date"] = pd.to_datetime(games_df["date"])

        # Filter H2H games
        h2h_mask = (
            (games_df["date"] < as_of_date) &
            (
                ((games_df["home_team"] == home_team) & (games_df["away_team"] == away_team)) |
                ((games_df["home_team"] == away_team) & (games_df["away_team"] == home_team))
            )
        )
        h2h_games = games_df[h2h_mask].tail(self.h2h_window_games)

        # Compute stats
        home_wins = 0
        away_wins = 0
        margins = []
        totals = []
        last_games = []

        for _, game in h2h_games.iterrows():
            total = game["home_score"] + game["away_score"]
            totals.append(total)

            if game["home_team"] == home_team:
                margin = game["home_score"] - game["away_score"]
                if margin > 0:
                    home_wins += 1
                elif margin < 0:
                    away_wins += 1
                # margin == 0 is a tie (should not happen in NBA, skip)
            else:
                margin = game["away_score"] - game["home_score"]
                if margin > 0:
                    home_wins += 1
                elif margin < 0:
                    away_wins += 1
                # margin == 0 is a tie (should not happen in NBA, skip)

            margins.append(margin)
            last_games.append({
                "date": game["date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
            })

        # Rivalry factor
        matchup = frozenset([home_team, away_team])
        home_div = self._team_to_division.get(home_team)
        away_div = self._team_to_division.get(away_team)

        rivalry_factor = 0.0
        if home_div == away_div:
            rivalry_factor += 0.3
        if matchup in self.HISTORIC_RIVALRIES:
            rivalry_factor += 0.5

        return MatchupStats(
            home_team=home_team,
            away_team=away_team,
            h2h_record=(home_wins, away_wins),
            avg_home_margin=np.mean(margins) if margins else 0.0,
            avg_total_points=np.mean(totals) if totals else 215.0,
            last_n_games=last_games[-5:],  # Last 5 only
            home_ats_record=(0, 0),  # Would need spread data
            rivalry_factor=rivalry_factor,
            sample_size=len(h2h_games),
        )


class MatchupAdjuster:
    """
    Adjusts model predictions based on matchup-specific factors.

    Can be used as a post-processing step on ensemble predictions.
    """

    def __init__(
        self,
        matchup_builder: Optional[MatchupFeatureBuilder] = None,
        adjustment_strength: float = 0.1,
    ):
        """
        Initialize matchup adjuster.

        Args:
            matchup_builder: MatchupFeatureBuilder instance
            adjustment_strength: How much to adjust predictions (0-1)
        """
        self.matchup_builder = matchup_builder or MatchupFeatureBuilder()
        self.adjustment_strength = adjustment_strength

    def adjust_prediction(
        self,
        base_pred: float,
        home_team: str,
        away_team: str,
        games_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Adjust base prediction using matchup factors.

        Args:
            base_pred: Base model prediction (spread or win probability)
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            games_df: Historical games data
            as_of_date: Date of game being predicted

        Returns:
            Tuple of (adjusted_pred, adjustment_breakdown)
        """
        stats = self.matchup_builder.get_matchup_stats(
            home_team, away_team, as_of_date, games_df
        )

        # Calculate adjustments
        adjustments = {}

        # H2H adjustment (if enough sample)
        if stats.sample_size >= 3:
            h2h_adj = stats.avg_home_margin * 0.1  # 10% of historical margin
            adjustments["h2h"] = h2h_adj * self.adjustment_strength
        else:
            adjustments["h2h"] = 0.0

        # Rivalry intensity (rivalries tend to be closer games)
        if stats.rivalry_factor > 0.3:
            # Pull prediction toward 0 (closer game)
            rivalry_adj = -base_pred * 0.05 * stats.rivalry_factor
            adjustments["rivalry"] = rivalry_adj * self.adjustment_strength
        else:
            adjustments["rivalry"] = 0.0

        # Total adjustment
        total_adjustment = sum(adjustments.values())
        adjusted_pred = base_pred + total_adjustment

        adjustments["total"] = total_adjustment
        adjustments["original"] = base_pred
        adjustments["adjusted"] = adjusted_pred

        return adjusted_pred, adjustments
