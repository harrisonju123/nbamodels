"""
Totals Strategy

Betting strategy for over/under totals using the TotalsModel.
Evaluates expected total points vs market line to find edges.
"""

import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.betting.strategies.base import (
    BettingStrategy,
    BetSignal,
    StrategyType,
    odds_to_implied_prob,
)
from src.models.totals import TotalsModel


class TotalsStrategy(BettingStrategy):
    """
    Totals betting strategy using existing TotalsModel.

    Uses XGBoost regression model to predict expected total points,
    then calculates probability of going over/under various lines.

    Filters:
    - Minimum edge threshold
    - Optional B2B filter (avoid betting on teams on back-to-back)
    - Optional pace filter (only high/low pace games)
    """

    strategy_type = StrategyType.TOTALS
    priority = 1  # High priority - totals model exists and is validated

    def __init__(
        self,
        model_path: str = "models/totals_model.pkl",
        min_edge: float = 0.05,
        require_no_b2b: bool = False,
        min_pace: Optional[float] = None,
        max_pace: Optional[float] = None,
        max_daily_bets: int = 5,
        enabled: bool = True,
    ):
        """
        Initialize totals strategy.

        Args:
            model_path: Path to trained TotalsModel
            min_edge: Minimum edge to trigger bet (default 5%)
            require_no_b2b: If True, skip games where either team is on B2B
            min_pace: Optional minimum pace filter (possessions per game)
            max_pace: Optional maximum pace filter (possessions per game)
            max_daily_bets: Maximum bets per day
            enabled: Whether strategy is active
        """
        super().__init__(
            min_edge=min_edge,
            max_daily_bets=max_daily_bets,
            enabled=enabled,
        )

        self.model_path = model_path
        self.require_no_b2b = require_no_b2b
        self.min_pace = min_pace
        self.max_pace = max_pace

        # Load model
        if os.path.exists(model_path):
            self.model = TotalsModel.load(model_path)
            logger.info(f"TotalsStrategy loaded model from {model_path}")
        else:
            logger.warning(f"TotalsModel not found at {model_path}, strategy will fail")
            self.model = None

    def evaluate_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        features: Dict,
        odds_data: Dict,
        **kwargs
    ) -> List[BetSignal]:
        """
        Evaluate a single game for totals betting opportunities.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            features: Dict of game features for model
            odds_data: Dict with 'total_line', 'over_odds', 'under_odds', 'bookmaker'
            **kwargs: Optional home_b2b, away_b2b, pace

        Returns:
            List of BetSignal objects (0-2 signals: over and/or under)
        """
        if self.model is None:
            logger.error("TotalsModel not loaded, cannot evaluate game")
            return []

        # Extract odds data
        total_line = odds_data.get('total_line')
        over_odds = odds_data.get('over_odds', -110)
        under_odds = odds_data.get('under_odds', -110)
        bookmaker = odds_data.get('bookmaker')

        if total_line is None:
            logger.debug(f"{game_id}: No total line available")
            return []

        # Apply filters
        filters_passed = []

        # B2B filter
        home_b2b = kwargs.get('home_b2b', False)
        away_b2b = kwargs.get('away_b2b', False)
        if self.require_no_b2b and (home_b2b or away_b2b):
            logger.debug(f"{game_id}: Filtered out by B2B (home={home_b2b}, away={away_b2b})")
            return []
        if not home_b2b and not away_b2b:
            filters_passed.append("no_b2b")

        # Pace filter
        pace = kwargs.get('pace')
        if self.min_pace is not None and pace is not None:
            if pace < self.min_pace:
                logger.debug(f"{game_id}: Filtered out by min_pace ({pace:.1f} < {self.min_pace:.1f})")
                return []
            filters_passed.append(f"pace>={self.min_pace:.0f}")

        if self.max_pace is not None and pace is not None:
            if pace > self.max_pace:
                logger.debug(f"{game_id}: Filtered out by max_pace ({pace:.1f} > {self.max_pace:.1f})")
                return []
            filters_passed.append(f"pace<={self.max_pace:.0f}")

        # Get model prediction
        features_df = pd.DataFrame([features])

        try:
            # Predict over probability
            over_prob = self.model.predict_over_prob(features_df, total_line)[0]
            under_prob = 1 - over_prob

            # Get market implied probabilities
            over_market_prob = odds_to_implied_prob(over_odds)
            under_market_prob = odds_to_implied_prob(under_odds)

            # Calculate edges
            over_edge = over_prob - over_market_prob
            under_edge = under_prob - under_market_prob

            signals = []

            # Check over edge
            if over_edge >= self.min_edge:
                confidence = self._determine_confidence(
                    over_edge,
                    filters_passed,
                    additional_score=min(over_prob, 1.0)  # Higher prob = higher confidence
                )

                signals.append(BetSignal(
                    strategy_type=StrategyType.TOTALS,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    bet_type="totals",
                    bet_side="OVER",
                    model_prob=over_prob,
                    market_prob=over_market_prob,
                    edge=over_edge,
                    line=total_line,
                    odds=over_odds,
                    confidence=confidence,
                    bookmaker=bookmaker,
                    filters_passed=filters_passed.copy(),
                ))

            # Check under edge
            if under_edge >= self.min_edge:
                confidence = self._determine_confidence(
                    under_edge,
                    filters_passed,
                    additional_score=min(under_prob, 1.0)
                )

                signals.append(BetSignal(
                    strategy_type=StrategyType.TOTALS,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    bet_type="totals",
                    bet_side="UNDER",
                    model_prob=under_prob,
                    market_prob=under_market_prob,
                    edge=under_edge,
                    line=total_line,
                    odds=under_odds,
                    confidence=confidence,
                    bookmaker=bookmaker,
                    filters_passed=filters_passed.copy(),
                ))

            return signals

        except Exception as e:
            logger.error(f"Error evaluating {game_id}: {e}", exc_info=True)
            return []

    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate multiple games for totals betting.

        Args:
            games_df: DataFrame with game_id, home_team, away_team, home_b2b, away_b2b
            features_df: DataFrame with model features (must match TotalsModel feature columns)
            odds_df: DataFrame with game_id, total_line, over_odds, under_odds, bookmaker

        Returns:
            List of all BetSignal objects from batch evaluation
        """
        all_signals = []

        # Merge games with odds
        # Keep only game_id, home_team, away_team, home_b2b, away_b2b from games_df
        # to avoid column conflicts
        games_subset = games_df[['game_id', 'home_team', 'away_team', 'home_b2b', 'away_b2b']].copy()
        # Merge on game_id, home_team, and away_team to avoid creating duplicate columns
        merged = games_subset.merge(odds_df, on=['game_id', 'home_team', 'away_team'], how='inner')

        if len(merged) == 0:
            logger.warning("No games with odds data available")
            return all_signals

        logger.info(f"Evaluating {len(merged)} games for totals betting")

        for _, row in merged.iterrows():
            # Get features for this game
            game_features = features_df[features_df['game_id'] == row['game_id']]

            if len(game_features) == 0:
                logger.warning(f"No features found for {row['game_id']}")
                continue

            # Convert features to dict
            features_dict = game_features.iloc[0].to_dict()

            # Build odds data dict
            odds_data = {
                'total_line': row.get('total_line'),
                'over_odds': row.get('over_odds', -110),
                'under_odds': row.get('under_odds', -110),
                'bookmaker': row.get('bookmaker'),
            }

            # Evaluate game
            signals = self.evaluate_game(
                game_id=row['game_id'],
                home_team=row['home_team'],
                away_team=row['away_team'],
                features=features_dict,
                odds_data=odds_data,
                home_b2b=row.get('home_b2b', False),
                away_b2b=row.get('away_b2b', False),
                pace=features_dict.get('pace', None),
            )

            all_signals.extend(signals)

        logger.info(
            f"TotalsStrategy found {len(all_signals)} signals from {len(merged)} games"
        )

        return all_signals

    @classmethod
    def high_pace_strategy(cls) -> 'TotalsStrategy':
        """
        Create strategy focused on high-pace games only.

        High-pace games tend to be more predictable for totals.
        """
        return cls(
            min_edge=0.05,
            min_pace=100,  # Only games with 100+ possessions
            require_no_b2b=True,
        )

    @classmethod
    def low_pace_strategy(cls) -> 'TotalsStrategy':
        """
        Create strategy focused on low-pace games only.

        Low-pace games may have market inefficiencies.
        """
        return cls(
            min_edge=0.05,
            max_pace=95,  # Only games with <95 possessions
            require_no_b2b=True,
        )

    @classmethod
    def conservative_strategy(cls) -> 'TotalsStrategy':
        """
        Create conservative strategy with higher edge threshold.

        Higher edge requirement and B2B filter for safety.
        """
        return cls(
            min_edge=0.07,  # 7% minimum edge
            require_no_b2b=True,
        )
