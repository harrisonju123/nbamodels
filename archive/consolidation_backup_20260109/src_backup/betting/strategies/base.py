"""
Base classes for betting strategies.

Provides unified interface for all betting strategies:
- BettingStrategy: Abstract base class
- BetSignal: Standardized betting signal output
- StrategyType: Enum for strategy types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum
import pandas as pd


class StrategyType(str, Enum):
    """Strategy type identifiers."""
    SPREAD = "spread"
    TOTALS = "totals"
    LIVE = "live"
    ARBITRAGE = "arbitrage"
    PLAYER_PROPS = "player_props"
    B2B_REST = "b2b_rest"


@dataclass
class BetSignal:
    """
    Unified bet signal across all strategies.

    All strategies return BetSignal objects with standardized fields,
    allowing the orchestrator to process them uniformly.
    """
    strategy_type: StrategyType
    game_id: str
    home_team: str
    away_team: str
    bet_type: Literal["moneyline", "spread", "totals", "player_prop"]
    bet_side: Literal["HOME", "AWAY", "OVER", "UNDER", "PASS"]
    model_prob: float  # Model's predicted probability (0-1)
    market_prob: float  # Market implied probability (0-1)
    edge: float  # model_prob - market_prob
    line: Optional[float] = None  # Spread/total line
    odds: Optional[int] = None  # American odds
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    bookmaker: Optional[str] = None
    filters_passed: List[str] = field(default_factory=list)

    # Player prop specific fields
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    prop_type: Optional[str] = None  # "PTS", "REB", "AST", "3PM", "STL", "BLK"

    # Arbitrage specific fields
    arb_opportunity: Optional[Dict] = None  # Full arbitrage details

    # Additional context
    pred_diff: Optional[float] = None  # For spread: predicted point diff
    market_spread: Optional[float] = None  # For spread: vegas line

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not PASS and has positive edge)."""
        return self.bet_side != "PASS" and self.edge > 0


class BettingStrategy(ABC):
    """
    Abstract base class for all betting strategies.

    All strategies must implement:
    - evaluate_game(): Evaluate single game/opportunity
    - evaluate_games(): Batch evaluation
    - get_actionable_bets(): Filter to actionable signals

    Strategies share common patterns from EdgeStrategy:
    - Return BetSignal objects
    - Apply filters (B2B, team exclusions, etc.)
    - Assign confidence levels based on edge and filters
    """

    strategy_type: StrategyType
    priority: int = 99  # Lower = higher priority in orchestrator

    def __init__(
        self,
        min_edge: float = 0.05,
        max_daily_bets: int = 10,
        enabled: bool = True,
    ):
        """
        Initialize strategy.

        Args:
            min_edge: Minimum edge to trigger bet (default 5%)
            max_daily_bets: Maximum bets per day for this strategy
            enabled: Whether strategy is active
        """
        self.min_edge = min_edge
        self.max_daily_bets = max_daily_bets
        self.enabled = enabled

    @abstractmethod
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
        Evaluate a single game, returning list of possible signals.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            features: Dict of game features for model
            odds_data: Dict of odds data (structure varies by strategy)
            **kwargs: Strategy-specific additional arguments

        Returns:
            List of BetSignal objects (can be empty, or multiple if both sides have edge)
        """
        pass

    @abstractmethod
    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate multiple games, returning all signals.

        Args:
            games_df: DataFrame with game_id, home_team, away_team, etc.
            features_df: DataFrame with model features
            odds_df: DataFrame with odds data

        Returns:
            List of all BetSignal objects from batch evaluation
        """
        pass

    def get_actionable_bets(
        self,
        signals: List[BetSignal],
        max_bets: int = None,
    ) -> List[BetSignal]:
        """
        Filter and rank actionable signals.

        Args:
            signals: List of BetSignal objects
            max_bets: Optional maximum number to return

        Returns:
            Filtered and sorted list of actionable signals
        """
        # Filter to actionable only
        actionable = [s for s in signals if s.is_actionable]

        # Sort by confidence then edge (descending)
        confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        actionable.sort(
            key=lambda s: (confidence_order.get(s.confidence, 3), -s.edge)
        )

        # Limit if specified
        if max_bets is not None:
            return actionable[:max_bets]

        return actionable

    @property
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self.enabled

    def _determine_confidence(
        self,
        edge: float,
        filters_passed: List[str],
        additional_score: float = 0.0
    ) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """
        Determine confidence level based on edge size and filters.

        Args:
            edge: Absolute edge value
            filters_passed: List of filter names passed
            additional_score: Optional additional score (0-1) to consider

        Returns:
            Confidence level: "HIGH", "MEDIUM", or "LOW"
        """
        # Edge contributes 70%, filters 20%, additional 10%
        edge_score = min(edge / 0.15, 1.0)  # Normalize: 15% edge = 1.0
        filter_score = min(len(filters_passed) / 4, 1.0)  # 4+ filters = 1.0

        combined_score = (
            edge_score * 0.7 +
            filter_score * 0.2 +
            additional_score * 0.1
        )

        if combined_score >= 0.7:
            return "HIGH"
        elif combined_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"


def odds_to_implied_prob(american_odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if american_odds < 0:
        # Favorite: -110 means 110/(110+100) = 52.4%
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        # Underdog: +150 means 100/(150+100) = 40%
        return 100 / (american_odds + 100)


def implied_prob_to_odds(prob: float) -> int:
    """
    Convert implied probability to American odds.

    Args:
        prob: Probability (0-1)

    Returns:
        American odds
    """
    if prob >= 0.5:
        # Favorite
        return int(-prob / (1 - prob) * 100)
    else:
        # Underdog
        return int((1 - prob) / prob * 100)
