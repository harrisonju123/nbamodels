"""
Betting strategies module.

Unified interface for all betting strategies (spread, totals, live, arbitrage, props).
"""

from src.betting.strategies.base import (
    BettingStrategy,
    BetSignal,
    StrategyType,
    odds_to_implied_prob,
    implied_prob_to_odds,
)
from src.betting.strategies.totals_strategy import TotalsStrategy
from src.betting.strategies.live_strategy import LiveBettingStrategy
from src.betting.strategies.arbitrage_strategy import ArbitrageStrategy
from src.betting.strategies.player_props_strategy import PlayerPropsStrategy

__all__ = [
    # Base classes
    "BettingStrategy",
    "BetSignal",
    "StrategyType",
    # Utility functions
    "odds_to_implied_prob",
    "implied_prob_to_odds",
    # Strategies
    "TotalsStrategy",
    "LiveBettingStrategy",
    "ArbitrageStrategy",
    "PlayerPropsStrategy",
]
