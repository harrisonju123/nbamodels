"""Betting and backtesting utilities."""

from .kelly import KellyBetSizer, UncertaintyAdjustedBet
# Archived: dual_model_backtest (dual_model archived)
from .edge_strategy import (
    EdgeStrategy,
    BetSignal,
    calculate_expected_value,
    calculate_roi,
    STRATEGY_PERFORMANCE,
    TEAMS_TO_EXCLUDE,
    TEAMS_TO_FADE,
)

__all__ = [
    "KellyBetSizer",
    "UncertaintyAdjustedBet",
    # Archived: DualModelATSBacktester, DualModelBacktestResult, etc.
    "EdgeStrategy",
    "BetSignal",
    "calculate_expected_value",
    "calculate_roi",
    "STRATEGY_PERFORMANCE",
    "TEAMS_TO_EXCLUDE",
    "TEAMS_TO_FADE",
]
