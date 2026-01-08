"""Betting and backtesting utilities."""

from .kelly import KellyBetSizer, UncertaintyAdjustedBet
from .dual_model_backtest import (
    DualModelATSBacktester,
    DualModelBacktestResult,
    ATSBet,
    run_dual_model_ats_backtest,
)
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
    "DualModelATSBacktester",
    "DualModelBacktestResult",
    "ATSBet",
    "run_dual_model_ats_backtest",
    "EdgeStrategy",
    "BetSignal",
    "calculate_expected_value",
    "calculate_roi",
    "STRATEGY_PERFORMANCE",
    "TEAMS_TO_EXCLUDE",
    "TEAMS_TO_FADE",
]
