"""Betting and backtesting utilities."""

from .backtest import KellyBacktester, BacktestResult
from .kelly import KellyBetSizer, UncertaintyAdjustedBet
from .realistic_backtest import (
    RealisticKellyBacktester,
    RealisticBacktestResult,
    MarketSimulator,
    run_realistic_backtest,
)
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
    "KellyBacktester",
    "BacktestResult",
    "KellyBetSizer",
    "UncertaintyAdjustedBet",
    "RealisticKellyBacktester",
    "RealisticBacktestResult",
    "MarketSimulator",
    "run_realistic_backtest",
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
