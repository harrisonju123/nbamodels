"""
Market Analysis Module

Comprehensive market analysis tools for NBA betting:
- Model drift and calibration monitoring
- Backtest vs live performance gap analysis
- Market efficiency and sharp vs public analysis
- Optimal bet timing analysis and guidance
"""

from .model_drift import (
    ModelDriftMonitor,
    DriftAlert,
    CalibrationSnapshot,
)

from .performance_gap import (
    BacktestLiveGapAnalyzer,
    PerformanceGapResult,
    GapMetrics,
)

from .market_efficiency import (
    MarketEfficiencyAnalyzer,
    TimeToEfficiencyTracker,
    SharpPublicDivergence,
    EdgeDecayPattern,
    SharpSignalPerformance,
)

from .timing_analysis import (
    TimingWindowsAnalyzer,
    HistoricalTimingAnalyzer,
    TimingWindow,
)

from .bet_timing_advisor import (
    BetTimingAdvisor,
    TimingRecommendation,
)

__all__ = [
    # Model drift
    "ModelDriftMonitor",
    "DriftAlert",
    "CalibrationSnapshot",
    # Performance gap
    "BacktestLiveGapAnalyzer",
    "PerformanceGapResult",
    "GapMetrics",
    # Market efficiency
    "MarketEfficiencyAnalyzer",
    "TimeToEfficiencyTracker",
    "SharpPublicDivergence",
    "EdgeDecayPattern",
    "SharpSignalPerformance",
    # Timing analysis
    "TimingWindowsAnalyzer",
    "HistoricalTimingAnalyzer",
    "TimingWindow",
    # Bet timing advisor
    "BetTimingAdvisor",
    "TimingRecommendation",
]
