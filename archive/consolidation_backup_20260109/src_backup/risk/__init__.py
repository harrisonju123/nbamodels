"""
Advanced Risk Management Module

Provides comprehensive risk management for NBA betting:
- Correlation-aware position sizing
- Drawdown-based bet scaling
- Daily/weekly exposure limits
- Multi-dimensional risk attribution

Usage:
    from src.risk import RiskConfig, CorrelationAwarePositionSizer
    from src.betting.kelly import KellyBetSizer

    # Initialize
    config = RiskConfig()
    kelly_sizer = KellyBetSizer()
    position_sizer = CorrelationAwarePositionSizer.create(config, kelly_sizer)

    # Evaluate bet
    result = position_sizer.evaluate_bet(
        bankroll=1000,
        win_prob=0.55,
        odds=-110,
        game_id="0022400123",
        home_team="LAL",
        away_team="BOS",
        bet_side="HOME"
    )

    print(f"Recommended size: ${result.final_size:.2f}")
"""

from .config import RiskConfig
from .models import (
    BetCorrelationContext,
    TeamCorrelation,
    ExposureSnapshot,
    PositionSizeResult,
    BetEvaluation,
    RiskStatus,
    EdgeBucket,
    BetSide,
    NBA_CONFERENCES,
    TEAM_LOOKUP,
    get_team_conference,
    get_team_division,
)
from .correlation_tracker import CorrelationTracker
from .drawdown_manager import DrawdownScaler
from .exposure_manager import ExposureManager
from .position_sizer import CorrelationAwarePositionSizer
from .risk_attribution import RiskAttributionEngine
from .database import RiskDatabaseManager


__all__ = [
    # Config
    'RiskConfig',

    # Models
    'BetCorrelationContext',
    'TeamCorrelation',
    'ExposureSnapshot',
    'PositionSizeResult',
    'BetEvaluation',
    'RiskStatus',
    'EdgeBucket',
    'BetSide',
    'NBA_CONFERENCES',
    'TEAM_LOOKUP',
    'get_team_conference',
    'get_team_division',

    # Core components
    'CorrelationTracker',
    'DrawdownScaler',
    'ExposureManager',
    'CorrelationAwarePositionSizer',
    'RiskAttributionEngine',

    # Database
    'RiskDatabaseManager',
]


__version__ = '1.0.0'
