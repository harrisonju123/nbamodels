"""
Strategy Orchestrator

Coordinates multiple betting strategies with unified bankroll management.
Manages per-strategy allocations, daily limits, and risk controls.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import pandas as pd

from src.betting.strategies.base import BettingStrategy, BetSignal, StrategyType
from src.betting.kelly import KellyBetSizer
from src.risk.exposure_manager import ExposureManager
from src.risk.config import RiskConfig


@dataclass
class OrchestratorConfig:
    """Configuration for strategy orchestrator."""

    bankroll: float = 1000.0
    """Total bankroll for all strategies"""

    kelly_fraction: float = 0.2
    """Fractional Kelly (default 20% of full Kelly)"""

    max_daily_exposure: float = 0.15
    """Maximum total daily wagered (15% of bankroll)"""

    max_pending_exposure: float = 0.25
    """Maximum total in unsettled bets (25% of bankroll)"""

    max_bets_per_strategy: Dict[StrategyType, int] = field(default_factory=lambda: {
        StrategyType.SPREAD: 10,
        StrategyType.ARBITRAGE: 5,
        StrategyType.PLAYER_PROPS: 8,
        StrategyType.B2B_REST: 5,
        # Removed: TOTALS (unprofitable), LIVE (untested)
    })
    """Maximum daily bets per strategy type (consolidated)"""

    strategy_allocation: Dict[StrategyType, float] = field(default_factory=lambda: {
        StrategyType.SPREAD: 0.35,        # 35% of bankroll
        StrategyType.ARBITRAGE: 0.30,     # 30%
        StrategyType.PLAYER_PROPS: 0.20,  # 20%
        StrategyType.B2B_REST: 0.15,      # 15%
        # Removed: TOTALS (unprofitable), LIVE (untested)
    })
    """Per-strategy bankroll allocation (as fraction of total) - consolidated"""

    min_bet_size: float = 1.0
    """Minimum bet size in dollars"""

    max_bet_pct: float = 0.05
    """Maximum single bet as fraction of bankroll (5%)"""


class StrategyOrchestrator:
    """
    Coordinates multiple betting strategies with unified bankroll management.

    Responsibilities:
    - Execute strategies in priority order
    - Manage combined bankroll across strategies
    - Apply risk limits at portfolio level
    - Track performance by strategy
    - Size bets using Kelly criterion

    Example:
        ```python
        config = OrchestratorConfig(bankroll=1000)
        orchestrator = StrategyOrchestrator([
            TotalsStrategy(),
            LiveBettingStrategy(),
            ArbitrageStrategy(),
        ], config)

        # Run all strategies
        signals = orchestrator.run_all_strategies(games_df, features_df, odds_df)

        # Size and filter bets
        recommendations = orchestrator.size_and_filter_bets(signals)
        ```
    """

    def __init__(
        self,
        strategies: List[BettingStrategy],
        config: OrchestratorConfig = None,
        risk_config: RiskConfig = None,
    ):
        """
        Initialize orchestrator.

        Args:
            strategies: List of betting strategies to coordinate
            config: Orchestrator configuration
            risk_config: Risk management configuration
        """
        self.config = config or OrchestratorConfig()
        self.risk_config = risk_config or RiskConfig()

        # Sort strategies by priority (lower number = higher priority)
        self.strategies = sorted(strategies, key=lambda s: s.priority)

        # Initialize bet sizing and risk management
        self.kelly_sizer = KellyBetSizer(
            fraction=self.config.kelly_fraction,
            max_bet_pct=self.config.max_bet_pct
        )
        self.exposure_manager = ExposureManager(self.risk_config)

        # Track daily usage per strategy
        self._daily_reset_date = datetime.now().strftime('%Y-%m-%d')
        self._daily_bets: Dict[StrategyType, int] = {}
        self._daily_exposure: Dict[StrategyType, float] = {}

        logger.info(
            f"StrategyOrchestrator initialized with {len(self.strategies)} strategies, "
            f"bankroll=${self.config.bankroll:.2f}"
        )

    def run_all_strategies(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        live_games: Optional[Dict] = None,
    ) -> List[BetSignal]:
        """
        Execute all strategies in priority order.

        Args:
            games_df: DataFrame with game_id, home_team, away_team
            features_df: DataFrame with model features
            odds_df: DataFrame with odds data
            live_games: Optional dict of live game states (for live strategy)

        Returns:
            Combined list of actionable bet signals from all strategies
        """
        self._check_daily_reset()

        all_signals: List[BetSignal] = []

        for strategy in self.strategies:
            if not strategy.is_enabled:
                logger.debug(f"Skipping {strategy.strategy_type}: disabled")
                continue

            # Check allocation remaining for this strategy
            allocation_pct = self.config.strategy_allocation.get(
                strategy.strategy_type, 0.10
            )
            remaining_allocation = self._get_remaining_allocation(
                strategy.strategy_type, allocation_pct
            )

            if remaining_allocation <= 0:
                logger.debug(
                    f"Skipping {strategy.strategy_type}: daily allocation exhausted"
                )
                continue

            # Run strategy
            try:
                # Live strategies get special handling
                if strategy.strategy_type == StrategyType.LIVE and live_games:
                    if hasattr(strategy, 'evaluate_live_games'):
                        signals = strategy.evaluate_live_games(live_games, odds_df)
                    else:
                        logger.warning(
                            f"{strategy.strategy_type} does not implement evaluate_live_games"
                        )
                        continue
                else:
                    signals = strategy.evaluate_games(games_df, features_df, odds_df)

                # Filter to actionable
                max_bets = self.config.max_bets_per_strategy.get(
                    strategy.strategy_type, 5
                )
                actionable = strategy.get_actionable_bets(signals, max_bets)

                all_signals.extend(actionable)
                logger.info(
                    f"{strategy.strategy_type}: {len(actionable)} actionable signals "
                    f"({len(signals)} total evaluated)"
                )

            except Exception as e:
                logger.error(f"Error in {strategy.strategy_type}: {e}", exc_info=True)
                continue

        logger.info(f"Total actionable signals from all strategies: {len(all_signals)}")
        return all_signals

    def size_and_filter_bets(
        self,
        signals: List[BetSignal],
        bankroll: Optional[float] = None,
    ) -> List[Dict]:
        """
        Apply Kelly sizing and risk limits to signals.

        Args:
            signals: List of BetSignal objects
            bankroll: Optional bankroll override (defaults to config.bankroll)

        Returns:
            List of sized bet recommendations with format:
            {
                'signal': BetSignal,
                'bet_size': float,
                'kelly_fraction': float,
                'limiting_factors': List[str],
            }
        """
        if bankroll is None:
            bankroll = self.config.bankroll

        recommendations = []

        for signal in signals:
            # Check daily limits for strategy
            if not self._check_daily_limits(signal.strategy_type):
                logger.debug(
                    f"Skipping {signal.strategy_type} bet: daily limit reached"
                )
                continue

            # Calculate Kelly size
            bet_size = self.kelly_sizer.calculate_bet_size(
                bankroll=bankroll,
                win_prob=signal.model_prob,
                odds=signal.odds or -110,  # Default to -110 if no odds specified
            )

            # Apply exposure limits via exposure manager
            daily_wagered = self.exposure_manager.get_daily_wagered()
            pending_exposure = self.exposure_manager.get_pending_exposure()

            limiting_factors = []

            # Check daily exposure limit
            max_daily = bankroll * self.config.max_daily_exposure
            if daily_wagered + bet_size > max_daily:
                capped = max_daily - daily_wagered
                if capped > 0:
                    bet_size = capped
                    limiting_factors.append("daily_exposure_limit")
                else:
                    logger.debug("Skipping bet: daily exposure limit reached")
                    continue

            # Check pending exposure limit
            max_pending = bankroll * self.config.max_pending_exposure
            if pending_exposure + bet_size > max_pending:
                capped = max_pending - pending_exposure
                if capped > 0:
                    bet_size = capped
                    limiting_factors.append("pending_exposure_limit")
                else:
                    logger.debug("Skipping bet: pending exposure limit reached")
                    continue

            # Apply min bet size
            if bet_size < self.config.min_bet_size:
                logger.debug(
                    f"Skipping bet: size ${bet_size:.2f} below minimum "
                    f"${self.config.min_bet_size:.2f}"
                )
                continue

            # Round to 2 decimal places
            bet_size = round(bet_size, 2)

            recommendations.append({
                'signal': signal,
                'bet_size': bet_size,
                'kelly_fraction': bet_size / bankroll,
                'limiting_factors': limiting_factors,
            })

            # Update tracking
            self._daily_bets[signal.strategy_type] = \
                self._daily_bets.get(signal.strategy_type, 0) + 1
            self._daily_exposure[signal.strategy_type] = \
                self._daily_exposure.get(signal.strategy_type, 0) + bet_size

        logger.info(
            f"Sized {len(recommendations)} bets "
            f"(filtered {len(signals) - len(recommendations)} due to limits)"
        )

        return recommendations

    def _get_remaining_allocation(
        self,
        strategy_type: StrategyType,
        allocation_pct: float,
    ) -> float:
        """
        Calculate remaining bankroll allocation for strategy.

        Args:
            strategy_type: Strategy type
            allocation_pct: Allocated fraction (0-1)

        Returns:
            Remaining allocation in dollars
        """
        max_allocation = self.config.bankroll * allocation_pct
        used = self._daily_exposure.get(strategy_type, 0)
        return max_allocation - used

    def _check_daily_limits(self, strategy_type: StrategyType) -> bool:
        """
        Check if strategy has remaining daily bet capacity.

        Args:
            strategy_type: Strategy type

        Returns:
            True if strategy can place more bets today
        """
        max_bets = self.config.max_bets_per_strategy.get(strategy_type, 5)
        current = self._daily_bets.get(strategy_type, 0)
        return current < max_bets

    def _check_daily_reset(self):
        """Reset daily tracking if date has changed."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self._daily_reset_date:
            logger.info(f"Daily reset: {self._daily_reset_date} -> {today}")
            self._daily_reset_date = today
            self._daily_bets = {}
            self._daily_exposure = {}

    def reset_daily_tracking(self):
        """Manually reset daily tracking (useful for testing)."""
        self._daily_bets = {}
        self._daily_exposure = {}
        logger.info("Daily tracking manually reset")

    def get_daily_stats(self) -> Dict:
        """
        Get daily statistics for all strategies.

        Returns:
            Dict with per-strategy stats
        """
        stats = {
            'date': self._daily_reset_date,
            'total_bets': sum(self._daily_bets.values()),
            'total_exposure': sum(self._daily_exposure.values()),
            'by_strategy': {}
        }

        for strategy_type in StrategyType:
            allocation = self.config.strategy_allocation.get(strategy_type, 0)
            max_allocation = self.config.bankroll * allocation

            stats['by_strategy'][strategy_type.value] = {
                'bets': self._daily_bets.get(strategy_type, 0),
                'exposure': self._daily_exposure.get(strategy_type, 0),
                'max_allocation': max_allocation,
                'remaining_allocation': max_allocation - self._daily_exposure.get(strategy_type, 0),
                'utilization': self._daily_exposure.get(strategy_type, 0) / max_allocation if max_allocation > 0 else 0,
            }

        return stats
