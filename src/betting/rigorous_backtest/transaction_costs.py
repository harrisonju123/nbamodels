"""
Transaction cost modeling for realistic betting simulation.

Models:
1. Vig: Sportsbook juice (4-5%)
2. Slippage: Line moves before bet placement
3. Execution: Not all bets get placed at desired odds
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .core import BacktestConfig, RigorousBet


@dataclass
class TransactionCosts:
    """Transaction costs for a single bet."""

    vig_cost: float  # Cost from vig
    slippage_cost: float  # Cost from line movement
    execution_failed: bool  # Bet not placed
    effective_odds: float  # Odds after slippage


class TransactionCostModel:
    """
    Models realistic transaction costs in betting.

    Components:
    1. Vig: Standard sportsbook juice (4-5%)
    2. Slippage: Line moves before bet placement
    3. Execution: Not all bets get placed at desired odds
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Slippage parameters (calibrated from market data)
        self.slippage_params = {
            "fixed": {"points": 0.5},
            "linear": {"base": 0.3, "rate": 0.002},  # rate per $100 bet
            "sqrt": {"base": 0.3, "rate": 0.05},
        }

    def calculate_vig(self, odds: float, vig_rate: float = None) -> float:
        """
        Calculate vig cost embedded in odds.

        For American odds -110/-110:
        - True probability = 0.5
        - Implied probability = 0.5238 (10/19)
        - Vig = 0.0238 per side = 4.76% total
        """
        vig_rate = vig_rate or self.config.base_vig
        return vig_rate / 2  # Split between both sides

    def calculate_slippage(
        self, bet_size: float, market_liquidity: float = 1.0
    ) -> float:
        """
        Calculate expected line slippage in points.

        Slippage increases with:
        - Larger bet sizes
        - Thinner markets
        - Sharp money detection

        Args:
            bet_size: Size of bet in dollars
            market_liquidity: 1.0 = normal, <1 = thin

        Returns:
            Expected slippage in points
        """
        model = self.config.slippage_model
        params = self.slippage_params[model]

        if model == "fixed":
            slippage = params["points"]

        elif model == "linear":
            slippage = params["base"] + params["rate"] * (bet_size / 100)

        elif model == "sqrt":
            slippage = params["base"] + params["rate"] * np.sqrt(bet_size / 100)

        else:
            slippage = 0.5  # Default

        # Adjust for market liquidity
        slippage *= 1 / market_liquidity

        # Add randomness (slippage is not deterministic)
        noise = self.rng.normal(0, slippage * 0.3)
        slippage = max(0, slippage + noise)

        return slippage

    def calculate_execution_probability(
        self, bet_size: float, time_to_game: float = 24.0
    ) -> float:
        """
        Calculate probability bet gets executed at target odds.

        Lower probability when:
        - Larger bets (limited by book)
        - Closer to game time (lines more volatile)

        Args:
            bet_size: Size of bet
            time_to_game: Hours before game

        Returns:
            Probability (0-1)
        """
        base_prob = self.config.execution_probability

        # Size penalty: larger bets less likely to execute
        size_penalty = np.clip(bet_size / 1000, 0, 0.3)

        # Time penalty: closer to game = harder to execute
        time_penalty = np.clip(1 / (time_to_game + 1), 0, 0.2)

        execution_prob = base_prob - size_penalty - time_penalty
        return np.clip(execution_prob, 0.5, 1.0)

    def apply_transaction_costs(
        self,
        bet: RigorousBet,
        market_liquidity: float = 1.0,
        time_to_game: float = 24.0,
    ) -> Tuple[RigorousBet, TransactionCosts]:
        """
        Apply all transaction costs to a bet.

        Args:
            bet: Original bet
            market_liquidity: Market liquidity factor
            time_to_game: Hours before game

        Returns:
            Tuple of (adjusted_bet, cost_breakdown)
        """
        # Check execution
        exec_prob = self.calculate_execution_probability(bet.bet_size, time_to_game)

        if self.rng.random() > exec_prob:
            # Bet didn't execute
            bet.execution_failed = True
            bet.execution_prob = exec_prob
            return bet, TransactionCosts(
                vig_cost=0,
                slippage_cost=0,
                execution_failed=True,
                effective_odds=bet.odds,
            )

        # Calculate costs
        vig_cost = self.calculate_vig(bet.odds)
        slippage_points = self.calculate_slippage(bet.bet_size, market_liquidity)

        # Convert slippage to cost (approx 0.5 pts = 2% prob = 4% EV)
        slippage_cost = slippage_points * 0.04

        # Adjust effective odds for slippage
        effective_odds = self._adjust_odds_for_slippage(bet.odds, slippage_points)

        # Update bet
        bet.odds = effective_odds
        bet.vig_cost = vig_cost
        bet.slippage = slippage_points
        bet.execution_prob = exec_prob

        return bet, TransactionCosts(
            vig_cost=vig_cost,
            slippage_cost=slippage_cost,
            execution_failed=False,
            effective_odds=effective_odds,
        )

    def _adjust_odds_for_slippage(self, odds: float, slippage_points: float) -> float:
        """
        Adjust decimal odds for slippage.

        Slippage means we get worse odds than expected.
        0.5 points of slippage is roughly 0.02-0.03 in decimal odds.
        """
        # Approximate adjustment (0.5 pts ~= -0.025 decimal odds)
        odds_adjustment = slippage_points * 0.05

        # Worse odds for bettor
        return max(1.01, odds - odds_adjustment)
