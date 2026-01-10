"""
Kelly Criterion Bet Sizing

Optimal bet sizing for maximizing long-term bankroll growth.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class UncertaintyAdjustedBet:
    """Bet sizing with uncertainty adjustment details."""
    kelly_fraction: float
    adjusted_fraction: float
    bet_size: float
    confidence_factor: float
    uncertainty: float
    should_bet: bool
    reason: str


class KellyBetSizer:
    """
    Kelly Criterion bet sizing with fractional Kelly support.

    The Kelly Criterion maximizes long-term geometric growth of bankroll
    but can be volatile. Fractional Kelly (e.g., 1/4 or 1/5 Kelly)
    reduces variance at the cost of some expected growth.
    """

    def __init__(self, fraction: float = 0.2, max_bet_pct: float = 0.05):
        """
        Initialize Kelly bet sizer.

        Args:
            fraction: Fraction of full Kelly to use (0.2 = 1/5 Kelly)
            max_bet_pct: Maximum bet as percentage of bankroll
        """
        self.fraction = fraction
        self.max_bet_pct = max_bet_pct

    def calculate_kelly(
        self,
        win_prob: float,
        odds: float,
        odds_format: str = "american",
    ) -> float:
        """
        Calculate Kelly fraction for a single bet.

        Args:
            win_prob: Probability of winning (0 to 1)
            odds: Betting odds
            odds_format: "american", "decimal", or "implied"

        Returns:
            Kelly fraction (fraction of bankroll to bet)
        """
        # Convert to decimal odds
        if odds_format == "american":
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
        elif odds_format == "implied":
            # Implied probability to decimal odds
            decimal_odds = 1 / odds
        else:
            decimal_odds = odds

        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = lose prob, b = decimal odds - 1
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p

        if b <= 0:
            return 0

        kelly = (p * b - q) / b

        # Apply fractional Kelly
        kelly = kelly * self.fraction

        # Cap at max bet percentage
        kelly = min(kelly, self.max_bet_pct)

        # No negative bets
        kelly = max(kelly, 0)

        return kelly

    def calculate_bet_size(
        self,
        bankroll: float,
        win_prob: float,
        odds: float,
        odds_format: str = "american",
    ) -> float:
        """
        Calculate actual bet size in dollars.

        Args:
            bankroll: Current bankroll
            win_prob: Probability of winning
            odds: Betting odds
            odds_format: Odds format

        Returns:
            Bet size in dollars
        """
        kelly = self.calculate_kelly(win_prob, odds, odds_format)
        return bankroll * kelly

    def calculate_edge(
        self,
        win_prob: float,
        implied_prob: float,
    ) -> float:
        """
        Calculate betting edge.

        Args:
            win_prob: Model's win probability
            implied_prob: Implied probability from odds

        Returns:
            Edge (positive = favorable)
        """
        return win_prob - implied_prob

    def calculate_ev(
        self,
        win_prob: float,
        odds: float,
        bet_size: float = 1.0,
        odds_format: str = "american",
    ) -> float:
        """
        Calculate expected value of a bet.

        Args:
            win_prob: Probability of winning
            odds: Betting odds
            bet_size: Amount wagered
            odds_format: Odds format

        Returns:
            Expected value
        """
        # Convert to decimal odds
        if odds_format == "american":
            if odds > 0:
                profit = bet_size * (odds / 100)
            else:
                profit = bet_size * (100 / abs(odds))
        else:
            profit = bet_size * (odds - 1)

        ev = (win_prob * profit) - ((1 - win_prob) * bet_size)
        return ev

    def should_bet(
        self,
        win_prob: float,
        odds: float,
        min_edge: float = 0.03,
        odds_format: str = "american",
    ) -> bool:
        """
        Determine if a bet should be placed.

        Args:
            win_prob: Model's win probability
            odds: Betting odds
            min_edge: Minimum required edge
            odds_format: Odds format

        Returns:
            True if bet should be placed
        """
        # Calculate implied probability
        if odds_format == "american":
            if odds > 0:
                implied = 100 / (odds + 100)
            else:
                implied = abs(odds) / (abs(odds) + 100)
        elif odds_format == "decimal":
            implied = 1 / odds
        else:
            implied = odds

        edge = self.calculate_edge(win_prob, implied)

        return edge >= min_edge

    def calculate_uncertainty_adjusted_kelly(
        self,
        win_prob: float,
        prob_uncertainty: float,
        odds: float,
        odds_format: str = "american",
        max_uncertainty: float = 0.15,
        min_confidence: float = 0.3,
    ) -> UncertaintyAdjustedBet:
        """
        Calculate Kelly fraction adjusted for prediction uncertainty.

        When prediction uncertainty is high, we reduce bet size. When
        uncertainty is low (high confidence), we bet closer to full
        fractional Kelly.

        The adjustment uses a confidence factor:
        - High uncertainty → low confidence → reduce bet
        - Low uncertainty → high confidence → normal bet

        Args:
            win_prob: Model's win probability (point estimate)
            prob_uncertainty: Uncertainty in probability (std dev or interval width)
            odds: Betting odds
            odds_format: Odds format
            max_uncertainty: Uncertainty threshold above which we don't bet
            min_confidence: Minimum confidence factor to place any bet

        Returns:
            UncertaintyAdjustedBet with sizing details
        """
        # Calculate base Kelly
        base_kelly = self.calculate_kelly(win_prob, odds, odds_format)

        # If base Kelly is zero or negative, don't bet
        if base_kelly <= 0:
            return UncertaintyAdjustedBet(
                kelly_fraction=0,
                adjusted_fraction=0,
                bet_size=0,
                confidence_factor=0,
                uncertainty=prob_uncertainty,
                should_bet=False,
                reason="No edge (base Kelly <= 0)"
            )

        # Calculate confidence factor (inverse of uncertainty)
        # Confidence = 1 when uncertainty = 0
        # Confidence approaches 0 as uncertainty approaches max_uncertainty
        if prob_uncertainty >= max_uncertainty:
            confidence_factor = 0
        else:
            # Linear decay: confidence = 1 - (uncertainty / max_uncertainty)
            confidence_factor = 1 - (prob_uncertainty / max_uncertainty)

        # Can also use exponential decay for smoother adjustment:
        # confidence_factor = np.exp(-3 * prob_uncertainty / max_uncertainty)

        # Don't bet if confidence is below minimum
        if confidence_factor < min_confidence:
            return UncertaintyAdjustedBet(
                kelly_fraction=base_kelly,
                adjusted_fraction=0,
                bet_size=0,
                confidence_factor=confidence_factor,
                uncertainty=prob_uncertainty,
                should_bet=False,
                reason=f"Confidence too low ({confidence_factor:.2f} < {min_confidence})"
            )

        # Adjust Kelly by confidence factor
        adjusted_kelly = base_kelly * confidence_factor

        # Apply max bet cap
        adjusted_kelly = min(adjusted_kelly, self.max_bet_pct)

        return UncertaintyAdjustedBet(
            kelly_fraction=base_kelly,
            adjusted_fraction=adjusted_kelly,
            bet_size=adjusted_kelly,  # As fraction of bankroll
            confidence_factor=confidence_factor,
            uncertainty=prob_uncertainty,
            should_bet=True,
            reason=f"Adjusted by confidence factor {confidence_factor:.2f}"
        )

    def calculate_uncertainty_adjusted_bet_size(
        self,
        bankroll: float,
        win_prob: float,
        prob_uncertainty: float,
        odds: float,
        odds_format: str = "american",
        **kwargs,
    ) -> Tuple[float, UncertaintyAdjustedBet]:
        """
        Calculate actual bet size in dollars with uncertainty adjustment.

        Args:
            bankroll: Current bankroll
            win_prob: Probability of winning
            prob_uncertainty: Uncertainty in probability
            odds: Betting odds
            odds_format: Odds format
            **kwargs: Additional args for uncertainty_adjusted_kelly

        Returns:
            Tuple of (bet_size_dollars, UncertaintyAdjustedBet)
        """
        result = self.calculate_uncertainty_adjusted_kelly(
            win_prob=win_prob,
            prob_uncertainty=prob_uncertainty,
            odds=odds,
            odds_format=odds_format,
            **kwargs,
        )

        bet_size = bankroll * result.adjusted_fraction
        result.bet_size = bet_size

        return bet_size, result

    def batch_uncertainty_adjusted_bets(
        self,
        bankroll: float,
        win_probs: np.ndarray,
        uncertainties: np.ndarray,
        odds: np.ndarray,
        odds_format: str = "american",
        **kwargs,
    ) -> Tuple[np.ndarray, list]:
        """
        Calculate uncertainty-adjusted bet sizes for multiple bets.

        Args:
            bankroll: Current bankroll
            win_probs: Array of win probabilities
            uncertainties: Array of uncertainty values
            odds: Array of betting odds
            odds_format: Odds format
            **kwargs: Additional args for uncertainty_adjusted_kelly

        Returns:
            Tuple of (bet_sizes_array, list_of_UncertaintyAdjustedBet)
        """
        bet_sizes = []
        results = []

        for prob, unc, odd in zip(win_probs, uncertainties, odds):
            bet_size, result = self.calculate_uncertainty_adjusted_bet_size(
                bankroll=bankroll,
                win_prob=prob,
                prob_uncertainty=unc,
                odds=odd,
                odds_format=odds_format,
                **kwargs,
            )
            bet_sizes.append(bet_size)
            results.append(result)

        return np.array(bet_sizes), results


def simulate_kelly_growth(
    results: np.ndarray,
    odds: np.ndarray,
    win_probs: np.ndarray,
    initial_bankroll: float = 1000,
    kelly_fraction: float = 0.2,
) -> dict:
    """
    Simulate bankroll growth using Kelly betting.

    Args:
        results: Array of bet outcomes (1 = win, 0 = loss)
        odds: Array of American odds for each bet
        win_probs: Array of model win probabilities
        initial_bankroll: Starting bankroll
        kelly_fraction: Fraction of Kelly to use

    Returns:
        Dictionary with simulation results
    """
    sizer = KellyBetSizer(fraction=kelly_fraction)
    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    total_wagered = 0
    total_profit = 0

    for result, odd, prob in zip(results, odds, win_probs):
        # Calculate bet size
        bet_size = sizer.calculate_bet_size(bankroll, prob, odd)

        if bet_size <= 0:
            bankroll_history.append(bankroll)
            continue

        total_wagered += bet_size

        # Calculate payout
        if odd > 0:
            profit = bet_size * (odd / 100)
        else:
            profit = bet_size * (100 / abs(odd))

        if result == 1:
            bankroll += profit
            total_profit += profit
        else:
            bankroll -= bet_size
            total_profit -= bet_size

        bankroll_history.append(bankroll)

    return {
        "final_bankroll": bankroll,
        "total_wagered": total_wagered,
        "total_profit": total_profit,
        "roi": total_profit / total_wagered if total_wagered > 0 else 0,
        "max_drawdown": min(bankroll_history) / initial_bankroll - 1,
        "bankroll_history": bankroll_history,
    }
