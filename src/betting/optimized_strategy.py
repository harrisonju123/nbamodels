"""
Optimized Betting Strategy

Addresses key issues found in backtest:
1. Home team bias (-17.1% ROI on home bets)
2. Confidence miscalibration (high confidence bets underperform)
3. Edge threshold too low (2% insufficient)
4. Alternative data integration (filter for data quality)

Based on backtest analysis from 2026-01-04.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class OptimizedStrategyConfig:
    """Configuration for optimized betting strategy."""

    # Edge thresholds (increased from 2% to 5%)
    min_edge: float = 0.05  # 5% minimum edge (vs 2% before)
    min_edge_home: float = 0.07  # 7% for home bets (address home bias)
    min_edge_high_confidence: float = 0.04  # 4% if very high confidence (>75%)

    # Confidence thresholds
    min_confidence: float = 0.55  # Minimum model confidence
    high_confidence_threshold: float = 0.75  # Consider "very confident"

    # Kelly sizing (reduced from 25% to 10%)
    kelly_fraction: float = 0.10  # 10% of Kelly (vs 25% before)
    max_bet_size: float = 0.05  # Max 5% of bankroll per bet
    min_bet_size: float = 0.01  # Min 1% of bankroll

    # Alternative data filters (NEW)
    require_alt_data: bool = False  # If True, only bet when alt data available
    news_volume_bonus: float = 0.01  # Add 1% edge if high news volume
    sentiment_enabled_bonus: float = 0.005  # Add 0.5% edge if sentiment available

    # Home/Away filters
    home_bias_penalty: float = 0.02  # Subtract 2% from home bets edge
    prefer_away: bool = True  # Prefer away bets when edge is similar

    # Disagreement threshold (tightened from 0.15 to 0.20)
    min_disagreement: float = 0.20  # Model must disagree with market by 20%

    # Bankroll protection
    max_drawdown_stop: float = 0.30  # Stop betting at 30% drawdown
    consecutive_loss_limit: int = 5  # Reduce sizing after 5 losses


class OptimizedBettingStrategy:
    """
    Optimized betting strategy with improved filters and risk management.

    Improvements over baseline:
    - Stricter edge requirements (5% vs 2%)
    - Home bias mitigation (7% edge required for home bets)
    - Lower Kelly fraction (10% vs 25%)
    - Alternative data quality filters
    - Dynamic bet sizing based on recent performance
    - Drawdown protection
    """

    def __init__(self, config: Optional[OptimizedStrategyConfig] = None):
        self.config = config or OptimizedStrategyConfig()
        self.consecutive_losses = 0
        self.peak_bankroll = None
        self.current_bankroll = None

        logger.info("Initialized OptimizedBettingStrategy")
        logger.info(f"  Min edge: {self.config.min_edge:.1%}")
        logger.info(f"  Min edge (home): {self.config.min_edge_home:.1%}")
        logger.info(f"  Kelly fraction: {self.config.kelly_fraction:.1%}")
        logger.info(f"  Min disagreement: {self.config.min_disagreement:.1%}")

    def should_bet(
        self,
        model_prob: float,
        market_prob: float,
        side: str,
        confidence: float = None,
        features: Dict[str, float] = None
    ) -> tuple[bool, str]:
        """
        Determine if we should place a bet.

        Args:
            model_prob: Model's probability for this side
            market_prob: Market's implied probability
            side: 'home' or 'away'
            confidence: Model confidence (optional)
            features: Game features dict (for alt data filtering)

        Returns:
            (should_bet: bool, reason: str)
        """
        features = features or {}
        confidence = confidence or model_prob

        # Calculate raw edge
        edge = model_prob - market_prob

        # Check minimum disagreement
        disagreement = abs(model_prob - market_prob)
        if disagreement < self.config.min_disagreement:
            return False, f"Low disagreement: {disagreement:.3f} < {self.config.min_disagreement}"

        # Apply home bias penalty
        if side.lower() == 'home':
            edge -= self.config.home_bias_penalty

        # Apply alternative data bonuses
        if features:
            # News volume bonus
            home_news = features.get('home_news_volume_24h', 0)
            away_news = features.get('away_news_volume_24h', 0)
            if side.lower() == 'home' and home_news > 10:
                edge += self.config.news_volume_bonus
            elif side.lower() == 'away' and away_news > 10:
                edge += self.config.news_volume_bonus

            # Sentiment bonus
            if features.get('sentiment_enabled', False):
                edge += self.config.sentiment_enabled_bonus

        # Determine minimum edge based on side and confidence
        if side.lower() == 'home':
            min_edge = self.config.min_edge_home
        elif confidence and confidence > self.config.high_confidence_threshold:
            min_edge = self.config.min_edge_high_confidence
        else:
            min_edge = self.config.min_edge

        # Check if edge meets threshold
        if edge < min_edge:
            return False, f"Edge too low: {edge:.3f} < {min_edge:.3f} (side: {side})"

        # Check confidence threshold
        if confidence and confidence < self.config.min_confidence:
            return False, f"Confidence too low: {confidence:.3f}"

        # If prefer_away, require higher edge for home
        if self.config.prefer_away and side.lower() == 'home':
            if edge < self.config.min_edge_home:
                return False, f"Prefer away bets (home edge {edge:.3f} < {self.config.min_edge_home:.3f})"

        # Alternative data filter
        if self.config.require_alt_data and features:
            has_news = features.get('news_volume_diff', 0) != 0
            has_sentiment = features.get('sentiment_enabled', False)
            if not (has_news or has_sentiment):
                return False, "No alternative data available"

        # Drawdown protection
        if self.current_bankroll and self.peak_bankroll:
            drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            if drawdown > self.config.max_drawdown_stop:
                return False, f"Drawdown limit reached: {drawdown:.1%}"

        return True, f"Edge: {edge:.3f}, Confidence: {confidence:.3f}"

    def calculate_bet_size(
        self,
        edge: float,
        odds: float,
        bankroll: float,
        confidence: float = None,
        side: str = None
    ) -> float:
        """
        Calculate optimal bet size using Kelly criterion with adjustments.

        Args:
            edge: Model edge over market
            odds: Decimal odds
            bankroll: Current bankroll
            confidence: Model confidence (optional)
            side: 'home' or 'away' (optional)

        Returns:
            Bet size in dollars
        """
        # Apply home bias adjustment
        adjusted_edge = edge
        if side and side.lower() == 'home':
            adjusted_edge -= self.config.home_bias_penalty

        # Kelly formula: (edge * odds - 1) / (odds - 1)
        b = odds - 1  # Net odds
        kelly = (adjusted_edge * odds - (1 - adjusted_edge)) / b

        # Apply Kelly fraction
        kelly_fraction = self.config.kelly_fraction

        # Reduce sizing after consecutive losses
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            kelly_fraction *= 0.5  # Half sizing after 5 losses
            logger.debug(f"Reduced Kelly to {kelly_fraction:.1%} after {self.consecutive_losses} losses")

        bet_size = kelly * kelly_fraction * bankroll

        # Apply min/max constraints
        min_bet = self.config.min_bet_size * bankroll
        max_bet = self.config.max_bet_size * bankroll

        bet_size = max(min_bet, min(bet_size, max_bet))

        return bet_size

    def update_bankroll(self, new_bankroll: float, won: bool):
        """
        Update bankroll tracking for drawdown protection.

        Args:
            new_bankroll: New bankroll after bet
            won: Whether the bet won
        """
        self.current_bankroll = new_bankroll

        # Track peak for drawdown calculation
        if self.peak_bankroll is None or new_bankroll > self.peak_bankroll:
            self.peak_bankroll = new_bankroll

        # Track consecutive losses
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def get_strategy_summary(self) -> Dict[str, any]:
        """Get summary of current strategy parameters."""
        return {
            'min_edge': self.config.min_edge,
            'min_edge_home': self.config.min_edge_home,
            'kelly_fraction': self.config.kelly_fraction,
            'min_disagreement': self.config.min_disagreement,
            'home_bias_penalty': self.config.home_bias_penalty,
            'prefer_away': self.config.prefer_away,
            'require_alt_data': self.config.require_alt_data,
            'max_drawdown_stop': self.config.max_drawdown_stop,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self._get_current_drawdown(),
        }

    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if not self.peak_bankroll or not self.current_bankroll:
            return 0.0
        return (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll


def filter_bets_with_optimized_strategy(
    predictions_df: pd.DataFrame,
    strategy: OptimizedBettingStrategy
) -> pd.DataFrame:
    """
    Filter predictions using optimized strategy.

    Args:
        predictions_df: DataFrame with columns: model_prob, market_prob, side, etc.
        strategy: OptimizedBettingStrategy instance

    Returns:
        Filtered DataFrame with only valid bets
    """
    valid_bets = []

    for idx, row in predictions_df.iterrows():
        # Extract features if available
        features = {}
        if 'home_news_volume_24h' in row:
            features = {
                'home_news_volume_24h': row.get('home_news_volume_24h', 0),
                'away_news_volume_24h': row.get('away_news_volume_24h', 0),
                'news_volume_diff': row.get('news_volume_diff', 0),
                'sentiment_enabled': row.get('sentiment_enabled', False),
            }

        should_bet, reason = strategy.should_bet(
            model_prob=row['model_prob'],
            market_prob=row['market_prob'],
            side=row['side'],
            confidence=row.get('confidence', row['model_prob']),
            features=features
        )

        if should_bet:
            valid_bets.append(idx)

    logger.info(f"Filtered {len(predictions_df)} predictions → {len(valid_bets)} bets")

    if valid_bets:
        return predictions_df.loc[valid_bets].copy()
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the strategy
    print("Optimized Betting Strategy")
    print("=" * 70)

    # Create strategy with default config
    strategy = OptimizedBettingStrategy()

    print("\nStrategy Configuration:")
    summary = strategy.get_strategy_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}" if value < 1 else f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\nTest Cases:")
    print("-" * 70)

    # Test 1: Good away bet
    should_bet, reason = strategy.should_bet(
        model_prob=0.60,
        market_prob=0.52,
        side='away',
        confidence=0.65
    )
    print(f"Away bet (8% edge, 65% conf): {should_bet} - {reason}")

    # Test 2: Home bet (requires higher edge)
    should_bet, reason = strategy.should_bet(
        model_prob=0.58,
        market_prob=0.52,
        side='home',
        confidence=0.65
    )
    print(f"Home bet (6% edge, 65% conf): {should_bet} - {reason}")

    # Test 3: Low edge
    should_bet, reason = strategy.should_bet(
        model_prob=0.54,
        market_prob=0.52,
        side='away',
        confidence=0.60
    )
    print(f"Away bet (2% edge, 60% conf): {should_bet} - {reason}")

    # Test 4: With alternative data
    should_bet, reason = strategy.should_bet(
        model_prob=0.58,
        market_prob=0.52,
        side='away',
        confidence=0.65,
        features={
            'home_news_volume_24h': 15,
            'away_news_volume_24h': 5,
            'sentiment_enabled': True
        }
    )
    print(f"Away bet with alt data: {should_bet} - {reason}")

    # Test 5: Bet sizing
    print("\n" + "-" * 70)
    print("Bet Sizing Examples:")

    bankroll = 1000

    # Away bet with 8% edge at -110 odds
    bet_size = strategy.calculate_bet_size(
        edge=0.08,
        odds=1.909,  # -110 in decimal
        bankroll=bankroll,
        side='away'
    )
    print(f"Away 8% edge: ${bet_size:.2f} ({bet_size/bankroll:.1%} of bankroll)")

    # Home bet (penalized)
    bet_size = strategy.calculate_bet_size(
        edge=0.08,
        odds=1.909,
        bankroll=bankroll,
        side='home'
    )
    print(f"Home 8% edge: ${bet_size:.2f} ({bet_size/bankroll:.1%} of bankroll)")
