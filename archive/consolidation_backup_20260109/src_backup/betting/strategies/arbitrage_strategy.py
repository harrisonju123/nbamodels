"""
Arbitrage Strategy

Scans multiple bookmakers for price discrepancies that create
risk-free profit opportunities (arbitrage).
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.betting.strategies.base import (
    BettingStrategy,
    BetSignal,
    StrategyType,
    odds_to_implied_prob,
)


class ArbitrageStrategy(BettingStrategy):
    """
    Arbitrage betting strategy.

    Scans multiple bookmakers to find price discrepancies where
    betting both sides guarantees profit regardless of outcome.

    Arbitrage exists when: 1/odds1 + 1/odds2 < 1
    (i.e., combined implied probabilities < 100%)

    Example:
        Book A: Lakers -110, Warriors +120
        Book B: Lakers +105, Warriors -105

        If we bet Lakers at +105 (Book B) and Warriors at +120 (Book A),
        we might have an arbitrage if total implied prob < 100%.
    """

    strategy_type = StrategyType.ARBITRAGE
    priority = 3  # Medium priority - requires multiple bookmakers

    def __init__(
        self,
        min_arb_profit: float = 0.01,  # 1% minimum profit
        bookmakers: Optional[List[str]] = None,
        max_daily_bets: int = 5,
        enabled: bool = True,
    ):
        """
        Initialize arbitrage strategy.

        Args:
            min_arb_profit: Minimum profit percentage (default 1%)
            bookmakers: List of bookmaker names to scan
            max_daily_bets: Maximum bets per day
            enabled: Whether strategy is active
        """
        super().__init__(
            min_edge=min_arb_profit,
            max_daily_bets=max_daily_bets,
            enabled=enabled,
        )

        self.min_arb_profit = min_arb_profit
        self.bookmakers = bookmakers or [
            "draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "betrivers"
        ]

        logger.info(
            f"ArbitrageStrategy initialized (min_profit={min_arb_profit:.1%}, "
            f"bookmakers={len(self.bookmakers)})"
        )

    def find_arbitrage(
        self,
        odds_df: pd.DataFrame,
        market_type: str = 'spread',
    ) -> List[Dict]:
        """
        Find arbitrage opportunities in odds data.

        Args:
            odds_df: DataFrame with columns:
                - game_id
                - market (spread, moneyline, total)
                - team/side (home, away, over, under)
                - odds (American odds)
                - bookmaker
            market_type: Which market to scan ('spread', 'moneyline', 'total')

        Returns:
            List of arbitrage opportunities with:
                - game_id
                - market
                - profit_pct
                - side1, side2 (the two sides)
                - odds1, odds2 (best odds for each side)
                - bookmaker1, bookmaker2
                - stake_pct1, stake_pct2 (how to split bankroll)
        """
        arbs = []

        # Filter to specified market
        market_odds = odds_df[odds_df['market'] == market_type].copy()

        if len(market_odds) == 0:
            return arbs

        for game_id in market_odds['game_id'].unique():
            game_odds = market_odds[market_odds['game_id'] == game_id]

            # Define sides based on market type
            if market_type == 'spread' or market_type == 'moneyline':
                sides = [('home', 'away')]
            elif market_type == 'total':
                sides = [('over', 'under')]
            else:
                continue

            for side1_name, side2_name in sides:
                # Find best odds for each side across all bookmakers
                side1_odds = game_odds[game_odds['team'] == side1_name]
                side2_odds = game_odds[game_odds['team'] == side2_name]

                if len(side1_odds) == 0 or len(side2_odds) == 0:
                    continue

                # Get best (highest) odds for each side
                best_side1 = side1_odds.loc[side1_odds['odds'].idxmax()]
                best_side2 = side2_odds.loc[side2_odds['odds'].idxmax()]

                odds1 = best_side1['odds']
                odds2 = best_side2['odds']

                # Calculate implied probabilities
                prob1 = odds_to_implied_prob(int(odds1))
                prob2 = odds_to_implied_prob(int(odds2))

                total_prob = prob1 + prob2

                # Arbitrage exists when total_prob < 1.0
                if total_prob < 1.0:
                    profit_pct = (1 - total_prob) / total_prob

                    if profit_pct >= self.min_arb_profit:
                        # Calculate optimal stake percentages
                        # To guarantee same profit on both outcomes:
                        # stake1 / stake2 = prob2 / prob1
                        stake_pct1 = prob2 / (prob1 + prob2)
                        stake_pct2 = prob1 / (prob1 + prob2)

                        arbs.append({
                            'game_id': game_id,
                            'market': market_type,
                            'profit_pct': profit_pct,
                            'side1': side1_name,
                            'side2': side2_name,
                            'odds1': int(odds1),
                            'odds2': int(odds2),
                            'bookmaker1': best_side1['bookmaker'],
                            'bookmaker2': best_side2['bookmaker'],
                            'stake_pct1': stake_pct1,
                            'stake_pct2': stake_pct2,
                            'prob1': prob1,
                            'prob2': prob2,
                            'total_prob': total_prob,
                        })

                        logger.info(
                            f"Arbitrage found: {game_id} {market_type} - "
                            f"profit={profit_pct:.2%} ({best_side1['bookmaker']} vs {best_side2['bookmaker']})"
                        )

        return arbs

    def evaluate_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        features: Dict,
        odds_data: Dict,
        **kwargs
    ) -> List[BetSignal]:
        """
        Not typically used - arbitrage works on odds_df across multiple bookmakers.

        Use evaluate_games() instead.
        """
        logger.warning("ArbitrageStrategy.evaluate_game() called - use evaluate_games() instead")
        return []

    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate multiple games for arbitrage opportunities.

        Args:
            games_df: DataFrame with game_id, home_team, away_team
            features_df: Not used for arbitrage (no model)
            odds_df: DataFrame with columns:
                - game_id
                - market
                - team (home, away, over, under)
                - odds (American odds)
                - bookmaker

        Returns:
            List of BetSignal objects (2 signals per arbitrage: both sides)
        """
        all_signals = []

        # Find arbitrage opportunities for each market type
        for market_type in ['spread', 'moneyline', 'total']:
            arbs = self.find_arbitrage(odds_df, market_type)

            # Convert each arbitrage to pair of BetSignals
            for arb in arbs:
                # Get game info
                game_info = games_df[games_df['game_id'] == arb['game_id']]

                if len(game_info) == 0:
                    logger.warning(f"Game info not found for {arb['game_id']}")
                    continue

                game_info = game_info.iloc[0]

                # Signal for side 1
                signal1 = BetSignal(
                    strategy_type=StrategyType.ARBITRAGE,
                    game_id=arb['game_id'],
                    home_team=game_info['home_team'],
                    away_team=game_info['away_team'],
                    bet_type=arb['market'],
                    bet_side=arb['side1'].upper(),
                    model_prob=arb['prob1'],  # For arb, this is implied prob
                    market_prob=arb['prob1'],
                    edge=arb['profit_pct'],  # Arbitrage profit
                    odds=arb['odds1'],
                    confidence="HIGH",  # Arbitrage is risk-free
                    bookmaker=arb['bookmaker1'],
                    filters_passed=["arbitrage"],
                    arb_opportunity={
                        'type': 'arbitrage',
                        'profit_pct': arb['profit_pct'],
                        'stake_pct': arb['stake_pct1'],
                        'paired_side': arb['side2'],
                        'total_prob': arb['total_prob'],
                    },
                )

                # Signal for side 2
                signal2 = BetSignal(
                    strategy_type=StrategyType.ARBITRAGE,
                    game_id=arb['game_id'],
                    home_team=game_info['home_team'],
                    away_team=game_info['away_team'],
                    bet_type=arb['market'],
                    bet_side=arb['side2'].upper(),
                    model_prob=arb['prob2'],
                    market_prob=arb['prob2'],
                    edge=arb['profit_pct'],
                    odds=arb['odds2'],
                    confidence="HIGH",
                    bookmaker=arb['bookmaker2'],
                    filters_passed=["arbitrage"],
                    arb_opportunity={
                        'type': 'arbitrage',
                        'profit_pct': arb['profit_pct'],
                        'stake_pct': arb['stake_pct2'],
                        'paired_side': arb['side1'],
                        'total_prob': arb['total_prob'],
                    },
                )

                all_signals.extend([signal1, signal2])

        logger.info(
            f"ArbitrageStrategy found {len(all_signals)//2} arbitrage opportunities "
            f"({len(all_signals)} total signals)"
        )

        return all_signals

    @classmethod
    def conservative_strategy(cls) -> 'ArbitrageStrategy':
        """
        Create conservative arbitrage strategy.

        Higher profit threshold to ensure meaningful opportunities.
        """
        return cls(
            min_arb_profit=0.02,  # 2% minimum profit
        )

    @classmethod
    def aggressive_strategy(cls) -> 'ArbitrageStrategy':
        """
        Create aggressive arbitrage strategy.

        Lower profit threshold to capture more opportunities.
        """
        return cls(
            min_arb_profit=0.005,  # 0.5% minimum profit
        )
