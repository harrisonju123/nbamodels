"""
Live Betting Strategy

Wraps the existing LiveEdgeDetector to provide real-time betting opportunities
during games.
"""

from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

from src.betting.strategies.base import (
    BettingStrategy,
    BetSignal,
    StrategyType,
)
from src.betting.live_edge_detector import LiveEdgeDetector


class LiveBettingStrategy(BettingStrategy):
    """
    Live betting strategy using existing LiveEdgeDetector.

    Monitors games in progress and identifies profitable opportunities
    by comparing live model probabilities vs live market odds.

    Filters:
    - Minimum edge threshold
    - Quarter limit (avoid overtime)
    - Minimum model confidence
    - Time remaining filters
    """

    strategy_type = StrategyType.LIVE
    priority = 2  # Medium priority - live opportunities are time-sensitive

    def __init__(
        self,
        min_edge: float = 0.05,
        min_confidence: float = 0.5,
        max_quarter: int = 4,
        min_time_remaining: Optional[float] = None,  # Minutes
        max_daily_bets: int = 3,
        enabled: bool = True,
    ):
        """
        Initialize live betting strategy.

        Args:
            min_edge: Minimum edge to trigger bet (default 5%)
            min_confidence: Minimum model confidence (default 50%)
            max_quarter: Don't bet past this quarter (default 4, no OT)
            min_time_remaining: Optional minimum minutes remaining
            max_daily_bets: Maximum bets per day
            enabled: Whether strategy is active
        """
        super().__init__(
            min_edge=min_edge,
            max_daily_bets=max_daily_bets,
            enabled=enabled,
        )

        self.min_confidence = min_confidence
        self.max_quarter = max_quarter
        self.min_time_remaining = min_time_remaining

        # Initialize edge detector
        self.detector = LiveEdgeDetector(
            min_edge=min_edge,
            min_confidence=min_confidence,
        )

        logger.info(
            f"LiveBettingStrategy initialized (min_edge={min_edge:.1%}, "
            f"min_confidence={min_confidence:.1%}, max_quarter={max_quarter})"
        )

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
        Not used for live strategy - use evaluate_live_games instead.

        Live betting requires game state (score, quarter, time) which
        is passed differently than pre-game evaluation.
        """
        logger.warning("LiveBettingStrategy.evaluate_game() called - use evaluate_live_games() instead")
        return []

    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Not used for live strategy - use evaluate_live_games instead.

        Live betting operates on in-progress games with different data structure.
        """
        logger.warning("LiveBettingStrategy.evaluate_games() called - use evaluate_live_games() instead")
        return []

    def evaluate_live_games(
        self,
        live_games: Dict[str, Dict],
        live_odds: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate live games for betting opportunities.

        Args:
            live_games: Dict mapping game_id to game state:
                {
                    'game_id': str,
                    'home_team': str,
                    'away_team': str,
                    'home_score': int,
                    'away_score': int,
                    'quarter': int,
                    'time_remaining': str,  # "5:00"
                }
            live_odds: DataFrame with columns:
                - game_id
                - spread_value, home_spread_odds, away_spread_odds
                - home_ml_odds, away_ml_odds
                - total_value, over_odds, under_odds
                - bookmaker

        Returns:
            List of BetSignal objects for live opportunities
        """
        all_signals = []

        for game_id, game_state in live_games.items():
            # Apply quarter filter
            if game_state['quarter'] > self.max_quarter:
                logger.debug(
                    f"{game_id}: Skipping (Q{game_state['quarter']} > max Q{self.max_quarter})"
                )
                continue

            # Apply time remaining filter if set
            if self.min_time_remaining is not None:
                time_str = game_state.get('time_remaining', '0:00')
                try:
                    mins, secs = time_str.split(':')
                    time_remaining_mins = int(mins) + int(secs) / 60
                    if time_remaining_mins < self.min_time_remaining:
                        logger.debug(
                            f"{game_id}: Skipping (time {time_remaining_mins:.1f}m < "
                            f"min {self.min_time_remaining:.1f}m)"
                        )
                        continue
                except Exception as e:
                    logger.warning(f"{game_id}: Could not parse time_remaining: {e}")

            # Get odds for this game
            game_odds = live_odds[live_odds['game_id'] == game_id]

            if len(game_odds) == 0:
                logger.debug(f"{game_id}: No live odds available")
                continue

            # Build odds dict for detector
            odds_dict = {}

            # Spread odds
            if 'spread_value' in game_odds.columns:
                spread_row = game_odds.iloc[0]
                odds_dict['spread'] = {
                    'spread_value': spread_row.get('spread_value'),
                    'home_odds': spread_row.get('home_spread_odds', -110),
                    'away_odds': spread_row.get('away_spread_odds', -110),
                }

            # Moneyline odds
            if 'home_ml_odds' in game_odds.columns:
                ml_row = game_odds.iloc[0]
                odds_dict['moneyline'] = {
                    'home_odds': ml_row.get('home_ml_odds'),
                    'away_odds': ml_row.get('away_ml_odds'),
                }

            # Total odds
            if 'total_value' in game_odds.columns:
                total_row = game_odds.iloc[0]
                odds_dict['total'] = {
                    'total_value': total_row.get('total_value'),
                    'over_odds': total_row.get('over_odds', -110),
                    'under_odds': total_row.get('under_odds', -110),
                }

            # Use detector to find edges
            try:
                edges = self.detector.find_edges(game_state, odds_dict)

                # Convert detector edges to BetSignal objects
                for edge in edges:
                    # Map detector bet_side to standard format
                    bet_side = edge['bet_side'].upper()

                    signal = BetSignal(
                        strategy_type=StrategyType.LIVE,
                        game_id=game_id,
                        home_team=game_state['home_team'],
                        away_team=game_state['away_team'],
                        bet_type=edge['alert_type'],  # 'spread', 'moneyline', 'total'
                        bet_side=bet_side,
                        model_prob=edge['model_prob'],
                        market_prob=edge['market_prob'],
                        edge=edge['edge'],
                        line=edge.get('line_value'),
                        odds=edge.get('odds'),
                        confidence=edge['confidence'],
                        bookmaker=game_odds.iloc[0].get('bookmaker') if len(game_odds) > 0 else None,
                        filters_passed=[
                            f"quarter_{game_state['quarter']}",
                            f"score_diff_{edge['score_diff']}",
                        ],
                    )

                    all_signals.append(signal)

                    logger.info(
                        f"Live edge: {game_state['away_team']} @ {game_state['home_team']} "
                        f"Q{game_state['quarter']} - {edge['alert_type'].upper()} {bet_side} "
                        f"edge={edge['edge']:.1%} confidence={edge['confidence']}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating live game {game_id}: {e}", exc_info=True)
                continue

        logger.info(
            f"LiveBettingStrategy found {len(all_signals)} signals from "
            f"{len(live_games)} live games"
        )

        return all_signals

    @classmethod
    def conservative_strategy(cls) -> 'LiveBettingStrategy':
        """
        Create conservative live strategy.

        Higher edge and confidence requirements, earlier quarters only.
        """
        return cls(
            min_edge=0.07,  # 7% minimum edge
            min_confidence=0.6,  # 60% confidence
            max_quarter=3,  # Only bet through Q3
            min_time_remaining=3.0,  # At least 3 minutes remaining
        )

    @classmethod
    def aggressive_strategy(cls) -> 'LiveBettingStrategy':
        """
        Create aggressive live strategy.

        Lower thresholds, willing to bet later in games.
        """
        return cls(
            min_edge=0.04,  # 4% minimum edge
            min_confidence=0.4,  # 40% confidence
            max_quarter=4,  # Bet through Q4
            min_time_remaining=1.0,  # At least 1 minute remaining
        )
