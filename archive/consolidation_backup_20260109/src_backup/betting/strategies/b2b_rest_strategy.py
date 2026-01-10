"""
B2B Rest Advantage Strategy

Bets on teams with significant rest advantage over their opponent.
Validated backtest: Rest advantage >= 2 days → 57.0% win rate, +8.7% ROI (237 games)

Strategy Logic:
- If home_rest_days - away_rest_days >= min_rest_advantage: Bet HOME spread
- If away_rest_days - home_rest_days >= min_rest_advantage: Bet AWAY spread
- Uses historical 57% win rate as model probability
"""

from typing import List, Dict
import pandas as pd
from loguru import logger

from .base import BettingStrategy, BetSignal, StrategyType, odds_to_implied_prob


class B2BRestStrategy(BettingStrategy):
    """
    Rest advantage betting strategy.

    Exploits fatigue by betting on teams with significant rest advantage.
    Backtest showed 57% win rate when rest advantage >= 2 days.
    """

    strategy_type = StrategyType.B2B_REST
    priority = 10  # High priority - strong edge

    def __init__(
        self,
        min_rest_advantage: int = 2,
        min_edge: float = 0.03,
        max_daily_bets: int = 5,
        enabled: bool = True,
    ):
        """
        Initialize B2B Rest Advantage Strategy.

        Args:
            min_rest_advantage: Minimum rest day difference to trigger bet (default: 2)
            min_edge: Minimum edge to place bet (default: 3%)
            max_daily_bets: Maximum bets per day
            enabled: Whether strategy is active
        """
        super().__init__(min_edge=min_edge, max_daily_bets=max_daily_bets, enabled=enabled)
        self.min_rest_advantage = min_rest_advantage

        # Historical win rate from backtest (rest advantage >= 2)
        self.model_win_rate = 0.570

        logger.info(
            f"B2BRestStrategy initialized (min_rest_advantage={min_rest_advantage}, "
            f"min_edge={min_edge:.1%}, model_win_rate={self.model_win_rate:.1%})"
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
        Evaluate single game for rest advantage opportunity.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            features: Dict with rest_diff (home_rest - away_rest)
            odds_data: Dict with spread odds for both teams

        Returns:
            List of BetSignal (0 or 1 signals)
        """
        # Check if rest advantage exists
        rest_diff = features.get('rest_diff')
        if rest_diff is None:
            return []

        # Determine if we have actionable rest advantage
        bet_home = rest_diff >= self.min_rest_advantage
        bet_away = rest_diff <= -self.min_rest_advantage

        if not (bet_home or bet_away):
            return []  # No significant rest advantage

        # Get spread odds
        spread_odds = odds_data.get('spread', {})

        if bet_home:
            # Bet HOME spread
            home_odds_data = spread_odds.get('home', {})
            line = home_odds_data.get('line')
            odds = home_odds_data.get('odds')
            bookmaker = home_odds_data.get('bookmaker', 'unknown')

            if line is None or odds is None:
                return []

            # Calculate edge
            model_prob = self.model_win_rate
            market_prob = odds_to_implied_prob(odds)
            edge = model_prob - market_prob

            if edge < self.min_edge:
                return []

            # Confidence based on rest advantage magnitude
            filters_passed = [
                f"rest_advantage_home_{int(rest_diff)}d",
                "situational_edge"
            ]

            additional_score = min((rest_diff - self.min_rest_advantage) / 3, 1.0)
            confidence = self._determine_confidence(edge, filters_passed, additional_score)

            return [BetSignal(
                strategy_type=StrategyType.B2B_REST,
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                bet_type="spread",
                bet_side="HOME",
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                line=line,
                odds=odds,
                confidence=confidence,
                bookmaker=bookmaker,
                filters_passed=filters_passed,
                market_spread=line,
            )]

        else:  # bet_away
            # Bet AWAY spread
            away_odds_data = spread_odds.get('away', {})
            line = away_odds_data.get('line')
            odds = away_odds_data.get('odds')
            bookmaker = away_odds_data.get('bookmaker', 'unknown')

            if line is None or odds is None:
                return []

            # Calculate edge
            model_prob = self.model_win_rate
            market_prob = odds_to_implied_prob(odds)
            edge = model_prob - market_prob

            if edge < self.min_edge:
                return []

            # Confidence based on rest advantage magnitude
            filters_passed = [
                f"rest_advantage_away_{int(abs(rest_diff))}d",
                "situational_edge"
            ]

            additional_score = min((abs(rest_diff) - self.min_rest_advantage) / 3, 1.0)
            confidence = self._determine_confidence(edge, filters_passed, additional_score)

            return [BetSignal(
                strategy_type=StrategyType.B2B_REST,
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                bet_type="spread",
                bet_side="AWAY",
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                line=line,
                odds=odds,
                confidence=confidence,
                bookmaker=bookmaker,
                filters_passed=filters_passed,
                market_spread=line,
            )]

    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate multiple games for rest advantage opportunities.

        Args:
            games_df: DataFrame with game_id, home_team, away_team
            features_df: DataFrame with game features (must include rest_diff)
            odds_df: DataFrame with spread odds

        Returns:
            List of BetSignal objects
        """
        signals = []

        # Handle empty DataFrames
        if games_df.empty or features_df.empty:
            logger.info("B2BRestStrategy: No games or features to evaluate")
            return signals

        # Ensure required columns exist
        if 'game_id' not in games_df.columns or 'game_id' not in features_df.columns:
            logger.warning("B2BRestStrategy: Missing game_id column")
            return signals

        # Merge games with features
        merged = games_df.merge(features_df, on='game_id', how='left')

        for _, game in merged.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']

            # Get rest_diff from features
            rest_diff = game.get('rest_diff')
            if pd.isna(rest_diff):
                continue

            # Build features dict
            features = {'rest_diff': rest_diff}

            # Get spread odds for this game
            game_odds = odds_df[odds_df['game_id'] == game_id]
            spread_odds = {}

            for side in ['home', 'away']:
                side_odds = game_odds[game_odds['team'] == side]
                if len(side_odds) > 0:
                    # Take first bookmaker (TODO: could optimize by choosing best odds)
                    best_odds = side_odds.iloc[0]
                    spread_odds[side] = {
                        'line': best_odds.get('line'),
                        'odds': best_odds.get('odds'),
                        'bookmaker': best_odds.get('bookmaker', 'unknown')
                    }

            odds_data = {'spread': spread_odds}

            # Evaluate game
            game_signals = self.evaluate_game(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                features=features,
                odds_data=odds_data
            )

            signals.extend(game_signals)

        logger.info(
            f"B2BRestStrategy found {len([s for s in signals if s.is_actionable])} "
            f"actionable signals ({len(signals)} total signals)"
        )

        return signals
