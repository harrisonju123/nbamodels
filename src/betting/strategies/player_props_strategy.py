"""
Player Props Strategy

Betting strategy for player props (points, rebounds, assists, etc.)
using individual player prop models.

IMPORTANT: Requires lineup and injury data to avoid betting on:
- Injured players (Out, Questionable)
- Non-starters with reduced minutes
- Players not confirmed in lineup
"""

import os
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

from src.betting.strategies.base import (
    BettingStrategy,
    BetSignal,
    StrategyType,
    odds_to_implied_prob,
)
from src.models.player_props import (
    PointsPropModel,
    ReboundsPropModel,
    AssistsPropModel,
    ThreesPropModel,
)
from src.data.lineup_scrapers import ESPNLineupClient
from src.data.espn_injuries import ESPNClient


class PlayerPropsStrategy(BettingStrategy):
    """
    Player props betting strategy.

    Uses dedicated models for each prop type (PTS, REB, AST, 3PM)
    to predict player performance and identify edges vs market lines.

    Filters:
    - Minimum edge threshold
    - Optional minimum minutes filter (starters only)
    - Optional team filter (avoid certain teams)
    """

    strategy_type = StrategyType.PLAYER_PROPS
    priority = 4  # Lower priority - requires trained models

    def __init__(
        self,
        models_dir: str = "models/player_props",
        prop_types: Optional[List[str]] = None,
        min_edge: float = 0.05,
        min_minutes: Optional[float] = None,  # Filter to players with X+ min/game
        max_daily_bets: int = 5,
        enabled: bool = True,
        lineup_client: Optional[ESPNLineupClient] = None,
        injury_client: Optional[ESPNClient] = None,
        require_starter: bool = True,  # Only bet on confirmed starters
        skip_questionable: bool = True,  # Skip questionable players
    ):
        """
        Initialize player props strategy.

        Args:
            models_dir: Directory containing trained prop models
            prop_types: List of prop types to bet (default: PTS, REB, AST, 3PM)
            min_edge: Minimum edge to trigger bet (default 5%)
            min_minutes: Optional minimum minutes filter
            max_daily_bets: Maximum bets per day
            enabled: Whether strategy is active
            lineup_client: Client for fetching confirmed lineups (REQUIRED)
            injury_client: Client for fetching injury reports (REQUIRED)
            require_starter: Only bet on confirmed starters (default True)
            skip_questionable: Skip questionable players (default True)
        """
        super().__init__(
            min_edge=min_edge,
            max_daily_bets=max_daily_bets,
            enabled=enabled,
        )

        self.models_dir = models_dir
        self.prop_types = prop_types or ["PTS", "REB", "AST", "3PM"]
        self.min_minutes = min_minutes
        self.require_starter = require_starter
        self.skip_questionable = skip_questionable

        # Lineup and injury clients
        self.lineup_client = lineup_client or ESPNLineupClient()
        self.injury_client = injury_client or ESPNClient()

        # Load models
        self.models = self._load_models()

        if not self.models:
            logger.warning(
                f"No player prop models loaded from {models_dir}. "
                "Strategy will not generate signals."
            )

        logger.info(
            f"PlayerPropsStrategy initialized with lineup/injury filtering: "
            f"require_starter={require_starter}, skip_questionable={skip_questionable}"
        )

    def _load_models(self) -> Dict[str, any]:
        """Load all available prop models."""
        models = {}

        model_classes = {
            "PTS": PointsPropModel,
            "REB": ReboundsPropModel,
            "AST": AssistsPropModel,
            "3PM": ThreesPropModel,
        }

        for prop_type in self.prop_types:
            if prop_type not in model_classes:
                logger.warning(f"Unknown prop type: {prop_type}")
                continue

            model_path = os.path.join(
                self.models_dir,
                f"{prop_type.lower()}_model.pkl"
            )

            if os.path.exists(model_path):
                try:
                    models[prop_type] = model_classes[prop_type].load(model_path)
                    logger.info(f"Loaded {prop_type} model from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading {prop_type} model: {e}")
            else:
                logger.debug(f"{prop_type} model not found at {model_path}")

        return models

    def is_player_safe_to_bet(
        self,
        player_name: str,
        player_team: str,
        game_id: str,
        game_date: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if it's safe to bet on a player.

        Returns:
            (is_safe, reason) - True if safe, False with reason if not
        """
        # 1. Check confirmed lineup
        try:
            lineups = self.lineup_client.get_lineup_for_game(game_id)

            if lineups is not None and not lineups.empty:
                player_lineup = lineups[
                    (lineups['player_name'] == player_name) &
                    (lineups['team_abbrev'] == player_team)
                ]

                if len(player_lineup) == 0:
                    return False, f"Not in confirmed lineup"

                player_info = player_lineup.iloc[0]

                # Check if starter (if required)
                if self.require_starter and not player_info.get('is_starter', False):
                    return False, f"Not a confirmed starter"

                # Check player status from lineup
                status = player_info.get('status', '').lower()
                if status in ['out', 'inactive']:
                    return False, f"Status: {status}"
            else:
                # No lineup data available - be conservative
                logger.warning(f"No lineup data for game {game_id}, proceeding with caution")

        except Exception as e:
            logger.warning(f"Error checking lineup for {player_name}: {e}")
            # Continue to injury check

        # 2. Check injury report
        try:
            injuries = self.injury_client.get_injuries()

            if injuries is not None and not injuries.empty:
                player_injury = injuries[
                    (injuries['player_name'] == player_name) &
                    (injuries['team'] == player_team)
                ]

                if len(player_injury) > 0:
                    injury_info = player_injury.iloc[0]
                    status = injury_info.get('status', '').lower()

                    # Always skip "Out" players
                    if status in ['out', 'o']:
                        return False, f"Injury status: Out"

                    # Skip questionable if configured
                    if self.skip_questionable and status in ['questionable', 'q', 'doubtful', 'd']:
                        return False, f"Injury status: {status}"

                    # Log but allow probable/day-to-day
                    if status in ['probable', 'p', 'day-to-day', 'gtd']:
                        logger.info(f"{player_name} is {status} but proceeding")

        except Exception as e:
            logger.warning(f"Error checking injuries for {player_name}: {e}")
            # Continue - better to have false positives than miss bets

        return True, None

    def evaluate_player_prop(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        player_id: str,
        player_name: str,
        player_team: str,
        prop_type: str,
        line: float,
        over_odds: int,
        under_odds: int,
        features: Dict,
        bookmaker: Optional[str] = None,
    ) -> List[BetSignal]:
        """
        Evaluate a single player prop.

        Args:
            game_id: Game identifier
            home_team, away_team: Team abbreviations
            player_id: Player identifier
            player_name: Player name
            player_team: Player's team
            prop_type: Prop type (PTS, REB, AST, 3PM)
            line: Prop line (e.g., 25.5 points)
            over_odds, under_odds: American odds
            features: Player features dict
            bookmaker: Bookmaker name

        Returns:
            List of BetSignal objects (0-2: over and/or under)
        """
        if prop_type not in self.models:
            return []

        # Apply filters
        filters_passed = []

        # Minutes filter
        if self.min_minutes is not None:
            min_avg = features.get("min_roll5", 0)
            if min_avg < self.min_minutes:
                logger.debug(
                    f"{player_name}: Filtered by min_minutes "
                    f"({min_avg:.1f} < {self.min_minutes:.1f})"
                )
                return []
            filters_passed.append(f"min>={self.min_minutes:.0f}")

        # Get model prediction
        model = self.models[prop_type]
        features_df = pd.DataFrame([features])

        try:
            over_prob = model.predict_over_prob(features_df, line)[0]
            under_prob = 1 - over_prob

            # Get market implied probabilities
            over_market_prob = odds_to_implied_prob(over_odds)
            under_market_prob = odds_to_implied_prob(under_odds)

            # Calculate edges
            over_edge = over_prob - over_market_prob
            under_edge = under_prob - under_market_prob

            signals = []

            # Check over edge
            if over_edge >= self.min_edge:
                confidence = self._determine_confidence(
                    over_edge,
                    filters_passed,
                    additional_score=over_prob
                )

                signals.append(BetSignal(
                    strategy_type=StrategyType.PLAYER_PROPS,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    bet_type="player_prop",
                    bet_side="OVER",
                    model_prob=over_prob,
                    market_prob=over_market_prob,
                    edge=over_edge,
                    line=line,
                    odds=over_odds,
                    confidence=confidence,
                    bookmaker=bookmaker,
                    filters_passed=filters_passed.copy(),
                    player_id=player_id,
                    player_name=player_name,
                    prop_type=prop_type,
                ))

            # Check under edge
            if under_edge >= self.min_edge:
                confidence = self._determine_confidence(
                    under_edge,
                    filters_passed,
                    additional_score=under_prob
                )

                signals.append(BetSignal(
                    strategy_type=StrategyType.PLAYER_PROPS,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    bet_type="player_prop",
                    bet_side="UNDER",
                    model_prob=under_prob,
                    market_prob=under_market_prob,
                    edge=under_edge,
                    line=line,
                    odds=under_odds,
                    confidence=confidence,
                    bookmaker=bookmaker,
                    filters_passed=filters_passed.copy(),
                    player_id=player_id,
                    player_name=player_name,
                    prop_type=prop_type,
                ))

            return signals

        except Exception as e:
            logger.error(f"Error evaluating {player_name} {prop_type}: {e}", exc_info=True)
            return []

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
        Not typically used for props - use evaluate_games() instead.

        Props require player-level data which is better handled in batch.
        """
        logger.warning("PlayerPropsStrategy.evaluate_game() called - use evaluate_games() instead")
        return []

    def evaluate_games(
        self,
        games_df: pd.DataFrame,
        features_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> List[BetSignal]:
        """
        Evaluate player props across all games.

        Args:
            games_df: DataFrame with game_id, home_team, away_team
            features_df: DataFrame with player features (player_id, player_name, prop features)
            odds_df: DataFrame with player prop odds (from OddsAPIClient.get_player_props()):
                - game_id
                - player_name, player_id, player_team
                - prop_type
                - line
                - side (over/under)
                - odds
                - bookmaker

        Returns:
            List of all BetSignal objects from all props
        """
        all_signals = []

        if not self.models:
            logger.warning("No models loaded, cannot evaluate props")
            return all_signals

        # Merge odds with games to get team info
        merged = odds_df.merge(
            games_df[['game_id', 'home_team', 'away_team']],
            on='game_id',
            how='inner'
        )

        if len(merged) == 0:
            logger.warning("No player props with game data available")
            return all_signals

        # Group by player/game/prop_type to get over and under odds
        grouped = merged.groupby(['game_id', 'player_name', 'prop_type'])

        logger.info(
            f"Evaluating {len(grouped)} player props across {len(games_df)} games"
        )

        for (game_id, player_name, prop_type), group in grouped:
            # Get over and under rows
            over_row = group[group['side'] == 'over']
            under_row = group[group['side'] == 'under']

            if len(over_row) == 0 or len(under_row) == 0:
                continue

            over_row = over_row.iloc[0]
            under_row = under_row.iloc[0]

            # Line should be same for both sides
            line = over_row['line']

            # SAFETY CHECK: Verify player is safe to bet on
            player_team = over_row.get('player_team', '')
            game_date = games_df[games_df['game_id'] == game_id]['game_date'].iloc[0] if 'game_date' in games_df.columns else None

            is_safe, reason = self.is_player_safe_to_bet(
                player_name=player_name,
                player_team=player_team,
                game_id=game_id,
                game_date=game_date,
            )

            if not is_safe:
                logger.info(f"Skipping {player_name} {prop_type}: {reason}")
                continue

            # Get player features
            player_features = features_df[
                (features_df['player_name'] == player_name) &
                (features_df['game_id'] == game_id)
            ]

            if len(player_features) == 0:
                logger.debug(f"No features for {player_name} in {game_id}")
                continue

            features_dict = player_features.iloc[0].to_dict()

            # Evaluate this prop
            signals = self.evaluate_player_prop(
                game_id=game_id,
                home_team=over_row['home_team'],
                away_team=over_row['away_team'],
                player_id=over_row.get('player_id'),
                player_name=player_name,
                player_team=over_row.get('player_team'),
                prop_type=prop_type,
                line=line,
                over_odds=over_row['odds'],
                under_odds=under_row['odds'],
                features=features_dict,
                bookmaker=over_row.get('bookmaker'),
            )

            all_signals.extend(signals)

        logger.info(
            f"PlayerPropsStrategy found {len(all_signals)} signals from "
            f"{len(grouped)} props"
        )

        return all_signals

    @classmethod
    def starters_only_strategy(cls) -> 'PlayerPropsStrategy':
        """
        Create strategy that only bets on starters (25+ min/game).

        Higher volume props tend to be more predictable.
        """
        return cls(
            min_edge=0.05,
            min_minutes=25.0,  # Starters average 25+ minutes
        )

    @classmethod
    def conservative_strategy(cls) -> 'PlayerPropsStrategy':
        """
        Create conservative props strategy.

        Higher edge requirement and starters only.
        """
        return cls(
            min_edge=0.07,  # 7% minimum edge
            min_minutes=28.0,  # Primary starters only
            prop_types=["PTS"],  # Points only (most predictable)
        )
