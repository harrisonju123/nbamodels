"""
Correlation Tracker

Tracks correlated exposures across multiple dimensions:
1. Same team (betting on/against same team multiple times)
2. Same game (multiple outcomes of same game)
3. Same conference (moderate correlation)
4. Same division (higher correlation than conference)

Uses multiplicative discounts to reduce position sizes when
correlations are detected.
"""

from typing import Dict
from datetime import datetime
from loguru import logger

from .config import RiskConfig
from .models import (
    BetCorrelationContext,
    ExposureSnapshot,
    PendingBet,
    TEAM_LOOKUP
)


class CorrelationTracker:
    """
    Tracks correlated exposures across pending bets.

    Maintains in-memory state of unsettled bets and calculates
    correlation factors for new bets.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize correlation tracker.

        Args:
            config: Risk configuration
        """
        self.config = config
        self._pending_bets: Dict[str, PendingBet] = {}
        self._team_lookup = TEAM_LOOKUP

        logger.debug("CorrelationTracker initialized")

    def add_pending_bet(
        self,
        bet_id: str,
        context: BetCorrelationContext,
        amount: float
    ) -> None:
        """
        Register a new pending bet for correlation tracking.

        Args:
            bet_id: Unique bet identifier
            context: Bet correlation context
            amount: Bet amount
        """
        timestamp = datetime.now().isoformat()

        pending_bet = PendingBet(
            bet_id=bet_id,
            game_id=context.game_id,
            context=context,
            amount=amount,
            timestamp=timestamp
        )

        self._pending_bets[bet_id] = pending_bet

        logger.debug(
            f"Added pending bet: {bet_id} | "
            f"Game: {context.game_id} | "
            f"Side: {context.bet_side} | "
            f"Amount: ${amount:.2f}"
        )

    def settle_bet(self, bet_id: str) -> None:
        """
        Remove settled bet from correlation tracking.

        Args:
            bet_id: Bet identifier to remove
        """
        if bet_id in self._pending_bets:
            del self._pending_bets[bet_id]
            logger.debug(f"Settled bet removed from tracking: {bet_id}")
        else:
            logger.warning(f"Attempted to settle unknown bet: {bet_id}")

    def get_exposure_snapshot(self) -> ExposureSnapshot:
        """
        Get current exposure breakdown by all dimensions.

        Returns:
            ExposureSnapshot with exposure by team, game, conference, division
        """
        snapshot = ExposureSnapshot()

        for pending_bet in self._pending_bets.values():
            context = pending_bet.context
            amount = pending_bet.amount

            # Total pending
            snapshot.total_pending += amount

            # By game
            if context.game_id not in snapshot.by_game:
                snapshot.by_game[context.game_id] = 0
            snapshot.by_game[context.game_id] += amount

            # By team (only for non-totals bets)
            if context.bet_team:
                if context.bet_team not in snapshot.by_team:
                    snapshot.by_team[context.bet_team] = 0
                snapshot.by_team[context.bet_team] += amount

            # By conference
            if context.conference:
                if context.conference not in snapshot.by_conference:
                    snapshot.by_conference[context.conference] = 0
                snapshot.by_conference[context.conference] += amount

            # By division
            if context.division:
                if context.division not in snapshot.by_division:
                    snapshot.by_division[context.division] = 0
                snapshot.by_division[context.division] += amount

        return snapshot

    def get_correlation_factor(self, new_bet: BetCorrelationContext, bankroll: float) -> float:
        """
        Calculate correlation discount factor for a new bet.

        Uses multiplicative discounts:
        - Same team: 0.7x
        - Same game: 0.5x
        - High conference exposure: 0.9x
        - High division exposure: 0.85x

        Args:
            new_bet: Context for the new bet being evaluated
            bankroll: Current bankroll for exposure percentage calculations

        Returns:
            Factor between 0.0 and 1.0 to multiply bet size by
        """
        factor = 1.0
        snapshot = self.get_exposure_snapshot()

        # 1. Same team discount
        team_exposure = self.get_team_exposure(new_bet.bet_team)
        if team_exposure > 0:
            factor *= self.config.same_team_discount
            logger.debug(
                f"Same team discount applied: {new_bet.bet_team} "
                f"(current exposure: ${team_exposure:.2f})"
            )

        # 2. Same game discount
        game_exposure = self.get_same_game_exposure(new_bet.game_id)
        if game_exposure > 0:
            factor *= self.config.same_game_discount
            logger.debug(
                f"Same game discount applied: {new_bet.game_id} "
                f"(current exposure: ${game_exposure:.2f})"
            )

        # 3. Conference discount (if exposure > 50% of limit)
        if new_bet.conference and bankroll > 0:
            conf_exposure = snapshot.by_conference.get(new_bet.conference, 0)
            conf_limit = bankroll * self.config.max_same_conference_exposure
            if conf_exposure > conf_limit * 0.5:
                factor *= self.config.same_conference_discount
                logger.debug(
                    f"Conference discount applied: {new_bet.conference} "
                    f"(exposure: ${conf_exposure:.2f} / ${conf_limit:.2f})"
                )

        # 4. Division discount (if exposure > 50% of limit)
        if new_bet.division and bankroll > 0:
            div_exposure = snapshot.by_division.get(new_bet.division, 0)
            div_limit = bankroll * self.config.max_same_division_exposure
            if div_exposure > div_limit * 0.5:
                factor *= self.config.same_division_discount
                logger.debug(
                    f"Division discount applied: {new_bet.division} "
                    f"(exposure: ${div_exposure:.2f} / ${div_limit:.2f})"
                )

        # Floor at 10% to allow small bets even with high correlation
        factor = max(0.1, factor)

        return factor

    def get_team_exposure(self, team: str) -> float:
        """
        Get total exposure involving a specific team.

        Args:
            team: Team abbreviation

        Returns:
            Total exposure amount
        """
        if not team:
            return 0.0

        snapshot = self.get_exposure_snapshot()
        return snapshot.by_team.get(team, 0.0)

    def get_same_game_exposure(self, game_id: str) -> float:
        """
        Get exposure already placed on a specific game.

        Args:
            game_id: Game identifier

        Returns:
            Total exposure amount on this game
        """
        snapshot = self.get_exposure_snapshot()
        return snapshot.by_game.get(game_id, 0.0)

    def get_conference_exposure(self, conference: str) -> float:
        """
        Get total exposure in a specific conference.

        Args:
            conference: Conference name ("Eastern" or "Western")

        Returns:
            Total exposure amount
        """
        snapshot = self.get_exposure_snapshot()
        return snapshot.by_conference.get(conference, 0.0)

    def get_division_exposure(self, division: str) -> float:
        """
        Get total exposure in a specific division.

        Args:
            division: Division name

        Returns:
            Total exposure amount
        """
        snapshot = self.get_exposure_snapshot()
        return snapshot.by_division.get(division, 0.0)

    def check_exposure_limits(
        self,
        new_bet: BetCorrelationContext,
        new_amount: float,
        bankroll: float
    ) -> tuple[bool, list[str]]:
        """
        Check if new bet would violate exposure limits.

        Args:
            new_bet: Context for new bet
            new_amount: Proposed bet amount
            bankroll: Current bankroll

        Returns:
            (is_allowed, violation_reasons)
        """
        violations = []
        snapshot = self.get_exposure_snapshot()

        # Check same team limit
        if new_bet.bet_team:
            team_exposure = snapshot.by_team.get(new_bet.bet_team, 0) + new_amount
            team_limit = bankroll * self.config.max_same_team_exposure
            if team_exposure > team_limit:
                violations.append(
                    f"Same team limit: ${team_exposure:.2f} > ${team_limit:.2f}"
                )

        # Check same game limit
        game_exposure = snapshot.by_game.get(new_bet.game_id, 0) + new_amount
        game_limit = bankroll * self.config.max_same_game_exposure
        if game_exposure > game_limit:
            violations.append(
                f"Same game limit: ${game_exposure:.2f} > ${game_limit:.2f}"
            )

        # Check conference limit
        if new_bet.conference:
            conf_exposure = snapshot.by_conference.get(new_bet.conference, 0) + new_amount
            conf_limit = bankroll * self.config.max_same_conference_exposure
            if conf_exposure > conf_limit:
                violations.append(
                    f"Conference limit: ${conf_exposure:.2f} > ${conf_limit:.2f}"
                )

        # Check division limit
        if new_bet.division:
            div_exposure = snapshot.by_division.get(new_bet.division, 0) + new_amount
            div_limit = bankroll * self.config.max_same_division_exposure
            if div_exposure > div_limit:
                violations.append(
                    f"Division limit: ${div_exposure:.2f} > ${div_limit:.2f}"
                )

        is_allowed = len(violations) == 0
        return is_allowed, violations

    def reset(self):
        """Clear all pending bets (useful for testing)."""
        self._pending_bets.clear()
        logger.info("CorrelationTracker reset")

    def get_pending_count(self) -> int:
        """Get number of pending bets."""
        return len(self._pending_bets)

    def __repr__(self) -> str:
        snapshot = self.get_exposure_snapshot()
        return (
            f"CorrelationTracker("
            f"pending_bets={self.get_pending_count()}, "
            f"total_exposure=${snapshot.total_pending:.2f}"
            f")"
        )
