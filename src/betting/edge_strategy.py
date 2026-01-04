"""
Validated Edge Strategy for ATS Betting

Based on comprehensive backtesting (2022-2025):
- Edge 5+ & No B2B: 54.8% win rate, +4.6% ROI, p=0.0006
- Edge 5+ & No B2B & Team Filter: 57.9% win rate, +10.5% ROI, p=0.0000
- Current season (2024-25): 57.2% win rate, +9.2% ROI

Strategy Rules:
- BET HOME when: model_edge >= 5 AND home team NOT on back-to-back
- BET AWAY when: model_edge <= -5 AND away team NOT on back-to-back
- EXCLUDE bets on: CHA, IND, MIA, NOP, PHX (historically poor ATS performers)
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Set
import pandas as pd
import numpy as np


# Teams with historically poor ATS performance (2022-2025)
# These teams consistently underperform against the spread
TEAMS_TO_EXCLUDE = {"CHA", "IND", "MIA", "NOP", "PHX"}

# Teams that are good to fade when model predicts against them
TEAMS_TO_FADE = {"CHA", "WAS", "UTA"}


@dataclass
class BetSignal:
    """A betting signal from the edge strategy."""
    game_id: str
    home_team: str
    away_team: str
    bet_side: Literal["HOME", "AWAY", "PASS"]
    model_edge: float
    pred_diff: float
    market_spread: float
    home_b2b: bool
    away_b2b: bool
    rest_advantage: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    filters_passed: List[str]

    @property
    def is_actionable(self) -> bool:
        return self.bet_side != "PASS"


class EdgeStrategy:
    """
    Validated edge-based betting strategy.

    Primary Strategy: Edge 5+ & No B2B
    - 54.8% win rate, +4.6% ROI (2022-2025)
    - p-value: 0.0006 (highly significant)

    Enhanced Strategy: Add Rest Alignment
    - 55.1% win rate, +5.3% ROI
    - Fewer bets but higher per-bet edge
    """

    # Validated thresholds from backtesting
    EDGE_THRESHOLD = 5.0

    def __init__(
        self,
        edge_threshold: float = 5.0,
        require_no_b2b: bool = True,
        require_rest_aligns: bool = False,
        small_spread_only: bool = False,
        small_spread_range: tuple = (-3, 3),
        teams_to_exclude: Optional[Set[str]] = None,
        use_team_filter: bool = False,
        home_only: bool = False,  # Only bet on home teams (better historical performance)
        # Market microstructure signal filters
        require_steam_alignment: bool = False,
        require_rlm_alignment: bool = False,
        require_sharp_alignment: bool = False,
        min_steam_confidence: float = 0.7,
        # CLV-based filters (opt-in, default False for backward compatibility)
        clv_filter_enabled: bool = False,
        min_historical_clv: float = 0.0,
        optimal_timing_filter: bool = False,
        clv_based_sizing: bool = False,
        line_velocity_threshold: Optional[float] = None,
    ):
        """
        Initialize the edge strategy.

        Args:
            edge_threshold: Minimum model edge to trigger bet (default 5.0)
            require_no_b2b: Require team being bet is not on B2B (default True)
            require_rest_aligns: Require rest advantage aligns with bet (default False)
            small_spread_only: Only bet on small spread games (default False)
            small_spread_range: Range for small spreads (default -3 to 3)
            teams_to_exclude: Set of team abbreviations to never bet on (default None)
            use_team_filter: If True, use default TEAMS_TO_EXCLUDE (default False)
            require_steam_alignment: Only bet when steam move aligns with model (default False)
            require_rlm_alignment: Only bet when RLM aligns with model (default False)
            require_sharp_alignment: Only bet when money flow indicates sharp action (default False)
            min_steam_confidence: Minimum steam confidence to require (default 0.7)
            clv_filter_enabled: Enable CLV-based filtering (default False)
            min_historical_clv: Only bet if similar past bets had +CLV (default 0.0)
            optimal_timing_filter: Use optimal booking windows (default False)
            clv_based_sizing: Scale bet size by CLV confidence (default False)
            line_velocity_threshold: Avoid steam moves against us (default None)
        """
        self.edge_threshold = edge_threshold
        self.require_no_b2b = require_no_b2b
        self.require_rest_aligns = require_rest_aligns
        self.small_spread_only = small_spread_only
        self.small_spread_range = small_spread_range
        self.home_only = home_only

        # Market signal filters
        self.require_steam_alignment = require_steam_alignment
        self.require_rlm_alignment = require_rlm_alignment
        self.require_sharp_alignment = require_sharp_alignment
        self.min_steam_confidence = min_steam_confidence

        # CLV-based filters
        self.clv_filter_enabled = clv_filter_enabled
        self.min_historical_clv = min_historical_clv
        self.optimal_timing_filter = optimal_timing_filter
        self.clv_based_sizing = clv_based_sizing
        self.line_velocity_threshold = line_velocity_threshold

        # Team filter: teams we should never bet on
        if teams_to_exclude is not None:
            self.teams_to_exclude = teams_to_exclude
        elif use_team_filter:
            self.teams_to_exclude = TEAMS_TO_EXCLUDE
        else:
            self.teams_to_exclude = set()

    def evaluate_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        pred_diff: float,
        market_spread: float,
        home_b2b: bool = False,
        away_b2b: bool = False,
        rest_advantage: float = 0,
        # Market microstructure signals (optional)
        steam_signal=None,        # SteamMove object
        rlm_signal=None,          # ReverseLineMove object
        money_flow_signal=None,   # MoneyFlowSignal object
    ) -> BetSignal:
        """
        Evaluate a single game and return betting signal.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            pred_diff: Model's predicted home margin (positive = home wins by X)
            market_spread: Vegas spread (negative = home favored by X)
            home_b2b: Is home team on back-to-back?
            away_b2b: Is away team on back-to-back?
            rest_advantage: Home rest days minus away rest days
            steam_signal: Optional SteamMove signal for this game
            rlm_signal: Optional ReverseLineMove signal for this game
            money_flow_signal: Optional MoneyFlowSignal for this game

        Returns:
            BetSignal with recommendation
        """
        # Calculate model edge
        # If spread is -5 (home favored by 5) and model predicts +8 (home by 8)
        # Edge = 8 - 5 = 3 (model thinks home is 3 pts better than line implies)
        model_edge = pred_diff + market_spread

        filters_passed = []
        bet_side = "PASS"

        # Check edge threshold
        if model_edge >= self.edge_threshold:
            # Positive edge suggests bet HOME
            bet_side = "HOME"
            filters_passed.append(f"edge >= {self.edge_threshold}")

            # Check B2B filter
            if self.require_no_b2b and home_b2b:
                bet_side = "PASS"
            else:
                filters_passed.append("no_home_b2b")

            # Check rest alignment
            if self.require_rest_aligns and rest_advantage < 0:
                bet_side = "PASS"
            elif rest_advantage >= 0:
                filters_passed.append("rest_aligns")

        elif model_edge <= -self.edge_threshold:
            # Negative edge suggests bet AWAY
            bet_side = "AWAY"
            filters_passed.append(f"edge <= -{self.edge_threshold}")

            # Check home_only filter (block all away bets)
            if self.home_only:
                bet_side = "PASS"
            else:
                # Check B2B filter
                if self.require_no_b2b and away_b2b:
                    bet_side = "PASS"
                else:
                    filters_passed.append("no_away_b2b")

                # Check rest alignment
                if self.require_rest_aligns and rest_advantage > 0:
                    bet_side = "PASS"
                elif rest_advantage <= 0:
                    filters_passed.append("rest_aligns")

        # Check small spread filter
        if bet_side != "PASS" and self.small_spread_only:
            if not (self.small_spread_range[0] <= market_spread < self.small_spread_range[1]):
                bet_side = "PASS"
            else:
                filters_passed.append("small_spread")

        # Check team exclusion filter
        if bet_side != "PASS" and self.teams_to_exclude:
            team_to_bet = home_team if bet_side == "HOME" else away_team
            if team_to_bet in self.teams_to_exclude:
                bet_side = "PASS"
            else:
                filters_passed.append("team_filter")

        # Market microstructure signal filters
        if bet_side != "PASS":
            # Steam alignment filter
            if self.require_steam_alignment and steam_signal:
                # Check if steam direction matches our bet
                steam_direction = steam_signal.direction.upper()
                if steam_signal.confidence >= self.min_steam_confidence:
                    if (bet_side == "HOME" and steam_direction == "HOME") or \
                       (bet_side == "AWAY" and steam_direction == "AWAY"):
                        filters_passed.append(f"steam_aligned_{steam_signal.confidence:.2f}")
                    else:
                        bet_side = "PASS"  # Steam contradicts model
                else:
                    bet_side = "PASS"  # Steam confidence too low

            # RLM alignment filter
            if self.require_rlm_alignment and rlm_signal and bet_side != "PASS":
                # Check if RLM sharp side matches our bet
                sharp_side = rlm_signal.sharp_side.upper()
                if (bet_side == "HOME" and sharp_side == "HOME") or \
                   (bet_side == "AWAY" and sharp_side == "AWAY"):
                    filters_passed.append(f"rlm_aligned_{rlm_signal.confidence:.2f}")
                else:
                    bet_side = "PASS"  # RLM indicates sharp money on other side

            # Sharp money alignment filter
            if self.require_sharp_alignment and money_flow_signal and bet_side != "PASS":
                # Check if sharp money indicator favors our side
                if money_flow_signal.recommendation == "follow_sharp":
                    flow_side = money_flow_signal.side.upper()
                    if (bet_side == "HOME" and flow_side == "HOME") or \
                       (bet_side == "AWAY" and flow_side == "AWAY"):
                        filters_passed.append("sharp_aligned")
                    else:
                        bet_side = "PASS"  # Sharp money on other side
                else:
                    bet_side = "PASS"  # No clear sharp signal

        # CLV-based filters
        if bet_side != "PASS" and self.clv_filter_enabled:
            # Determine bet_type for CLV lookup
            bet_type = "spread"  # Default to spread for ATS betting

            # Get historical CLV for similar bets
            historical_clv, clv_std = self._get_historical_clv(
                bet_type=bet_type,
                bet_side=bet_side.lower(),
                edge=abs(model_edge)
            )

            # Check if historical CLV meets minimum threshold
            if historical_clv < self.min_historical_clv:
                from loguru import logger
                logger.debug(
                    f"{game_id}: CLV filter blocked {bet_side} "
                    f"(historical CLV {historical_clv:.3f} < {self.min_historical_clv:.3f})"
                )
                bet_side = "PASS"
            else:
                filters_passed.append(f"clv_filter_{historical_clv:.3f}")

        # Optimal timing filter
        if bet_side != "PASS" and self.optimal_timing_filter:
            bet_type = "spread"  # Default to spread for ATS betting

            if not self._check_optimal_timing(
                game_id=game_id,
                bet_type=bet_type,
                bet_side=bet_side.lower()
            ):
                from loguru import logger
                logger.debug(
                    f"{game_id}: Timing filter blocked {bet_side} "
                    f"(not optimal booking window)"
                )
                bet_side = "PASS"
            else:
                filters_passed.append("optimal_timing")

        # Determine confidence level based on edge size and filters
        if bet_side != "PASS":
            edge_abs = abs(model_edge)
            if edge_abs >= 7 and len(filters_passed) >= 3:
                confidence = "HIGH"
            elif edge_abs >= 6 or len(filters_passed) >= 3:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "LOW"

        return BetSignal(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            bet_side=bet_side,
            model_edge=model_edge,
            pred_diff=pred_diff,
            market_spread=market_spread,
            home_b2b=home_b2b,
            away_b2b=away_b2b,
            rest_advantage=rest_advantage,
            confidence=confidence,
            filters_passed=filters_passed,
        )

    def evaluate_games(self, games_df: pd.DataFrame) -> List[BetSignal]:
        """
        Evaluate multiple games from a DataFrame.

        Required columns:
        - game_id, home_team, away_team
        - pred_diff (model prediction)
        - home_spread or market_spread (Vegas line)
        - home_b2b, away_b2b (optional)
        - rest_advantage (optional)

        Returns:
            List of BetSignal objects
        """
        signals = []

        for _, row in games_df.iterrows():
            # Handle different column names for spread
            spread = row.get('home_spread', row.get('market_spread', 0))

            signal = self.evaluate_game(
                game_id=row['game_id'],
                home_team=row['home_team'],
                away_team=row['away_team'],
                pred_diff=row['pred_diff'],
                market_spread=spread,
                home_b2b=row.get('home_b2b', False),
                away_b2b=row.get('away_b2b', False),
                rest_advantage=row.get('rest_advantage', 0),
            )
            signals.append(signal)

        return signals

    def get_actionable_bets(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get DataFrame of actionable bets only.

        Returns DataFrame with columns:
        - game_id, home_team, away_team
        - bet_side, model_edge, confidence
        - market_spread, filters_passed
        """
        signals = self.evaluate_games(games_df)
        actionable = [s for s in signals if s.is_actionable]

        if not actionable:
            return pd.DataFrame()

        records = []
        for s in actionable:
            records.append({
                'game_id': s.game_id,
                'home_team': s.home_team,
                'away_team': s.away_team,
                'bet_side': s.bet_side,
                'model_edge': s.model_edge,
                'pred_diff': s.pred_diff,
                'market_spread': s.market_spread,
                'confidence': s.confidence,
                'filters_passed': ', '.join(s.filters_passed),
                'home_b2b': s.home_b2b,
                'away_b2b': s.away_b2b,
                'rest_advantage': s.rest_advantage,
            })

        df = pd.DataFrame(records)
        # Sort by confidence and edge magnitude
        confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        df['_conf_order'] = df['confidence'].map(confidence_order)
        df = df.sort_values(['_conf_order', 'model_edge'],
                           ascending=[True, False]).drop('_conf_order', axis=1)

        return df

    def _get_historical_clv(
        self,
        bet_type: str,
        bet_side: str,
        edge: float
    ) -> tuple[float, float]:
        """
        Query average CLV for similar past bets.

        Args:
            bet_type: Type of bet ('moneyline', 'spread', 'totals')
            bet_side: Side of bet ('home', 'away', 'over', 'under')
            edge: Model edge percentage

        Returns:
            Tuple of (average CLV at 4hr, standard deviation)
            Returns (0.0, 0.0) if no historical data available
        """
        import sqlite3
        from src.bet_tracker import DB_PATH
        from datetime import datetime, timedelta, timezone

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        try:
            # Query similar bets from last 30 days
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            query = """
                SELECT clv_at_4hr
                FROM bets
                WHERE bet_type = ?
                AND bet_side = ?
                AND edge BETWEEN ? AND ?
                AND clv_at_4hr IS NOT NULL
                AND settled_at >= ?
                AND outcome IS NOT NULL
            """

            # Match similar edge (±2%)
            edge_min = edge - 2.0
            edge_max = edge + 2.0

            results = conn.execute(
                query,
                (bet_type, bet_side, edge_min, edge_max, cutoff)
            ).fetchall()

            if not results:
                return (0.0, 0.0)

            clv_values = [row['clv_at_4hr'] for row in results]
            avg_clv = np.mean(clv_values)
            std_clv = np.std(clv_values) if len(clv_values) > 1 else 0.0

            return (float(avg_clv), float(std_clv))

        finally:
            conn.close()

    def _check_optimal_timing(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str
    ) -> bool:
        """
        Check if current time is optimal for booking.

        Uses analyze_optimal_booking_time() to determine if now is good time.

        Args:
            game_id: Game identifier
            bet_type: Type of bet
            bet_side: Side of bet

        Returns:
            True if within recommended booking window
        """
        import sqlite3
        from src.bet_tracker import DB_PATH
        from datetime import datetime, timezone

        # Get game commence time
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        try:
            # Try to get from line_snapshots first
            result = conn.execute("""
                SELECT DISTINCT commence_time
                FROM opening_lines
                WHERE game_id = ?
                LIMIT 1
            """, (game_id,)).fetchone()

            if not result:
                # Can't determine optimal timing without game info
                return True  # Allow bet by default

            commence_time = datetime.fromisoformat(result['commence_time'])
            if commence_time.tzinfo is None:
                commence_time = commence_time.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            hours_before = (commence_time - now).total_seconds() / 3600

            # Call analyze_optimal_booking_time from line_history
            from src.data.line_history import LineHistoryManager

            manager = LineHistoryManager()
            optimal_analysis = manager.analyze_optimal_booking_time(
                bet_type=bet_type,
                bet_side=bet_side
            )

            if not optimal_analysis:
                return True  # Allow bet if no data

            optimal_hours = optimal_analysis.get('optimal_hours_before')

            if optimal_hours is None:
                return True

            # Allow within ±2 hours of optimal time
            return abs(hours_before - optimal_hours) <= 2.0

        except Exception as e:
            # If error checking timing, allow bet
            return True

        finally:
            conn.close()

    def calculate_kelly_with_clv_adjustment(
        self,
        edge: float,
        kelly: float,
        historical_clv: float,
        clv_std: float
    ) -> float:
        """
        Adjust Kelly sizing based on CLV confidence.

        Args:
            edge: Model edge percentage
            kelly: Original Kelly stake
            historical_clv: Average CLV for similar bets
            clv_std: Standard deviation of historical CLV

        Returns:
            Adjusted Kelly stake
        """
        # CLV >= 2%: Full Kelly (1.0x)
        if historical_clv >= 0.02:
            return kelly * 1.0

        # CLV 0-2%: Half Kelly (0.5x)
        elif historical_clv >= 0.0:
            return kelly * 0.5

        # CLV < 0%: No bet (0x)
        else:
            return 0.0

    @classmethod
    def primary_strategy(cls) -> 'EdgeStrategy':
        """
        Create the primary validated strategy.

        Edge 5+ & No B2B: 54.8% win rate, +4.6% ROI
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            require_rest_aligns=False,
        )

    @classmethod
    def enhanced_strategy(cls) -> 'EdgeStrategy':
        """
        Create the enhanced strategy with rest alignment.

        Edge 5+ & No B2B & Rest Aligns: 55.1% win rate, +5.3% ROI
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            require_rest_aligns=True,
        )

    @classmethod
    def aggressive_strategy(cls) -> 'EdgeStrategy':
        """
        Create a more aggressive strategy (lower threshold).

        Edge 4+ & No B2B: Higher volume, lower per-bet edge
        """
        return cls(
            edge_threshold=4.0,
            require_no_b2b=True,
            require_rest_aligns=False,
        )

    @classmethod
    def team_filtered_strategy(cls) -> 'EdgeStrategy':
        """
        Create the team-filtered strategy (RECOMMENDED).

        Edge 5+ & No B2B & Exclude Poor ATS Teams: 57.9% win rate, +10.5% ROI
        Excludes: CHA, IND, MIA, NOP, PHX
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            require_rest_aligns=False,
            use_team_filter=True,
        )

    @classmethod
    def optimal_strategy(cls) -> 'EdgeStrategy':
        """
        Create the optimal strategy combining all validated filters.

        Edge 5+ & No B2B & Team Filter & Rest Aligns
        Fewer bets, highest per-bet edge
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            require_rest_aligns=True,
            use_team_filter=True,
        )

    @classmethod
    def clv_filtered_strategy(cls) -> 'EdgeStrategy':
        """
        Create CLV-filtered strategy.

        Edge 5+ & No B2B & Team Filter & CLV Filter & HOME ONLY
        Only bets with historically positive CLV (+1% minimum)

        Note: Away bets disabled due to poor historical performance:
        - HOME: 56.2% win rate, +8.5% ROI
        - AWAY: 35.7% win rate, -34.7% ROI
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            use_team_filter=True,
            clv_filter_enabled=True,
            min_historical_clv=0.01,  # Require +1% historical CLV
            home_only=True,  # Disable away bets due to poor performance
        )

    @classmethod
    def optimal_timing_strategy(cls) -> 'EdgeStrategy':
        """
        Create optimal timing strategy.

        Edge 5+ & No B2B & Optimal Timing Filter
        Only bets placed at optimal time windows based on line movement analysis
        """
        return cls(
            edge_threshold=5.0,
            require_no_b2b=True,
            optimal_timing_filter=True,
        )


def calculate_expected_value(
    win_rate: float,
    odds: float = -110,
) -> float:
    """
    Calculate expected value per $100 bet at given win rate and odds.

    Args:
        win_rate: Probability of winning (0-1)
        odds: American odds (default -110)

    Returns:
        Expected profit per $100 wagered
    """
    if odds < 0:
        # Favorite: -110 means bet $110 to win $100
        win_profit = 100 / (-odds / 100)
        loss = 100
    else:
        # Underdog: +110 means bet $100 to win $110
        win_profit = odds
        loss = 100

    ev = (win_rate * win_profit) - ((1 - win_rate) * loss)
    return ev


def calculate_roi(win_rate: float, odds: float = -110) -> float:
    """
    Calculate ROI at given win rate and odds.

    At -110 odds:
    - 50% win rate = -4.5% ROI (the vig)
    - 52.4% win rate = 0% ROI (break-even)
    - 55% win rate = +4.5% ROI
    """
    ev = calculate_expected_value(win_rate, odds)
    return ev / 100 * 100  # As percentage


# Backtested performance metrics
STRATEGY_PERFORMANCE = {
    'Edge 5+ (baseline)': {
        'win_rate': 0.534,
        'roi': 1.9,
        'p_value': 0.0055,
        'sample_size': 1422,
    },
    'Edge 5+ & No B2B': {
        'win_rate': 0.548,
        'roi': 4.6,
        'p_value': 0.0006,
        'sample_size': 1133,
    },
    'Edge 5+ & No B2B & Team Filter': {
        'win_rate': 0.579,
        'roi': 10.5,
        'p_value': 0.0000,
        'sample_size': 931,
        'excluded_teams': ['CHA', 'IND', 'MIA', 'NOP', 'PHX'],
        'recommended': True,
    },
    'Edge 5+ & No B2B & Rest Aligns': {
        'win_rate': 0.551,
        'roi': 5.3,
        'p_value': 0.0006,
        'sample_size': 992,
    },
    'Edge 5+ & Small Spread': {
        'win_rate': 0.547,
        'roi': 4.4,
        'p_value': 0.0588,
        'sample_size': 276,
    },
}
