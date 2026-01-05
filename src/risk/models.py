"""
Risk Management Data Models

Contains NBA conference/division mappings and dataclasses for
risk tracking and correlation analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from enum import Enum


# === NBA Conference and Division Mappings ===

NBA_CONFERENCES = {
    "Eastern": {
        "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
        "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
        "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"]
    },
    "Western": {
        "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
        "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
        "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"]
    }
}


def build_team_lookup() -> Dict[str, tuple]:
    """
    Build reverse lookup: team -> (conference, division).

    Returns:
        Dict mapping team abbreviation to (conference, division) tuple

    Example:
        >>> lookup = build_team_lookup()
        >>> lookup['LAL']
        ('Western', 'Pacific')
    """
    lookup = {}
    for conference, divisions in NBA_CONFERENCES.items():
        for division, teams in divisions.items():
            for team in teams:
                lookup[team] = (conference, division)
    return lookup


# Build once at module level
TEAM_LOOKUP = build_team_lookup()


def get_team_conference(team: str) -> str:
    """Get conference for a team."""
    return TEAM_LOOKUP.get(team, ("Unknown", "Unknown"))[0]


def get_team_division(team: str) -> str:
    """Get division for a team."""
    return TEAM_LOOKUP.get(team, ("Unknown", "Unknown"))[1]


# === Enums ===

class EdgeBucket(str, Enum):
    """Edge bucket categories for attribution."""
    LOW = "5-7%"
    MEDIUM = "7-10%"
    HIGH = "10%+"


class BetSide(str, Enum):
    """Standardized bet side labels."""
    HOME = "HOME"
    AWAY = "AWAY"
    OVER = "OVER"
    UNDER = "UNDER"
    PASS = "PASS"


# === Correlation Dataclasses ===

@dataclass
class TeamCorrelation:
    """Correlation context for a team."""
    team: str
    conference: str
    division: str

    @classmethod
    def from_team(cls, team: str) -> 'TeamCorrelation':
        """
        Create TeamCorrelation from team abbreviation.

        Args:
            team: Team abbreviation (e.g., "LAL", "BOS")

        Returns:
            TeamCorrelation instance
        """
        conf, div = TEAM_LOOKUP.get(team, ("Unknown", "Unknown"))
        return cls(team=team, conference=conf, division=div)


@dataclass
class BetCorrelationContext:
    """
    Full correlation context for a bet.

    Contains all information needed to calculate correlation adjustments.
    """
    game_id: str
    home_team: TeamCorrelation
    away_team: TeamCorrelation
    bet_side: str  # "HOME", "AWAY", "OVER", "UNDER"
    bet_type: str  # "moneyline", "spread", "totals"

    @property
    def bet_team(self) -> str:
        """Get the team being bet on (or empty string for totals)."""
        if self.bet_side == "HOME":
            return self.home_team.team
        elif self.bet_side == "AWAY":
            return self.away_team.team
        else:
            return ""  # Totals don't have a specific team

    @property
    def conference(self) -> str:
        """Get conference of the bet team."""
        if self.bet_side == "HOME":
            return self.home_team.conference
        elif self.bet_side == "AWAY":
            return self.away_team.conference
        else:
            return ""

    @property
    def division(self) -> str:
        """Get division of the bet team."""
        if self.bet_side == "HOME":
            return self.home_team.division
        elif self.bet_side == "AWAY":
            return self.away_team.division
        else:
            return ""

    @classmethod
    def from_game(cls, game_id: str, home_team: str, away_team: str,
                  bet_side: str, bet_type: str) -> 'BetCorrelationContext':
        """
        Create BetCorrelationContext from game details.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            bet_side: Bet side (HOME/AWAY/OVER/UNDER)
            bet_type: Bet type (moneyline/spread/totals)

        Returns:
            BetCorrelationContext instance
        """
        return cls(
            game_id=game_id,
            home_team=TeamCorrelation.from_team(home_team),
            away_team=TeamCorrelation.from_team(away_team),
            bet_side=bet_side,
            bet_type=bet_type
        )


@dataclass
class ExposureSnapshot:
    """
    Current portfolio exposure snapshot.

    Tracks exposure across multiple dimensions for correlation analysis.
    """
    by_team: Dict[str, float] = field(default_factory=dict)
    """Team -> total exposure amount"""

    by_game: Dict[str, float] = field(default_factory=dict)
    """Game ID -> total exposure amount"""

    by_conference: Dict[str, float] = field(default_factory=dict)
    """Conference -> total exposure amount"""

    by_division: Dict[str, float] = field(default_factory=dict)
    """Division -> total exposure amount"""

    total_pending: float = 0.0
    """Total exposure in unsettled bets"""

    daily_wagered: float = 0.0
    """Total wagered today"""

    weekly_wagered: float = 0.0
    """Total wagered this week"""


@dataclass
class PendingBet:
    """Tracking info for a pending (unsettled) bet."""
    bet_id: str
    game_id: str
    context: BetCorrelationContext
    amount: float
    timestamp: str


# === Position Sizing Result ===

@dataclass
class PositionSizeResult:
    """
    Result of position sizing calculation.

    Contains the final bet size and all adjustments applied.
    """
    final_size: float
    """Final recommended bet amount after all adjustments"""

    base_kelly: float
    """Original Kelly fraction before adjustments"""

    base_size: float
    """Base bet size (bankroll * base_kelly)"""

    correlation_factor: float
    """Multiplier from correlation adjustments (0-1)"""

    drawdown_factor: float
    """Multiplier from drawdown scaling (0-1)"""

    exposure_cap: Optional[float] = None
    """Final cap from exposure limits (if applied)"""

    adjustments: List[str] = field(default_factory=list)
    """List of adjustment reasons applied"""

    warnings: List[str] = field(default_factory=list)
    """List of risk warnings"""

    def should_bet(self) -> bool:
        """Check if bet should be placed."""
        return self.final_size >= 1.0  # Minimum $1 bet


# === Risk Status ===

@dataclass
class RiskStatus:
    """
    Current risk management status.

    Used for dashboards and monitoring.
    """
    current_bankroll: float
    peak_bankroll: float
    drawdown_pct: float
    drawdown_scale_factor: float
    betting_paused: bool
    pause_reason: Optional[str]

    daily_wagered: float
    daily_remaining: float
    weekly_wagered: float
    weekly_remaining: float

    pending_exposure: float
    pending_exposure_pct: float

    exposure_snapshot: ExposureSnapshot


# === Bet Evaluation Result ===

@dataclass
class BetEvaluation:
    """
    Result of evaluating a potential bet through risk framework.

    Used by LiveRiskManager.
    """
    should_bet: bool
    """Whether the bet should be placed"""

    recommended_size: float
    """Recommended bet amount"""

    position_result: Optional[PositionSizeResult] = None
    """Detailed position sizing result"""

    rejection_reason: Optional[str] = None
    """Reason for rejecting the bet (if rejected)"""

    warnings: List[str] = field(default_factory=list)
    """Risk warnings"""
