from .team_features import TeamFeatureBuilder
from .game_features import GameFeatureBuilder
from .player_features import PlayerFeatureBuilder, PlayerStatsClient

# Matchup features
try:
    from .matchup_features import (
        MatchupFeatureBuilder,
        MatchupStats,
        MatchupAdjuster,
    )
except (ImportError, Exception):
    MatchupFeatureBuilder = None
    MatchupStats = None
    MatchupAdjuster = None

# Player impact and lineup features (Phase 5)
try:
    from .player_impact import (
        PlayerImpactModel,
        PlayerImpact,
        StintDataFetcher,
        build_player_impact_model,
    )
except (ImportError, Exception):
    PlayerImpactModel = None
    PlayerImpact = None
    StintDataFetcher = None
    build_player_impact_model = None

try:
    from .lineup_features import (
        LineupFeatureBuilder,
        get_expected_lineup_impact,
    )
except (ImportError, Exception):
    LineupFeatureBuilder = None
    get_expected_lineup_impact = None

__all__ = [
    # Team and game features
    "TeamFeatureBuilder",
    "GameFeatureBuilder",
    # Player rolling stats
    "PlayerFeatureBuilder",
    "PlayerStatsClient",
    # Matchup features
    "MatchupFeatureBuilder",
    "MatchupStats",
    "MatchupAdjuster",
    # Player impact (RAPM)
    "PlayerImpactModel",
    "PlayerImpact",
    "StintDataFetcher",
    "build_player_impact_model",
    # Lineup features
    "LineupFeatureBuilder",
    "get_expected_lineup_impact",
]
