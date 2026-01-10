"""Rebounds Prop Model - Predicts player total rebounds."""

from typing import List
from src.models.player_props.base_prop_model import BasePlayerPropModel


class ReboundsPropModel(BasePlayerPropModel):
    """Model for predicting player rebounds."""

    prop_type = "REB"

    def get_required_features(self) -> List[str]:
        """
        Features for rebounds prediction.

        Key indicators:
        - Recent rebounding averages
        - Minutes played
        - Opponent rebounding stats
        - Team pace (more possessions = more rebounds)
        - Position/role indicators
        """
        return [
            # Rolling averages
            "reb_roll3", "reb_roll5", "reb_roll10",
            "oreb_roll5", "dreb_roll5",
            "min_roll5", "min_roll10",

            # Matchup features
            "opp_reb_allowed_roll5",
            "opp_pace",

            # Team context
            "team_pace",
            "team_reb_roll5",

            # Context
            "is_home",
            "days_rest",
        ]
