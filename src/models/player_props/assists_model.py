"""Assists Prop Model - Predicts player assists."""

from typing import List
from src.models.player_props.base_prop_model import BasePlayerPropModel


class AssistsPropModel(BasePlayerPropModel):
    """Model for predicting player assists."""

    prop_type = "AST"

    def get_required_features(self) -> List[str]:
        """
        Features for assists prediction.

        Key indicators:
        - Recent assist averages
        - Usage rate (ball handlers get more assists)
        - Team pace (more possessions = more opportunities)
        - Opponent perimeter defense
        """
        return [
            # Rolling averages
            "ast_roll3", "ast_roll5", "ast_roll10",
            "min_roll5", "min_roll10",

            # Usage and ball handling
            "usage_rate",
            "tov_roll5",  # Turnover rate (related to ball handling)

            # Matchup features
            "opp_def_rating",
            "opp_pace",

            # Team context
            "team_pace",
            "team_ast_roll5",

            # Context
            "is_home",
            "days_rest",
        ]
