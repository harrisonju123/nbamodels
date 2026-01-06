"""Threes Prop Model - Predicts player 3-pointers made."""

from typing import List
from src.models.player_props.base_prop_model import BasePlayerPropModel


class ThreesPropModel(BasePlayerPropModel):
    """Model for predicting player 3-pointers made."""

    prop_type = "3PM"

    def get_required_features(self) -> List[str]:
        """
        Features for 3PM prediction.

        Key indicators:
        - Recent 3PM averages
        - 3-point attempts (volume)
        - 3-point shooting percentage
        - Opponent 3-point defense
        - Game pace
        """
        return [
            # Rolling averages
            "fg3m_roll3", "fg3m_roll5", "fg3m_roll10",
            "fg3a_roll5", "fg3a_roll10",
            "fg3_pct_roll5", "fg3_pct_roll10",
            "min_roll5", "min_roll10",

            # Matchup features
            "opp_3pt_defense",
            "opp_pace",

            # Team context
            "team_pace",
            "team_3pm_roll5",

            # Context
            "is_home",
            "days_rest",
        ]
