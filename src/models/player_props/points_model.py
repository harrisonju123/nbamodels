"""Points Prop Model - Predicts player points scored."""

from typing import List
from src.models.player_props.base_prop_model import BasePlayerPropModel


class PointsPropModel(BasePlayerPropModel):
    """Model for predicting player points."""

    prop_type = "PTS"

    def get_required_features(self) -> List[str]:
        """
        Features for points prediction.

        Key indicators:
        - Recent scoring averages (rolling 3/5/10 games)
        - Minutes played (correlated with opportunities)
        - Shot attempts and efficiency
        - Opponent defensive rating
        - Game pace and context
        """
        return [
            # Rolling averages
            "pts_roll3", "pts_roll5", "pts_roll10",
            "min_roll5", "min_roll10",

            # Shooting volume and efficiency
            "fga_roll5", "fga_roll10",
            "fta_roll5", "fta_roll10",
            "fg_pct_roll5",

            # Matchup features
            "opp_def_rating",
            "opp_pts_allowed_roll5",

            # Usage and pace
            "usage_rate",
            "team_pace",

            # Context
            "is_home",
            "days_rest",
        ]
