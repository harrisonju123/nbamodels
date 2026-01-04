"""
Input validation utilities for NBA betting models.

Provides validation functions for player stats, spreads, probabilities, and other inputs.
"""

from typing import Optional
import pandas as pd
from loguru import logger

from src.utils.constants import (
    MAX_PPG, MAX_RPG, MAX_APG, MAX_MPG,
    MIN_PPG, MIN_RPG, MIN_APG, MIN_MPG,
    MAX_REASONABLE_SPREAD, MIN_REASONABLE_SPREAD,
    MAX_PROBABILITY, MIN_PROBABILITY,
    VALID_TEAM_ABBREVS,
)


def validate_player_stats(stats: dict, player_name: str = "Unknown") -> dict:
    """
    Validate and clip player statistics to reasonable bounds.

    Args:
        stats: Dictionary with player stats (ppg, rpg, apg, mpg, etc.)
        player_name: Player name for logging

    Returns:
        Validated stats dictionary with clipped values
    """
    validated = stats.copy()

    # Validate PPG
    if 'ppg' in validated:
        ppg = validated['ppg']
        if ppg is not None:
            if ppg < MIN_PPG or ppg > MAX_PPG:
                logger.warning(f"{player_name}: PPG {ppg} out of bounds, clipping to [{MIN_PPG}, {MAX_PPG}]")
                validated['ppg'] = max(MIN_PPG, min(ppg, MAX_PPG))

    # Validate RPG
    if 'rpg' in validated:
        rpg = validated['rpg']
        if rpg is not None:
            if rpg < MIN_RPG or rpg > MAX_RPG:
                logger.warning(f"{player_name}: RPG {rpg} out of bounds, clipping to [{MIN_RPG}, {MAX_RPG}]")
                validated['rpg'] = max(MIN_RPG, min(rpg, MAX_RPG))

    # Validate APG
    if 'apg' in validated:
        apg = validated['apg']
        if apg is not None:
            if apg < MIN_APG or apg > MAX_APG:
                logger.warning(f"{player_name}: APG {apg} out of bounds, clipping to [{MIN_APG}, {MAX_APG}]")
                validated['apg'] = max(MIN_APG, min(apg, MAX_APG))

    # Validate MPG
    if 'mpg' in validated:
        mpg = validated['mpg']
        if mpg is not None:
            if mpg < MIN_MPG or mpg > MAX_MPG:
                logger.warning(f"{player_name}: MPG {mpg} out of bounds, clipping to [{MIN_MPG}, {MAX_MPG}]")
                validated['mpg'] = max(MIN_MPG, min(mpg, MAX_MPG))

    return validated


def validate_spread(spread: float, game_id: str = "Unknown") -> float:
    """
    Validate spread is within reasonable bounds.

    Args:
        spread: Point spread value
        game_id: Game identifier for logging

    Returns:
        Validated spread value
    """
    if spread < MIN_REASONABLE_SPREAD or spread > MAX_REASONABLE_SPREAD:
        logger.warning(
            f"Game {game_id}: Spread {spread} out of reasonable bounds "
            f"[{MIN_REASONABLE_SPREAD}, {MAX_REASONABLE_SPREAD}]"
        )
        return max(MIN_REASONABLE_SPREAD, min(spread, MAX_REASONABLE_SPREAD))
    return spread


def validate_probability(prob: float, context: str = "") -> float:
    """
    Validate probability is in valid range.

    Args:
        prob: Probability value
        context: Context string for logging

    Returns:
        Clipped probability value
    """
    if prob < MIN_PROBABILITY or prob > MAX_PROBABILITY:
        logger.warning(
            f"{context}: Probability {prob} out of bounds [{MIN_PROBABILITY}, {MAX_PROBABILITY}], clipping"
        )
        return max(MIN_PROBABILITY, min(prob, MAX_PROBABILITY))
    return prob


def validate_team_abbrev(team: str) -> bool:
    """
    Check if team abbreviation is valid.

    Args:
        team: Team abbreviation

    Returns:
        True if valid, False otherwise
    """
    if team not in VALID_TEAM_ABBREVS:
        logger.warning(f"Invalid team abbreviation: {team}")
        return False
    return True


def validate_api_response(response: dict, required_fields: list, source: str = "API") -> bool:
    """
    Validate API response has required fields.

    Args:
        response: API response dictionary
        required_fields: List of required field names
        source: API source name for logging

    Returns:
        True if all required fields present, False otherwise
    """
    missing_fields = [field for field in required_fields if field not in response]

    if missing_fields:
        logger.error(f"{source} response missing required fields: {missing_fields}")
        return False

    return True


def validate_dataframe_columns(df: pd.DataFrame, required_columns: list, context: str = "") -> bool:
    """
    Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        context: Context string for logging

    Returns:
        True if all required columns present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"{context}: DataFrame missing required columns: {missing_columns}")
        return False

    return True


def validate_odds(odds: int, context: str = "") -> Optional[int]:
    """
    Validate American odds format.

    Args:
        odds: American odds (e.g., -110, +150)
        context: Context string for logging

    Returns:
        Odds if valid, None otherwise
    """
    # American odds should never be between -100 and +100 (except exactly ±100)
    if -100 < odds < 100 and odds != 0:
        logger.warning(f"{context}: Invalid American odds {odds} (must be <= -100 or >= +100)")
        return None

    # Extremely large odds are suspicious
    if abs(odds) > 10000:
        logger.warning(f"{context}: Suspicious odds value {odds}")
        return None

    return odds
