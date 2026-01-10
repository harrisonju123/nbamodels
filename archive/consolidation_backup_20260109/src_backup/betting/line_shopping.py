"""
Line Shopping Module

Compares odds across multiple bookmakers to find the best available lines.
Can add 1-2% to ROI by getting optimal odds for each bet.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class LineShoppingEngine:
    """
    Engine for comparing odds across multiple bookmakers.

    Finds the best available line for each bet opportunity.
    """

    # Bookmaker rankings (higher = more reliable/better limits)
    BOOKMAKER_TIER = {
        'draftkings': 1,
        'fanduel': 1,
        'betmgm': 1,
        'caesars': 2,
        'pointsbet': 2,
        'betrivers': 2,
        'wynnbet': 3,
        'unibet': 3,
    }

    def __init__(self, min_bookmakers: int = 2):
        """
        Initialize line shopping engine.

        Args:
            min_bookmakers: Minimum number of bookmakers required for comparison
        """
        self.min_bookmakers = min_bookmakers

    def find_best_spread_odds(
        self,
        odds_df: pd.DataFrame,
        game_id: str,
        side: str,
        target_line: Optional[float] = None
    ) -> Dict:
        """
        Find the best spread odds for a given game and side.

        Args:
            odds_df: DataFrame with odds from multiple bookmakers
            game_id: Game identifier
            side: 'home' or 'away'
            target_line: Specific line to target (optional)

        Returns:
            Dict with best odds info:
            {
                'bookmaker': str,
                'line': float,
                'odds': int,
                'tier': int,
                'alternatives': List[Dict],
                'edge_vs_worst': float,
                'total_books': int
            }
        """
        # Filter for this game and market
        game_odds = odds_df[
            (odds_df['game_id'] == game_id) &
            (odds_df['market'] == 'spread') &
            (odds_df['team'] == side)
        ].copy()

        if len(game_odds) == 0:
            logger.warning(f"No spread odds found for {game_id} {side}")
            return None

        # If target line specified, filter to that line or closest
        if target_line is not None:
            # Find odds at or near target line (within 0.5 points)
            game_odds['line_diff'] = abs(game_odds['line'] - target_line)
            game_odds = game_odds[game_odds['line_diff'] <= 0.5]

            if len(game_odds) == 0:
                logger.warning(f"No odds found at target line {target_line}")
                return None

        # Add bookmaker tier
        game_odds['tier'] = game_odds['bookmaker'].map(self.BOOKMAKER_TIER).fillna(3)

        # Sort by odds (higher is better for American odds when negative)
        # For negative odds: -105 is better than -110
        # For positive odds: +150 is better than +140
        game_odds = game_odds.sort_values('odds', ascending=False)

        # Get best odds
        best = game_odds.iloc[0]

        # Get worst odds for comparison
        worst_odds = game_odds['odds'].min()

        # Calculate edge vs worst line (in terms of implied probability)
        best_prob = self._american_to_implied_prob(best['odds'])
        worst_prob = self._american_to_implied_prob(worst_odds)
        edge_vs_worst = worst_prob - best_prob  # Lower prob = better odds

        # Get alternative options
        alternatives = []
        for idx, row in game_odds.iloc[1:min(4, len(game_odds))].iterrows():
            alternatives.append({
                'bookmaker': row['bookmaker'],
                'line': row['line'],
                'odds': row['odds'],
                'tier': row['tier'],
            })

        return {
            'bookmaker': best['bookmaker'],
            'line': best['line'],
            'odds': best['odds'],
            'tier': best['tier'],
            'alternatives': alternatives,
            'edge_vs_worst': edge_vs_worst,
            'total_books': len(game_odds),
        }

    def find_best_moneyline_odds(
        self,
        odds_df: pd.DataFrame,
        game_id: str,
        side: str
    ) -> Dict:
        """
        Find the best moneyline odds for a given game and side.

        Args:
            odds_df: DataFrame with odds from multiple bookmakers
            game_id: Game identifier
            side: 'home' or 'away'

        Returns:
            Dict with best odds info
        """
        # Filter for this game and market
        game_odds = odds_df[
            (odds_df['game_id'] == game_id) &
            (odds_df['market'] == 'moneyline') &
            (odds_df['team'] == side)
        ].copy()

        if len(game_odds) == 0:
            logger.warning(f"No moneyline odds found for {game_id} {side}")
            return None

        # Add bookmaker tier
        game_odds['tier'] = game_odds['bookmaker'].map(self.BOOKMAKER_TIER).fillna(3)

        # Sort by odds (higher is better)
        game_odds = game_odds.sort_values('odds', ascending=False)

        # Get best odds
        best = game_odds.iloc[0]
        worst_odds = game_odds['odds'].min()

        # Calculate edge
        best_prob = self._american_to_implied_prob(best['odds'])
        worst_prob = self._american_to_implied_prob(worst_odds)
        edge_vs_worst = worst_prob - best_prob

        # Get alternatives
        alternatives = []
        for idx, row in game_odds.iloc[1:min(4, len(game_odds))].iterrows():
            alternatives.append({
                'bookmaker': row['bookmaker'],
                'odds': row['odds'],
                'tier': row['tier'],
            })

        return {
            'bookmaker': best['bookmaker'],
            'odds': best['odds'],
            'tier': best['tier'],
            'alternatives': alternatives,
            'edge_vs_worst': edge_vs_worst,
            'total_books': len(game_odds),
        }

    def find_best_total_odds(
        self,
        odds_df: pd.DataFrame,
        game_id: str,
        side: str,  # 'over' or 'under'
        target_line: Optional[float] = None
    ) -> Dict:
        """
        Find the best totals (over/under) odds.

        Args:
            odds_df: DataFrame with odds from multiple bookmakers
            game_id: Game identifier
            side: 'over' or 'under'
            target_line: Specific total to target (optional)

        Returns:
            Dict with best odds info
        """
        # Filter for this game and market
        game_odds = odds_df[
            (odds_df['game_id'] == game_id) &
            (odds_df['market'] == 'total') &
            (odds_df['team'] == side)
        ].copy()

        if len(game_odds) == 0:
            logger.warning(f"No total odds found for {game_id} {side}")
            return None

        # If target line specified, filter
        if target_line is not None:
            game_odds['line_diff'] = abs(game_odds['line'] - target_line)
            game_odds = game_odds[game_odds['line_diff'] <= 0.5]

            if len(game_odds) == 0:
                logger.warning(f"No odds found at target total {target_line}")
                return None

        # Add bookmaker tier
        game_odds['tier'] = game_odds['bookmaker'].map(self.BOOKMAKER_TIER).fillna(3)

        # Sort by odds
        game_odds = game_odds.sort_values('odds', ascending=False)

        # Get best odds
        best = game_odds.iloc[0]
        worst_odds = game_odds['odds'].min()

        # Calculate edge
        best_prob = self._american_to_implied_prob(best['odds'])
        worst_prob = self._american_to_implied_prob(worst_odds)
        edge_vs_worst = worst_prob - best_prob

        # Get alternatives
        alternatives = []
        for idx, row in game_odds.iloc[1:min(4, len(game_odds))].iterrows():
            alternatives.append({
                'bookmaker': row['bookmaker'],
                'line': row['line'],
                'odds': row['odds'],
                'tier': row['tier'],
            })

        return {
            'bookmaker': best['bookmaker'],
            'line': best['line'],
            'odds': best['odds'],
            'tier': best['tier'],
            'alternatives': alternatives,
            'edge_vs_worst': edge_vs_worst,
            'total_books': len(game_odds),
        }

    def compare_all_books(
        self,
        odds_df: pd.DataFrame,
        game_id: str,
        market: str = 'spread',
        side: str = 'home'
    ) -> pd.DataFrame:
        """
        Get comparison table of all available bookmakers for a specific bet.

        Args:
            odds_df: DataFrame with odds from multiple bookmakers
            game_id: Game identifier
            market: 'spread', 'moneyline', or 'total'
            side: Which side to compare

        Returns:
            DataFrame with all bookmaker odds sorted by value
        """
        # Filter for this game and market
        comparison = odds_df[
            (odds_df['game_id'] == game_id) &
            (odds_df['market'] == market) &
            (odds_df['team'] == side)
        ].copy()

        if len(comparison) == 0:
            return pd.DataFrame()

        # Add tier
        comparison['tier'] = comparison['bookmaker'].map(self.BOOKMAKER_TIER).fillna(3)

        # Add implied probability
        comparison['implied_prob'] = comparison['odds'].apply(self._american_to_implied_prob)

        # Calculate potential profit on $100 bet
        comparison['profit_on_100'] = comparison['odds'].apply(
            lambda x: 100 * (x / 100) if x > 0 else 100 * (100 / abs(x))
        )

        # Sort by odds (best first)
        comparison = comparison.sort_values('odds', ascending=False)

        # Select relevant columns
        cols = ['bookmaker', 'line', 'odds', 'implied_prob', 'profit_on_100', 'tier', 'last_update']
        if market in ['spread', 'total']:
            result = comparison[cols]
        else:
            # No line for moneyline
            result = comparison[[c for c in cols if c != 'line']]

        return result

    def calculate_line_shopping_value(
        self,
        odds_df: pd.DataFrame,
        game_id: str,
        market: str,
        side: str
    ) -> Dict:
        """
        Calculate the value gained from line shopping vs using single book.

        Args:
            odds_df: DataFrame with odds from multiple bookmakers
            game_id: Game identifier
            market: Market type
            side: Which side

        Returns:
            Dict with value metrics
        """
        game_odds = odds_df[
            (odds_df['game_id'] == game_id) &
            (odds_df['market'] == market) &
            (odds_df['team'] == side)
        ].copy()

        if len(game_odds) < 2:
            return {'value': 0, 'message': 'Not enough bookmakers for comparison'}

        best_odds = game_odds['odds'].max()
        worst_odds = game_odds['odds'].min()
        avg_odds = game_odds['odds'].mean()

        # Convert to implied probabilities
        best_prob = self._american_to_implied_prob(best_odds)
        worst_prob = self._american_to_implied_prob(worst_odds)
        avg_prob = self._american_to_implied_prob(avg_odds)

        # Value = probability saved by shopping
        value_vs_worst = worst_prob - best_prob
        value_vs_avg = avg_prob - best_prob

        return {
            'best_odds': best_odds,
            'worst_odds': worst_odds,
            'avg_odds': avg_odds,
            'value_vs_worst': value_vs_worst,
            'value_vs_avg': value_vs_avg,
            'spread': abs(best_odds - worst_odds),
            'num_books': len(game_odds),
        }

    @staticmethod
    def _american_to_implied_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


def get_line_shopping_summary(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary of line shopping opportunities across all games.

    Args:
        odds_df: DataFrame with odds from multiple bookmakers

    Returns:
        Summary DataFrame with line shopping value for each game
    """
    engine = LineShoppingEngine()

    summaries = []

    # Get unique games
    games = odds_df['game_id'].unique()

    for game_id in games:
        game_data = odds_df[odds_df['game_id'] == game_id].iloc[0]

        # Check spread line shopping value for both sides
        for side in ['home', 'away']:
            spread_value = engine.calculate_line_shopping_value(
                odds_df, game_id, 'spread', side
            )

            if spread_value.get('num_books', 0) >= 2:
                summaries.append({
                    'game_id': game_id,
                    'home_team': game_data['home_team'],
                    'away_team': game_data['away_team'],
                    'market': 'spread',
                    'side': side,
                    'best_odds': spread_value['best_odds'],
                    'worst_odds': spread_value['worst_odds'],
                    'value_vs_worst': spread_value['value_vs_worst'],
                    'num_books': spread_value['num_books'],
                })

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Test the line shopping engine
    print("Line Shopping Engine - Test Mode")
    print("=" * 70)
    print()
    print("This module compares odds across bookmakers to find best lines.")
    print()
    print("Features:")
    print("  - Find best spread odds")
    print("  - Find best moneyline odds")
    print("  - Find best totals odds")
    print("  - Compare all bookmakers")
    print("  - Calculate line shopping value")
    print()
    print("Expected ROI improvement: +1-2% from better odds")
