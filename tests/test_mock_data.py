"""
Mock Data Generators for Testing

Creates realistic mock data for testing strategies without requiring real API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_games(n_games: int = 5) -> pd.DataFrame:
    """Generate mock games DataFrame."""
    teams = ['LAL', 'BOS', 'GSW', 'MIL', 'PHX', 'DEN', 'MIA', 'NYK']

    games = []
    for i in range(n_games):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        games.append({
            'game_id': f'mock_game_{i}',
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': datetime.now() + timedelta(hours=i*2),
            'home_b2b': np.random.choice([True, False], p=[0.2, 0.8]),
            'away_b2b': np.random.choice([True, False], p=[0.2, 0.8]),
        })

    df = pd.DataFrame(games)
    # Ensure required columns exist
    if 'home_b2b' not in df.columns:
        df['home_b2b'] = False
    if 'away_b2b' not in df.columns:
        df['away_b2b'] = False

    return df


def generate_mock_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock features for games."""
    features = []

    for _, game in games_df.iterrows():
        features.append({
            'game_id': game['game_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            # Team stats
            'home_pts_for_5g': np.random.uniform(108, 118),
            'away_pts_for_5g': np.random.uniform(108, 118),
            'home_pace': np.random.uniform(95, 103),
            'away_pace': np.random.uniform(95, 103),
            'home_def_rating': np.random.uniform(108, 115),
            'away_def_rating': np.random.uniform(108, 115),
            # Elo
            'home_elo': np.random.uniform(1450, 1600),
            'away_elo': np.random.uniform(1450, 1600),
            'elo_diff': np.random.uniform(-100, 100),
            # Context
            'rest_advantage': np.random.randint(-2, 3),
            'pace': np.random.uniform(96, 102),
        })

    return pd.DataFrame(features)


def generate_mock_totals_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock totals odds."""
    odds = []

    for _, game in games_df.iterrows():
        total_line = np.random.uniform(215, 230)

        odds.append({
            'game_id': game['game_id'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'total_line': total_line,
            'over_odds': np.random.choice([-115, -110, -105, +100]),
            'under_odds': np.random.choice([-115, -110, -105, +100]),
            'bookmaker': np.random.choice(['draftkings', 'fanduel', 'betmgm']),
        })

    return pd.DataFrame(odds)


def generate_mock_spread_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock spread odds for arbitrage testing."""
    odds = []
    bookmakers = ['draftkings', 'fanduel', 'betmgm', 'caesars']

    for _, game in games_df.iterrows():
        spread = np.random.uniform(-8, 8)

        for bookmaker in bookmakers:
            # Add some variance to create arbitrage opportunities
            spread_var = np.random.uniform(-0.5, 0.5)
            odds_var = np.random.randint(-5, 5)

            odds.append({
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'market': 'spread',
                'team': 'home',
                'line': spread + spread_var,
                'odds': -110 + odds_var,
                'bookmaker': bookmaker,
            })

            odds.append({
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'market': 'spread',
                'team': 'away',
                'line': -(spread + spread_var),
                'odds': -110 + odds_var,
                'bookmaker': bookmaker,
            })

    return pd.DataFrame(odds)


def generate_mock_live_games() -> dict:
    """Generate mock live game states."""
    return {
        'mock_live_game_1': {
            'game_id': 'mock_live_game_1',
            'home_team': 'LAL',
            'away_team': 'BOS',
            'home_score': 92,
            'away_score': 88,
            'quarter': 4,
            'time_remaining': '5:30',
        },
        'mock_live_game_2': {
            'game_id': 'mock_live_game_2',
            'home_team': 'GSW',
            'away_team': 'DEN',
            'home_score': 105,
            'away_score': 102,
            'quarter': 4,
            'time_remaining': '2:15',
        }
    }


def generate_mock_live_odds() -> pd.DataFrame:
    """Generate mock live odds."""
    return pd.DataFrame([
        {
            'game_id': 'mock_live_game_1',
            'spread_value': -3.5,
            'home_spread_odds': -110,
            'away_spread_odds': -110,
            'home_ml_odds': -180,
            'away_ml_odds': +155,
            'total_value': 220.5,
            'over_odds': -110,
            'under_odds': -110,
            'bookmaker': 'draftkings',
        },
        {
            'game_id': 'mock_live_game_2',
            'spread_value': -2.5,
            'home_spread_odds': -105,
            'away_spread_odds': -115,
            'home_ml_odds': -140,
            'away_ml_odds': +120,
            'total_value': 218.5,
            'over_odds': -115,
            'under_odds': -105,
            'bookmaker': 'fanduel',
        }
    ])


def generate_mock_player_props() -> pd.DataFrame:
    """Generate mock player prop odds."""
    players = [
        ('LeBron James', 'LAL', 'PTS', 27.5),
        ('Jayson Tatum', 'BOS', 'PTS', 26.5),
        ('Stephen Curry', 'GSW', 'PTS', 28.5),
        ('Nikola Jokic', 'DEN', 'REB', 11.5),
        ('Draymond Green', 'GSW', 'AST', 7.5),
    ]

    props = []
    for player_name, team, prop_type, line in players:
        # Over
        props.append({
            'game_id': 'mock_game_0',
            'player_name': player_name,
            'player_team': team,
            'prop_type': prop_type,
            'side': 'over',
            'line': line,
            'odds': np.random.choice([-120, -115, -110, -105]),
            'bookmaker': 'draftkings',
            'home_team': 'LAL',
            'away_team': 'BOS',
        })

        # Under
        props.append({
            'game_id': 'mock_game_0',
            'player_name': player_name,
            'player_team': team,
            'prop_type': prop_type,
            'side': 'under',
            'line': line,
            'odds': np.random.choice([-120, -115, -110, -105]),
            'bookmaker': 'draftkings',
            'home_team': 'LAL',
            'away_team': 'BOS',
        })

    return pd.DataFrame(props)


def generate_mock_player_features(props_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock player features."""
    features = []

    for player_name in props_df['player_name'].unique():
        features.append({
            'game_id': 'mock_game_0',
            'player_name': player_name,
            'pts_roll3': np.random.uniform(20, 30),
            'pts_roll5': np.random.uniform(20, 30),
            'pts_roll10': np.random.uniform(20, 30),
            'min_roll5': np.random.uniform(32, 38),
            'min_roll10': np.random.uniform(32, 38),
            'fga_roll5': np.random.uniform(18, 22),
            'fta_roll5': np.random.uniform(4, 8),
            'fg_pct_roll5': np.random.uniform(0.45, 0.52),
            'opp_def_rating': np.random.uniform(110, 115),
            'opp_pts_allowed_roll5': np.random.uniform(108, 115),
            'usage_rate': np.random.uniform(0.25, 0.32),
            'team_pace': np.random.uniform(98, 102),
            'is_home': 1,
            'days_rest': 1,
            # Rebounds
            'reb_roll3': np.random.uniform(8, 13),
            'reb_roll5': np.random.uniform(8, 13),
            'reb_roll10': np.random.uniform(8, 13),
            'oreb_roll5': np.random.uniform(2, 4),
            'dreb_roll5': np.random.uniform(6, 9),
            'opp_reb_allowed_roll5': np.random.uniform(42, 46),
            'opp_pace': np.random.uniform(98, 102),
            'team_reb_roll5': np.random.uniform(43, 47),
            # Assists
            'ast_roll3': np.random.uniform(5, 10),
            'ast_roll5': np.random.uniform(5, 10),
            'ast_roll10': np.random.uniform(5, 10),
            'tov_roll5': np.random.uniform(2, 4),
            'team_ast_roll5': np.random.uniform(23, 27),
            # Threes
            'fg3m_roll3': np.random.uniform(2, 5),
            'fg3m_roll5': np.random.uniform(2, 5),
            'fg3m_roll10': np.random.uniform(2, 5),
            'fg3a_roll5': np.random.uniform(6, 10),
            'fg3a_roll10': np.random.uniform(6, 10),
            'fg3_pct_roll5': np.random.uniform(0.35, 0.42),
            'fg3_pct_roll10': np.random.uniform(0.35, 0.42),
            'opp_3pt_defense': np.random.uniform(0.34, 0.38),
            'team_3pm_roll5': np.random.uniform(12, 15),
        })

    return pd.DataFrame(features)
