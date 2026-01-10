"""
Generate Realistic Synthetic Backtest Data

Creates synthetic betting data based on real historical patterns:
- Edge distribution matching actual bets
- Odds distribution from real markets
- Team matchup diversity
- Realistic win/loss outcomes
"""

import sys
sys.path.insert(0, '.')

import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.betting.rigorous_backtest.core import RigorousBet


# NBA teams
NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]


def analyze_real_patterns(db_path: str = "data/bets/bets.db"):
    """Analyze real betting patterns from database."""
    logger.info(f"Analyzing real betting patterns from {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get edge distribution
    cursor.execute("""
        SELECT edge, odds, model_prob, outcome
        FROM bets
        WHERE outcome IS NOT NULL AND edge IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.warning("No data found, using defaults")
        return {
            'edge_mean': 0.08,
            'edge_std': 0.03,
            'odds_mean': -105,
            'odds_std': 5,
            'win_rate': 0.56
        }

    edges = [r[0] for r in rows]
    odds = [r[1] for r in rows]
    win_count = sum(1 for r in rows if r[3] == 'win')

    patterns = {
        'edge_mean': np.mean(edges),
        'edge_std': np.std(edges),
        'edge_min': np.min(edges),
        'edge_max': np.max(edges),
        'odds_mean': np.mean(odds),
        'odds_std': np.std(odds),
        'win_rate': win_count / len(rows) if rows else 0.56
    }

    logger.info(f"Patterns analyzed: {len(rows)} bets")
    logger.info(f"  Edge: {patterns['edge_mean']:.3f} ± {patterns['edge_std']:.3f}")
    logger.info(f"  Odds: {patterns['odds_mean']:.1f} ± {patterns['odds_std']:.1f}")
    logger.info(f"  Win Rate: {patterns['win_rate']:.1%}")

    return patterns


def generate_synthetic_bets(
    num_bets: int = 1000,
    patterns: dict = None,
    start_date: datetime = None,
    spread_days: int = 180
) -> list:
    """
    Generate synthetic bets based on real patterns.

    Args:
        num_bets: Number of bets to generate
        patterns: Pattern dict from analyze_real_patterns
        start_date: Start date for bets
        spread_days: Number of days to spread bets across

    Returns:
        List of RigorousBet objects
    """
    if patterns is None:
        patterns = analyze_real_patterns()

    if start_date is None:
        start_date = datetime(2024, 10, 1)  # Start of NBA season

    logger.info(f"Generating {num_bets} synthetic bets...")

    bets = []

    for i in range(num_bets):
        # Random date within spread
        days_offset = random.randint(0, spread_days)
        bet_date = start_date + timedelta(days=days_offset)

        # Random matchup
        home_team = random.choice(NBA_TEAMS)
        away_team = random.choice([t for t in NBA_TEAMS if t != home_team])

        # Generate edge from normal distribution, clipped to realistic range
        edge = np.random.normal(patterns['edge_mean'], patterns['edge_std'])
        edge = np.clip(edge, 0.02, 0.20)  # 2% to 20% edge

        # Generate odds from normal distribution (American format)
        american_odds = np.random.normal(patterns['odds_mean'], patterns['odds_std'])
        american_odds = np.clip(american_odds, -150, -100)  # Typical spread odds range

        # Calculate market prob from American odds
        if american_odds < 0:
            market_prob = abs(american_odds) / (abs(american_odds) + 100)
        else:
            market_prob = 100 / (american_odds + 100)

        # Convert American odds to decimal for backtest compatibility
        if american_odds < 0:
            odds = 1 + (100 / abs(american_odds))
        else:
            odds = 1 + (american_odds / 100)

        # Model prob = market prob + edge
        model_prob = market_prob + edge
        model_prob = np.clip(model_prob, 0.51, 0.75)  # Realistic range

        # Random bet side (HOME or AWAY for spreads)
        bet_side = random.choice(['HOME', 'AWAY'])

        # Random spread line
        line = random.choice([-12.5, -10.5, -8.5, -6.5, -4.5, -2.5, -1.5,
                             1.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5])

        # Generate game_id
        game_id = f"{bet_date.strftime('%Y%m%d')}_{home_team.replace(' ', '')[:3]}_{away_team.replace(' ', '')[:3]}"

        # Placeholder bet size (will be recalculated by Kelly in backtest)
        bet_size = 100.0

        # Generate realistic outcome based on model_prob
        won = np.random.random() < model_prob

        bet = RigorousBet(
            date=bet_date,
            game_id=game_id,
            bet_side=bet_side,
            bet_type='spread',
            bet_size=bet_size,
            odds=odds,
            line=line,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            won=won
        )

        bets.append(bet)

    # Sort chronologically
    bets.sort(key=lambda b: b.date)

    logger.info(f"Generated {len(bets)} synthetic bets")
    logger.info(f"  Date range: {bets[0].date.date()} to {bets[-1].date.date()}")
    logger.info(f"  Avg edge: {np.mean([b.edge for b in bets]):.3f}")

    return bets


def main():
    """Generate and save synthetic data."""
    # Analyze real patterns
    patterns = analyze_real_patterns()

    # Generate 1000 bets
    bets = generate_synthetic_bets(
        num_bets=1000,
        patterns=patterns,
        start_date=datetime(2024, 10, 1),
        spread_days=180
    )

    # Save to file
    output_path = Path("data/backtest/synthetic_bets_realistic.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(bets, f)

    logger.info(f"Saved {len(bets)} synthetic bets to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SYNTHETIC DATASET SUMMARY")
    print("="*80)
    print(f"Total Bets: {len(bets)}")
    print(f"Date Range: {bets[0].date.date()} to {bets[-1].date.date()}")
    print(f"Unique Teams: {len(set([b.game_id.split('_')[1] for b in bets]))}")
    print(f"Edge Range: {min(b.edge for b in bets):.3f} to {max(b.edge for b in bets):.3f}")
    print(f"Avg Edge: {np.mean([b.edge for b in bets]):.3f}")
    print(f"Avg Odds: {np.mean([b.odds for b in bets]):.1f}")
    print("="*80)


if __name__ == '__main__':
    main()
