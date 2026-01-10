"""
Weekly Strategy Report Generator

Analyzes weekly performance and generates strategy review reports.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from pathlib import Path
from loguru import logger


class WeeklyReportGenerator:
    """Generate weekly strategy performance reports."""

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize weekly report generator.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path

    def get_weekly_data(self, weeks_back: int = 1) -> pd.DataFrame:
        """
        Get betting data for the past N weeks.

        Args:
            weeks_back: Number of weeks to look back

        Returns:
            DataFrame with betting data
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=weeks_back * 7)

            query = """
                SELECT
                    id as bet_id,
                    logged_at as timestamp,
                    game_id,
                    home_team,
                    away_team,
                    bet_side,
                    bet_type,
                    line,
                    odds,
                    bet_amount,
                    model_prob,
                    market_prob,
                    edge,
                    outcome,
                    profit,
                    settled_at as settlement_timestamp,
                    bookmaker,
                    clv,
                    home_b2b,
                    away_b2b,
                    rest_advantage
                FROM bets
                WHERE DATE(logged_at) >= ? AND DATE(logged_at) <= ?
                ORDER BY logged_at DESC
            """

            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            return df
        except Exception as e:
            logger.error(f"Error getting weekly data: {e}")
            return pd.DataFrame()

    def analyze_strategy_performance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze performance by strategy.

        Args:
            df: DataFrame with betting data

        Returns:
            Strategy performance metrics
        """
        if df.empty:
            return {}

        # Filter to settled bets only
        settled = df[df['outcome'].notna()].copy()

        if settled.empty:
            return {
                'overall': {
                    'total_bets': len(df),
                    'settled_bets': 0,
                    'pending_bets': len(df)
                }
            }

        # Overall metrics
        overall = {
            'total_bets': len(df),
            'settled_bets': len(settled),
            'pending_bets': len(df) - len(settled),
            'total_wagered': settled['bet_amount'].sum(),
            'total_profit': settled['profit'].sum(),
            'roi': (settled['profit'].sum() / settled['bet_amount'].sum() * 100) if settled['bet_amount'].sum() > 0 else 0,
            'win_rate': (settled['outcome'] == 'win').sum() / len(settled) * 100,
            'avg_edge': settled['edge'].mean() * 100,
            'avg_clv': settled['clv'].mean() if 'clv' in settled.columns and settled['clv'].notna().any() else 0,
        }

        # By strategy (if available - not in current schema)
        by_strategy = {}
        # Note: strategy column not currently in database schema
        # Could be added later or inferred from bet characteristics

        # By bet type
        by_bet_type = {}
        for bet_type in settled['bet_type'].unique():
            if pd.isna(bet_type):
                continue

            type_data = settled[settled['bet_type'] == bet_type]
            by_bet_type[bet_type] = {
                'bets': len(type_data),
                'win_rate': (type_data['outcome'] == 'win').sum() / len(type_data) * 100,
                'profit': type_data['profit'].sum(),
                'roi': (type_data['profit'].sum() / type_data['bet_amount'].sum() * 100) if type_data['bet_amount'].sum() > 0 else 0,
            }

        # By situation (back-to-back, rest advantage)
        situations = {}
        if 'home_b2b' in settled.columns:
            # Home team on back-to-back
            home_b2b = settled[(settled['home_b2b'] == 1) & (settled['bet_side'] == 'HOME')]
            if len(home_b2b) > 0:
                situations['home_b2b'] = {
                    'bets': len(home_b2b),
                    'win_rate': (home_b2b['outcome'] == 'win').sum() / len(home_b2b) * 100,
                    'roi': (home_b2b['profit'].sum() / home_b2b['bet_amount'].sum() * 100) if home_b2b['bet_amount'].sum() > 0 else 0,
                }

            # Away team on back-to-back
            away_b2b = settled[(settled['away_b2b'] == 1) & (settled['bet_side'] == 'AWAY')]
            if len(away_b2b) > 0:
                situations['away_b2b'] = {
                    'bets': len(away_b2b),
                    'win_rate': (away_b2b['outcome'] == 'win').sum() / len(away_b2b) * 100,
                    'roi': (away_b2b['profit'].sum() / away_b2b['bet_amount'].sum() * 100) if away_b2b['bet_amount'].sum() > 0 else 0,
                }

        return {
            'overall': overall,
            'by_strategy': by_strategy,
            'by_bet_type': by_bet_type,
            'by_situation': situations,
        }

    def get_best_worst_teams(self, df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Get best and worst performing teams.

        Args:
            df: DataFrame with betting data
            top_n: Number of top/bottom teams to return

        Returns:
            Dict with best/worst teams
        """
        settled = df[df['outcome'].notna()].copy()

        if settled.empty:
            return {'best': [], 'worst': []}

        # Betting FOR teams
        for_teams = []
        for bet_side in ['HOME', 'AWAY']:
            team_col = 'home_team' if bet_side == 'HOME' else 'away_team'
            side_data = settled[settled['bet_side'] == bet_side].copy()

            if not side_data.empty:
                team_stats = side_data.groupby(team_col).agg({
                    'profit': 'sum',
                    'bet_amount': 'sum',
                    'outcome': lambda x: (x == 'win').sum() / len(x) * 100
                }).reset_index()
                team_stats.columns = ['team', 'profit', 'wagered', 'win_rate']
                team_stats['roi'] = (team_stats['profit'] / team_stats['wagered'] * 100)
                for_teams.append(team_stats)

        if for_teams:
            all_teams = pd.concat(for_teams).groupby('team').agg({
                'profit': 'sum',
                'wagered': 'sum',
                'win_rate': 'mean'
            }).reset_index()
            all_teams['roi'] = (all_teams['profit'] / all_teams['wagered'] * 100)
            all_teams = all_teams.sort_values('profit', ascending=False)

            best = all_teams.head(top_n)[['team', 'profit', 'roi', 'win_rate']].to_dict('records')
            worst = all_teams.tail(top_n)[['team', 'profit', 'roi', 'win_rate']].to_dict('records')

            return {'best': best, 'worst': worst}

        return {'best': [], 'worst': []}

    def generate_report(self, weeks_back: int = 1) -> Dict:
        """
        Generate complete weekly report.

        Args:
            weeks_back: Number of weeks to analyze

        Returns:
            Complete report dict
        """
        logger.info(f"Generating weekly report for past {weeks_back} week(s)...")

        df = self.get_weekly_data(weeks_back)

        if df.empty:
            return {
                'error': 'No data available for the specified period',
                'period': f"Last {weeks_back} week(s)"
            }

        performance = self.analyze_strategy_performance(df)
        teams = self.get_best_worst_teams(df)

        report = {
            'period': f"Last {weeks_back} week(s)",
            'start_date': df['timestamp'].min(),
            'end_date': df['timestamp'].max(),
            'performance': performance,
            'top_teams': teams['best'],
            'bottom_teams': teams['worst'],
            'generated_at': datetime.now().isoformat(),
        }

        logger.info("Weekly report generated successfully")
        return report

    def format_report_text(self, report: Dict) -> str:
        """
        Format report as human-readable text.

        Args:
            report: Report dictionary

        Returns:
            Formatted text report
        """
        if 'error' in report:
            return f"Error: {report['error']}"

        lines = [
            "=" * 80,
            f"📊 WEEKLY STRATEGY REVIEW - {report['period']}",
            "=" * 80,
            ""
        ]

        # Overall performance
        overall = report['performance']['overall']
        lines.extend([
            "📈 OVERALL PERFORMANCE",
            "-" * 80,
            f"Total Bets: {overall['total_bets']} ({overall['settled_bets']} settled, {overall['pending_bets']} pending)",
            f"Total Wagered: ${overall['total_wagered']:,.2f}",
            f"Total Profit: ${overall['total_profit']:+,.2f}",
            f"ROI: {overall['roi']:+.2f}%",
            f"Win Rate: {overall['win_rate']:.1f}%",
            f"Avg Edge: {overall['avg_edge']:+.2f}%",
            f"Avg CLV: {overall['avg_clv']:+.2f}%",
            ""
        ])

        # By strategy
        if report['performance']['by_strategy']:
            lines.extend([
                "🎯 PERFORMANCE BY STRATEGY",
                "-" * 80
            ])
            for strategy, metrics in report['performance']['by_strategy'].items():
                lines.append(f"\n{strategy}:")
                lines.append(f"  Bets: {metrics['bets']} | Win Rate: {metrics['win_rate']:.1f}% | ROI: {metrics['roi']:+.2f}%")
                lines.append(f"  Wagered: ${metrics['wagered']:,.2f} | Profit: ${metrics['profit']:+,.2f}")
            lines.append("")

        # Best teams
        if report['top_teams']:
            lines.extend([
                "⭐ TOP PERFORMING TEAMS",
                "-" * 80
            ])
            for team in report['top_teams']:
                lines.append(f"{team['team']}: ${team['profit']:+,.2f} ({team['roi']:+.1f}% ROI, {team['win_rate']:.0f}% WR)")
            lines.append("")

        # Worst teams
        if report['bottom_teams']:
            lines.extend([
                "⚠️  WORST PERFORMING TEAMS",
                "-" * 80
            ])
            for team in report['bottom_teams']:
                lines.append(f"{team['team']}: ${team['profit']:+,.2f} ({team['roi']:+.1f}% ROI, {team['win_rate']:.0f}% WR)")
            lines.append("")

        lines.extend([
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])

        return "\n".join(lines)
