"""
Monthly Strategy Report Generator

Generates comprehensive investor-grade monthly performance reports
with advanced metrics like Sharpe ratio, max drawdown, and rolling statistics.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from .weekly_report import WeeklyReportGenerator


class MonthlyReportGenerator(WeeklyReportGenerator):
    """Generate monthly investor-grade performance reports."""

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize monthly report generator.

        Args:
            db_path: Path to bets database
        """
        super().__init__(db_path)

    def get_monthly_data(self, months_back: int = 1) -> pd.DataFrame:
        """
        Get betting data for the past N months.

        Args:
            months_back: Number of months to look back

        Returns:
            DataFrame with betting data
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=months_back * 30)

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
                ORDER BY logged_at ASC
            """

            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            # Convert timestamps to datetime (handle various formats)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                df['settlement_timestamp'] = pd.to_datetime(df['settlement_timestamp'], format='mixed', errors='coerce')

            return df
        except Exception as e:
            logger.error(f"Error getting monthly data: {e}")
            return pd.DataFrame()

    def calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio for betting performance.

        Args:
            df: DataFrame with settled bets
            risk_free_rate: Annual risk-free rate (default 0.0)

        Returns:
            Sharpe ratio
        """
        if df.empty or 'profit' not in df.columns:
            return 0.0

        # Filter to settled bets
        settled = df[df['outcome'].notna()].copy()

        if len(settled) < 2:
            return 0.0

        # Calculate returns as profit / bet_amount
        settled['returns'] = settled['profit'] / settled['bet_amount']

        # Calculate mean and std of returns
        mean_return = settled['returns'].mean()
        std_return = settled['returns'].std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # Sharpe ratio (assuming daily bets, annualized)
        sharpe = (mean_return - risk_free_rate) / std_return

        # Annualize (assuming ~250 trading days/year)
        sharpe_annualized = sharpe * np.sqrt(250)

        return sharpe_annualized

    def calculate_max_drawdown(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate maximum drawdown from peak.

        Args:
            df: DataFrame with settled bets

        Returns:
            Dict with max_drawdown, drawdown_duration, and recovery info
        """
        if df.empty:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration_days': 0,
                'current_drawdown_pct': 0.0,
                'underwater': False
            }

        # Filter and sort by settlement time
        settled = df[df['outcome'].notna()].copy()
        settled = settled.sort_values('settlement_timestamp')

        if settled.empty:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration_days': 0,
                'current_drawdown_pct': 0.0,
                'underwater': False
            }

        # Calculate cumulative profit
        settled['cumulative_profit'] = settled['profit'].cumsum()

        # Calculate running maximum
        settled['running_max'] = settled['cumulative_profit'].cummax()

        # Calculate drawdown from peak
        settled['drawdown'] = settled['cumulative_profit'] - settled['running_max']
        settled['drawdown_pct'] = (settled['drawdown'] / settled['running_max'].abs()) * 100

        # Find maximum drawdown
        max_dd_idx = settled['drawdown'].idxmin()
        max_drawdown = settled.loc[max_dd_idx, 'drawdown']
        max_drawdown_pct = settled.loc[max_dd_idx, 'drawdown_pct']

        # Calculate drawdown duration
        # Find when drawdown started (last peak before max DD)
        dd_start_idx = settled.loc[:max_dd_idx, 'running_max'].idxmax()
        dd_start_date = settled.loc[dd_start_idx, 'settlement_timestamp']
        dd_end_date = settled.loc[max_dd_idx, 'settlement_timestamp']
        dd_duration = (dd_end_date - dd_start_date).days

        # Current drawdown (from current peak)
        current_profit = settled['cumulative_profit'].iloc[-1]
        peak_profit = settled['running_max'].iloc[-1]
        current_dd = current_profit - peak_profit
        current_dd_pct = (current_dd / abs(peak_profit)) * 100 if peak_profit != 0 else 0

        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_pct': abs(max_drawdown_pct),
            'drawdown_duration_days': dd_duration,
            'current_drawdown_pct': abs(current_dd_pct),
            'underwater': current_dd < 0
        }

    def calculate_win_loss_streaks(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate longest winning and losing streaks.

        Args:
            df: DataFrame with settled bets

        Returns:
            Dict with streak information
        """
        settled = df[df['outcome'].notna()].copy()
        settled = settled.sort_values('settlement_timestamp')

        if settled.empty:
            return {
                'longest_win_streak': 0,
                'longest_loss_streak': 0,
                'current_streak': 0,
                'current_streak_type': None
            }

        # Convert outcomes to binary (1 = win, 0 = loss/push)
        wins = (settled['outcome'] == 'win').astype(int)

        # Calculate streaks
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        current_type = None

        streak = 0
        prev_val = None

        for val in wins:
            if val == prev_val:
                streak += 1
            else:
                streak = 1
                prev_val = val

            if val == 1:
                max_win_streak = max(max_win_streak, streak)
                if prev_val == 1:
                    current_streak = streak
                    current_type = 'win'
            else:
                max_loss_streak = max(max_loss_streak, streak)
                if prev_val == 0:
                    current_streak = streak
                    current_type = 'loss'

        return {
            'longest_win_streak': max_win_streak,
            'longest_loss_streak': max_loss_streak,
            'current_streak': current_streak,
            'current_streak_type': current_type
        }

    def calculate_rolling_statistics(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate rolling statistics (moving averages, volatility).

        Args:
            df: DataFrame with settled bets
            window: Rolling window size (default 20 bets)

        Returns:
            Dict with rolling statistics
        """
        settled = df[df['outcome'].notna()].copy()
        settled = settled.sort_values('settlement_timestamp')

        if len(settled) < window:
            return {
                'rolling_roi': 0.0,
                'rolling_win_rate': 0.0,
                'rolling_volatility': 0.0,
                'trend': 'insufficient_data'
            }

        # Calculate returns
        settled['returns'] = settled['profit'] / settled['bet_amount']

        # Rolling metrics
        rolling_roi = (settled['profit'].rolling(window).sum() /
                      settled['bet_amount'].rolling(window).sum() * 100).iloc[-1]

        rolling_win_rate = ((settled['outcome'] == 'win').rolling(window).sum() /
                           window * 100).iloc[-1]

        rolling_volatility = settled['returns'].rolling(window).std().iloc[-1] * 100

        # Determine trend (compare recent vs older performance)
        recent_roi = (settled['profit'].iloc[-window:].sum() /
                     settled['bet_amount'].iloc[-window:].sum() * 100)

        if len(settled) >= window * 2:
            older_roi = (settled['profit'].iloc[-window*2:-window].sum() /
                        settled['bet_amount'].iloc[-window*2:-window].sum() * 100)

            if recent_roi > older_roi + 2:
                trend = 'improving'
            elif recent_roi < older_roi - 2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'rolling_roi': rolling_roi,
            'rolling_win_rate': rolling_win_rate,
            'rolling_volatility': rolling_volatility,
            'trend': trend
        }

    def calculate_kelly_criterion_accuracy(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how well actual bet sizing matched Kelly criterion.

        Args:
            df: DataFrame with betting data

        Returns:
            Dict with Kelly accuracy metrics
        """
        settled = df[df['outcome'].notna()].copy()

        if settled.empty or 'edge' not in settled.columns:
            return {
                'avg_kelly_fraction': 0.0,
                'kelly_correlation': 0.0,
                'oversized_bets': 0,
                'undersized_bets': 0
            }

        # Calculate optimal Kelly size for each bet
        # Kelly = (edge * odds) / (odds - 1)
        settled['optimal_kelly_pct'] = (settled['edge'] * settled['odds']) / (settled['odds'] - 1)

        # Get current bankroll at time of bet (approximate)
        # For simplicity, use average bet size as proxy
        avg_bet = settled['bet_amount'].mean()
        settled['estimated_bankroll'] = avg_bet * 100  # Rough estimate

        # Calculate actual Kelly fraction used
        settled['actual_kelly_fraction'] = settled['bet_amount'] / settled['estimated_bankroll']

        # Metrics
        avg_kelly = settled['actual_kelly_fraction'].mean()

        # Correlation between edge and bet size
        kelly_corr = settled['edge'].corr(settled['bet_amount'])

        # Count oversized/undersized bets (comparing to 25% Kelly as target)
        target_kelly = 0.25
        oversized = (settled['actual_kelly_fraction'] > target_kelly * 1.5).sum()
        undersized = (settled['actual_kelly_fraction'] < target_kelly * 0.5).sum()

        return {
            'avg_kelly_fraction': avg_kelly,
            'kelly_correlation': kelly_corr if not np.isnan(kelly_corr) else 0.0,
            'oversized_bets': int(oversized),
            'undersized_bets': int(undersized)
        }

    def generate_monthly_report(self, months_back: int = 1) -> Dict:
        """
        Generate comprehensive monthly report.

        Args:
            months_back: Number of months to analyze

        Returns:
            Complete monthly report dict
        """
        logger.info(f"Generating monthly report for past {months_back} month(s)...")

        df = self.get_monthly_data(months_back)

        if df.empty:
            return {
                'error': 'No data available for the specified period',
                'period': f"Last {months_back} month(s)"
            }

        # Get base performance metrics from parent class
        performance = self.analyze_strategy_performance(df)
        teams = self.get_best_worst_teams(df)

        # Calculate advanced metrics
        sharpe_ratio = self.calculate_sharpe_ratio(df)
        drawdown = self.calculate_max_drawdown(df)
        streaks = self.calculate_win_loss_streaks(df)
        rolling_stats = self.calculate_rolling_statistics(df)
        kelly_accuracy = self.calculate_kelly_criterion_accuracy(df)

        report = {
            'period': f"Last {months_back} month(s)",
            'start_date': df['timestamp'].min().isoformat() if not df.empty else None,
            'end_date': df['timestamp'].max().isoformat() if not df.empty else None,
            'performance': performance,
            'advanced_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': drawdown,
                'streaks': streaks,
                'rolling_stats': rolling_stats,
                'kelly_accuracy': kelly_accuracy
            },
            'top_teams': teams['best'],
            'bottom_teams': teams['worst'],
            'generated_at': datetime.now().isoformat(),
        }

        logger.info("Monthly report generated successfully")
        return report

    def format_monthly_report_text(self, report: Dict) -> str:
        """
        Format monthly report as human-readable text.

        Args:
            report: Report dictionary

        Returns:
            Formatted text report
        """
        if 'error' in report:
            return f"Error: {report['error']}"

        lines = [
            "=" * 80,
            f"MONTHLY PERFORMANCE REPORT - {report['period']}",
            "=" * 80,
            f"Period: {report['start_date']} to {report['end_date']}",
            ""
        ]

        # Overall performance
        overall = report['performance']['overall']
        lines.extend([
            "OVERALL PERFORMANCE",
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

        # Advanced metrics
        adv = report['advanced_metrics']
        lines.extend([
            "ADVANCED METRICS (INVESTOR-GRADE)",
            "-" * 80,
            f"Sharpe Ratio (Annualized): {adv['sharpe_ratio']:.2f}",
            f"Maximum Drawdown: ${adv['max_drawdown']['max_drawdown']:,.2f} ({adv['max_drawdown']['max_drawdown_pct']:.1f}%)",
            f"Drawdown Duration: {adv['max_drawdown']['drawdown_duration_days']} days",
            f"Current Drawdown: {adv['max_drawdown']['current_drawdown_pct']:.1f}%",
            f"Underwater: {'Yes' if adv['max_drawdown']['underwater'] else 'No'}",
            ""
        ])

        # Streaks
        streaks = adv['streaks']
        lines.extend([
            "WIN/LOSS STREAKS",
            "-" * 80,
            f"Longest Win Streak: {streaks['longest_win_streak']} bets",
            f"Longest Loss Streak: {streaks['longest_loss_streak']} bets",
            f"Current Streak: {streaks['current_streak']} {streaks['current_streak_type'] or 'N/A'}",
            ""
        ])

        # Rolling statistics
        rolling = adv['rolling_stats']
        lines.extend([
            "ROLLING STATISTICS (Last 20 Bets)",
            "-" * 80,
            f"Rolling ROI: {rolling['rolling_roi']:+.2f}%",
            f"Rolling Win Rate: {rolling['rolling_win_rate']:.1f}%",
            f"Rolling Volatility: {rolling['rolling_volatility']:.2f}%",
            f"Performance Trend: {rolling['trend'].upper()}",
            ""
        ])

        # Kelly accuracy
        kelly = adv['kelly_accuracy']
        lines.extend([
            "BET SIZING ANALYSIS",
            "-" * 80,
            f"Avg Kelly Fraction: {kelly['avg_kelly_fraction']:.2%}",
            f"Edge/Size Correlation: {kelly['kelly_correlation']:+.3f}",
            f"Oversized Bets (>37.5% Kelly): {kelly['oversized_bets']}",
            f"Undersized Bets (<12.5% Kelly): {kelly['undersized_bets']}",
            ""
        ])

        # By strategy
        if report['performance']['by_strategy']:
            lines.extend([
                "PERFORMANCE BY STRATEGY",
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
                "TOP PERFORMING TEAMS",
                "-" * 80
            ])
            for i, team in enumerate(report['top_teams'], 1):
                lines.append(f"{i}. {team['team']}: ${team['profit']:+,.2f} ({team['roi']:+.1f}% ROI, {team['win_rate']:.0f}% WR)")
            lines.append("")

        # Worst teams
        if report['bottom_teams']:
            lines.extend([
                "WORST PERFORMING TEAMS",
                "-" * 80
            ])
            for i, team in enumerate(report['bottom_teams'], 1):
                lines.append(f"{i}. {team['team']}: ${team['profit']:+,.2f} ({team['roi']:+.1f}% ROI, {team['win_rate']:.0f}% WR)")
            lines.append("")

        lines.extend([
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])

        return "\n".join(lines)
