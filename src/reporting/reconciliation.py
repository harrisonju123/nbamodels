"""
Reconciliation Engine

Validates betting data against external sources, checks for inconsistencies,
and generates reconciliation reports to ensure data integrity.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from loguru import logger

# Import data fetchers for validation
try:
    from src.data.nba_api_client import NBADataClient
except ImportError:
    NBADataClient = None


class ReconciliationEngine:
    """
    Reconcile betting data against actual game results and identify discrepancies.
    """

    def __init__(self, db_path: str = "data/bets/bets.db"):
        """
        Initialize reconciliation engine.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path
        self.nba_client = NBADataClient() if NBADataClient else None

    def get_settled_bets(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get all settled bets from the past N days.

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with settled bets
        """
        try:
            conn = sqlite3.connect(self.db_path)

            cutoff_date = (datetime.now() - timedelta(days=days_back)).date()

            query = """
                SELECT
                    id as bet_id,
                    logged_at as timestamp,
                    settled_at as settlement_timestamp,
                    game_id,
                    home_team,
                    away_team,
                    bet_side,
                    bet_type,
                    line,
                    odds,
                    bet_amount,
                    outcome,
                    profit,
                    bookmaker
                FROM bets
                WHERE outcome IS NOT NULL
                  AND DATE(settled_at) >= ?
                ORDER BY settled_at DESC
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            conn.close()

            return df
        except Exception as e:
            logger.error(f"Error getting settled bets: {e}")
            return pd.DataFrame()

    def get_game_results(self, game_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch actual game results from NBA API.

        Args:
            game_ids: List of game IDs to fetch

        Returns:
            Dict mapping game_id to game result info
        """
        if not self.nba_client:
            logger.warning("NBA API client not available, cannot fetch game results")
            return {}

        results = {}
        for game_id in game_ids:
            try:
                # Fetch game data from API
                game_data = self.nba_client.get_game_by_id(game_id)

                if game_data and 'home_score' in game_data and 'away_score' in game_data:
                    results[game_id] = {
                        'home_score': game_data['home_score'],
                        'away_score': game_data['away_score'],
                        'home_team': game_data.get('home_team'),
                        'away_team': game_data.get('away_team'),
                        'game_date': game_data.get('game_date'),
                        'final': game_data.get('final', True)
                    }
            except Exception as e:
                logger.warning(f"Could not fetch game {game_id}: {e}")
                continue

        return results

    def validate_bet_outcomes(self, df: pd.DataFrame) -> List[Dict]:
        """
        Validate bet outcomes against actual game results.

        Args:
            df: DataFrame with settled bets

        Returns:
            List of discrepancies found
        """
        if df.empty:
            return []

        discrepancies = []

        # Get unique game IDs
        game_ids = df['game_id'].unique().tolist()

        # Fetch actual results
        logger.info(f"Validating {len(df)} bets across {len(game_ids)} games...")
        game_results = self.get_game_results(game_ids)

        if not game_results:
            logger.warning("No game results available for validation")
            return discrepancies

        # Validate each bet
        for idx, bet in df.iterrows():
            game_id = bet['game_id']

            if game_id not in game_results:
                discrepancies.append({
                    'bet_id': bet['bet_id'],
                    'type': 'missing_game_data',
                    'message': f"Game {game_id} not found in API results",
                    'severity': 'warning'
                })
                continue

            result = game_results[game_id]

            # Validate spread bet outcome
            if bet['bet_type'] == 'spread':
                home_score = result['home_score']
                away_score = result['away_score']
                line = bet['line']

                if bet['bet_side'] == 'HOME':
                    # Home team + line
                    expected_outcome = 'win' if (home_score + line) > away_score else 'loss'
                    if abs((home_score + line) - away_score) < 0.5:
                        expected_outcome = 'push'
                else:
                    # Away team - line (equivalent to home team + line)
                    expected_outcome = 'win' if (away_score - line) > home_score else 'loss'
                    if abs((away_score - line) - home_score) < 0.5:
                        expected_outcome = 'push'

                # Compare with recorded outcome
                if bet['outcome'] != expected_outcome:
                    discrepancies.append({
                        'bet_id': bet['bet_id'],
                        'game_id': game_id,
                        'type': 'outcome_mismatch',
                        'message': f"Expected {expected_outcome}, got {bet['outcome']}",
                        'expected': expected_outcome,
                        'actual': bet['outcome'],
                        'game_score': f"{result['home_team']} {home_score} - {away_score} {result['away_team']}",
                        'bet_details': f"{bet['bet_side']} {line:+.1f}",
                        'severity': 'error'
                    })

        return discrepancies

    def check_data_integrity(self) -> List[Dict]:
        """
        Check for data integrity issues in the database.

        Returns:
            List of integrity issues found
        """
        issues = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check 1: Bets with missing critical fields
            cursor.execute("""
                SELECT id, game_id
                FROM bets
                WHERE home_team IS NULL
                   OR away_team IS NULL
                   OR bet_type IS NULL
                   OR odds IS NULL
            """)
            missing_fields = cursor.fetchall()
            for bet_id, game_id in missing_fields:
                issues.append({
                    'type': 'missing_fields',
                    'bet_id': bet_id,
                    'message': f"Bet {bet_id} has missing critical fields",
                    'severity': 'error'
                })

            # Check 2: Impossible odds values
            cursor.execute("""
                SELECT id, odds
                FROM bets
                WHERE odds < 1.0 OR odds > 100.0
            """)
            bad_odds = cursor.fetchall()
            for bet_id, odds in bad_odds:
                issues.append({
                    'type': 'invalid_odds',
                    'bet_id': bet_id,
                    'odds': odds,
                    'message': f"Bet {bet_id} has invalid odds: {odds}",
                    'severity': 'error'
                })

            # Check 3: Settled bets with NULL settled_at
            cursor.execute("""
                SELECT id
                FROM bets
                WHERE outcome IS NOT NULL
                  AND settled_at IS NULL
            """)
            missing_settlement = cursor.fetchall()
            for (bet_id,) in missing_settlement:
                issues.append({
                    'type': 'missing_settlement_time',
                    'bet_id': bet_id,
                    'message': f"Settled bet {bet_id} missing settlement timestamp",
                    'severity': 'warning'
                })

            # Check 4: Negative bet amounts
            cursor.execute("""
                SELECT id, bet_amount
                FROM bets
                WHERE bet_amount <= 0
            """)
            negative_amounts = cursor.fetchall()
            for bet_id, amount in negative_amounts:
                issues.append({
                    'type': 'invalid_bet_amount',
                    'bet_id': bet_id,
                    'amount': amount,
                    'message': f"Bet {bet_id} has invalid amount: ${amount}",
                    'severity': 'error'
                })

            # Check 5: Profit calculation errors for settled bets
            cursor.execute("""
                SELECT id, bet_amount, odds, outcome, profit
                FROM bets
                WHERE outcome IS NOT NULL
            """)
            for bet_id, bet_amount, odds, outcome, profit in cursor.fetchall():
                if outcome == 'win':
                    expected_profit = bet_amount * (odds - 1)
                elif outcome == 'loss':
                    expected_profit = -bet_amount
                else:  # push
                    expected_profit = 0

                # Allow small rounding errors
                if abs(profit - expected_profit) > 0.01:
                    issues.append({
                        'type': 'profit_calculation_error',
                        'bet_id': bet_id,
                        'expected_profit': expected_profit,
                        'actual_profit': profit,
                        'message': f"Bet {bet_id} profit mismatch: expected ${expected_profit:.2f}, got ${profit:.2f}",
                        'severity': 'error'
                    })

            # Check 6: Duplicate game_id + logged_at combinations (possible double-bets)
            cursor.execute("""
                SELECT game_id, logged_at, COUNT(*) as cnt
                FROM bets
                GROUP BY game_id, logged_at
                HAVING cnt > 1
            """)
            duplicates = cursor.fetchall()
            for game_id, timestamp, count in duplicates:
                issues.append({
                    'type': 'possible_duplicate',
                    'game_id': game_id,
                    'timestamp': timestamp,
                    'count': count,
                    'message': f"Found {count} bets for game {game_id} at {timestamp}",
                    'severity': 'warning'
                })

            conn.close()

        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
            issues.append({
                'type': 'database_error',
                'message': str(e),
                'severity': 'error'
            })

        return issues

    def check_bankroll_consistency(self) -> Dict:
        """
        Verify bankroll calculations are consistent with bet history.

        Returns:
            Dict with consistency check results
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Get all settled bets in chronological order
            bets_df = pd.read_sql_query("""
                SELECT bet_amount, profit, settled_at
                FROM bets
                WHERE outcome IS NOT NULL
                ORDER BY settled_at ASC
            """, conn)

            # Get bankroll history
            bankroll_df = pd.read_sql_query("""
                SELECT timestamp, current_bankroll, change_amount, change_reason
                FROM bankroll
                ORDER BY timestamp ASC
            """, conn)

            conn.close()

            if bets_df.empty or bankroll_df.empty:
                return {
                    'consistent': True,
                    'message': 'Insufficient data for bankroll validation'
                }

            # Calculate expected final bankroll from bets
            total_profit = bets_df['profit'].sum()
            initial_bankroll = bankroll_df['current_bankroll'].iloc[0]
            current_bankroll = bankroll_df['current_bankroll'].iloc[-1]

            # Expected current bankroll
            # Note: This assumes initial bankroll + all profits, may need adjustment
            # if there were deposits/withdrawals
            expected_bankroll = initial_bankroll + total_profit

            # Allow small rounding errors
            difference = abs(current_bankroll - expected_bankroll)

            if difference > 1.0:  # More than $1 difference
                return {
                    'consistent': False,
                    'initial_bankroll': initial_bankroll,
                    'current_bankroll': current_bankroll,
                    'expected_bankroll': expected_bankroll,
                    'difference': difference,
                    'total_profit': total_profit,
                    'message': f"Bankroll inconsistency detected: ${difference:.2f} discrepancy"
                }
            else:
                return {
                    'consistent': True,
                    'initial_bankroll': initial_bankroll,
                    'current_bankroll': current_bankroll,
                    'total_profit': total_profit,
                    'message': 'Bankroll calculations are consistent'
                }

        except Exception as e:
            logger.error(f"Error checking bankroll consistency: {e}")
            return {
                'consistent': False,
                'message': f"Error during validation: {str(e)}"
            }

    def run_full_reconciliation(self, days_back: int = 7) -> Dict:
        """
        Run complete reconciliation process.

        Args:
            days_back: Number of days to reconcile

        Returns:
            Complete reconciliation report
        """
        logger.info(f"Running full reconciliation for past {days_back} days...")

        # Get settled bets
        settled_bets = self.get_settled_bets(days_back)

        # Validate outcomes against API
        outcome_discrepancies = self.validate_bet_outcomes(settled_bets)

        # Check data integrity
        integrity_issues = self.check_data_integrity()

        # Check bankroll consistency
        bankroll_check = self.check_bankroll_consistency()

        # Count issues by severity
        errors = [i for i in (outcome_discrepancies + integrity_issues) if i.get('severity') == 'error']
        warnings = [i for i in (outcome_discrepancies + integrity_issues) if i.get('severity') == 'warning']

        report = {
            'reconciliation_date': datetime.now().isoformat(),
            'period_days': days_back,
            'bets_reviewed': len(settled_bets),
            'outcome_discrepancies': outcome_discrepancies,
            'integrity_issues': integrity_issues,
            'bankroll_check': bankroll_check,
            'summary': {
                'total_issues': len(errors) + len(warnings),
                'errors': len(errors),
                'warnings': len(warnings),
                'bankroll_consistent': bankroll_check.get('consistent', False)
            }
        }

        logger.info(f"Reconciliation complete: {len(errors)} errors, {len(warnings)} warnings")
        return report

    def format_reconciliation_report(self, report: Dict) -> str:
        """
        Format reconciliation report as human-readable text.

        Args:
            report: Reconciliation report dict

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 80,
            "RECONCILIATION REPORT",
            "=" * 80,
            f"Date: {report['reconciliation_date']}",
            f"Period: Last {report['period_days']} days",
            f"Bets Reviewed: {report['bets_reviewed']}",
            ""
        ]

        # Summary
        summary = report['summary']
        lines.extend([
            "SUMMARY",
            "-" * 80,
            f"Total Issues Found: {summary['total_issues']}",
            f"  - Errors: {summary['errors']}",
            f"  - Warnings: {summary['warnings']}",
            f"Bankroll Consistent: {'YES' if summary['bankroll_consistent'] else 'NO'}",
            ""
        ])

        # Bankroll check details
        bankroll = report['bankroll_check']
        lines.extend([
            "BANKROLL VALIDATION",
            "-" * 80,
            f"Status: {bankroll['message']}",
        ])
        if 'current_bankroll' in bankroll:
            lines.extend([
                f"Initial Bankroll: ${bankroll.get('initial_bankroll', 0):,.2f}",
                f"Current Bankroll: ${bankroll.get('current_bankroll', 0):,.2f}",
                f"Total Profit: ${bankroll.get('total_profit', 0):+,.2f}",
            ])
            if not bankroll['consistent']:
                lines.append(f"Discrepancy: ${bankroll.get('difference', 0):,.2f}")
        lines.append("")

        # Outcome discrepancies
        if report['outcome_discrepancies']:
            lines.extend([
                "OUTCOME DISCREPANCIES",
                "-" * 80
            ])
            for disc in report['outcome_discrepancies']:
                lines.append(f"[{disc['severity'].upper()}] Bet #{disc['bet_id']}: {disc['message']}")
                if 'game_score' in disc:
                    lines.append(f"  Game: {disc['game_score']}")
                    lines.append(f"  Bet: {disc['bet_details']}")
                    lines.append(f"  Expected: {disc['expected']}, Recorded: {disc['actual']}")
            lines.append("")

        # Integrity issues
        if report['integrity_issues']:
            lines.extend([
                "DATA INTEGRITY ISSUES",
                "-" * 80
            ])
            for issue in report['integrity_issues']:
                lines.append(f"[{issue['severity'].upper()}] {issue['message']}")
            lines.append("")

        # Final status
        if summary['errors'] == 0 and summary['warnings'] == 0 and summary['bankroll_consistent']:
            lines.extend([
                "=" * 80,
                "STATUS: ALL CHECKS PASSED",
                "=" * 80
            ])
        else:
            lines.extend([
                "=" * 80,
                f"STATUS: ISSUES DETECTED ({summary['errors']} errors, {summary['warnings']} warnings)",
                "=" * 80
            ])

        return "\n".join(lines)
