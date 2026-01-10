"""
Risk Attribution Engine

Breaks down P&L by multiple dimensions:
1. Team-level: P&L betting on/against each team
2. Situation: B2B, rest advantage, home/away, edge bucket
3. Time: Daily, weekly, monthly rollups
"""

import sqlite3
from datetime import datetime
from typing import Optional
import pandas as pd
from loguru import logger

from .models import get_team_conference, get_team_division


DB_PATH = "data/bets/bets.db"


class RiskAttributionEngine:
    """
    Calculates P&L attribution across multiple dimensions.

    Queries the bets database to generate attribution reports.
    """

    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize attribution engine.

        Args:
            db_path: Path to bets database
        """
        self.db_path = db_path
        logger.debug("RiskAttributionEngine initialized")

    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_team_attribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get P&L breakdown by team.

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with columns:
            - team: Team abbreviation
            - bets_for: Number of bets on this team
            - bets_against: Number of bets against this team
            - profit_for: Profit betting on team
            - profit_against: Profit betting against team
            - wagered_for: Amount wagered on team
            - wagered_against: Amount wagered against team
            - win_rate_for: Win rate betting on team
            - win_rate_against: Win rate betting against team
            - roi_for: ROI betting on team
            - roi_against: ROI betting against team
        """
        conn = self._get_connection()

        # Build WHERE clause
        where_clauses = ["outcome IS NOT NULL"]  # Only settled bets
        params = []

        if start_date:
            where_clauses.append("DATE(settled_at) >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("DATE(settled_at) <= ?")
            params.append(end_date)

        where_clause = " AND ".join(where_clauses)

        # Query for bets on teams (bet_side = 'HOME' or 'AWAY')
        query = f"""
        SELECT
            CASE
                WHEN bet_side = 'home' THEN home_team
                WHEN bet_side = 'away' THEN away_team
                ELSE NULL
            END as team,
            'for' as direction,
            COUNT(*) as bet_count,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(profit) as total_profit,
            SUM(bet_amount) as total_wagered
        FROM bets
        WHERE {where_clause}
            AND bet_side IN ('home', 'away')
        GROUP BY team

        UNION ALL

        SELECT
            CASE
                WHEN bet_side = 'home' THEN away_team
                WHEN bet_side = 'away' THEN home_team
                ELSE NULL
            END as team,
            'against' as direction,
            COUNT(*) as bet_count,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(profit) as total_profit,
            SUM(bet_amount) as total_wagered
        FROM bets
        WHERE {where_clause}
            AND bet_side IN ('home', 'away')
        GROUP BY team
        """

        df = pd.read_sql_query(query, conn, params=params * 2)  # params used twice (UNION)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=[
                'team', 'bets_for', 'bets_against', 'profit_for', 'profit_against',
                'wagered_for', 'wagered_against', 'win_rate_for', 'win_rate_against',
                'roi_for', 'roi_against', 'conference', 'division'
            ])

        # Pivot to get for/against columns
        pivot = df.pivot_table(
            index='team',
            columns='direction',
            values=['bet_count', 'wins', 'total_profit', 'total_wagered'],
            aggfunc='sum',
            fill_value=0
        )

        # Flatten column names
        result = pd.DataFrame()
        result['team'] = pivot.index
        result['bets_for'] = pivot[('bet_count', 'for')].values
        result['bets_against'] = pivot[('bet_count', 'against')].values
        result['profit_for'] = pivot[('total_profit', 'for')].values
        result['profit_against'] = pivot[('total_profit', 'against')].values
        result['wagered_for'] = pivot[('total_wagered', 'for')].values
        result['wagered_against'] = pivot[('total_wagered', 'against')].values

        # Calculate win rates
        result['wins_for'] = pivot[('wins', 'for')].values
        result['wins_against'] = pivot[('wins', 'against')].values
        result['win_rate_for'] = result['wins_for'] / result['bets_for'].replace(0, 1)
        result['win_rate_against'] = result['wins_against'] / result['bets_against'].replace(0, 1)

        # Calculate ROI
        result['roi_for'] = result['profit_for'] / result['wagered_for'].replace(0, 1)
        result['roi_against'] = result['profit_against'] / result['wagered_against'].replace(0, 1)

        # Add conference and division
        result['conference'] = result['team'].apply(get_team_conference)
        result['division'] = result['team'].apply(get_team_division)

        return result.reset_index(drop=True)

    def get_situation_attribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get P&L breakdown by game situation.

        Requires home_b2b, away_b2b, rest_advantage columns in bets table.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with P&L by situation type
        """
        conn = self._get_connection()

        # Build WHERE clause
        where_clauses = ["outcome IS NOT NULL"]
        params = []

        if start_date:
            where_clauses.append("DATE(settled_at) >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("DATE(settled_at) <= ?")
            params.append(end_date)

        where_clause = " AND ".join(where_clauses)

        query = f"""
        SELECT
            CASE WHEN bet_side = 'home' THEN 1 ELSE 0 END as is_home,
            home_b2b,
            away_b2b,
            CASE WHEN rest_advantage > 0 THEN 1 ELSE 0 END as has_rest_advantage,
            edge_bucket,
            COUNT(*) as bet_count,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(profit) as total_profit,
            SUM(bet_amount) as total_wagered
        FROM bets
        WHERE {where_clause}
        GROUP BY is_home, home_b2b, away_b2b, has_rest_advantage, edge_bucket
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        # Calculate metrics
        df['win_rate'] = df['wins'] / df['bet_count']
        df['roi'] = df['total_profit'] / df['total_wagered'].replace(0, 1)

        return df

    def get_time_attribution(
        self,
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get P&L rollup by time period.

        Args:
            period: "daily", "weekly", or "monthly"
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with P&L by time period
        """
        conn = self._get_connection()

        # Build WHERE clause
        where_clauses = ["outcome IS NOT NULL"]
        params = []

        if start_date:
            where_clauses.append("DATE(settled_at) >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("DATE(settled_at) <= ?")
            params.append(end_date)

        where_clause = " AND ".join(where_clauses)

        # Determine date grouping
        if period == "daily":
            date_group = "DATE(settled_at)"
        elif period == "weekly":
            date_group = "strftime('%Y-W%W', settled_at)"
        elif period == "monthly":
            date_group = "strftime('%Y-%m', settled_at)"
        else:
            raise ValueError(f"Invalid period: {period}")

        query = f"""
        SELECT
            {date_group} as period,
            COUNT(*) as bet_count,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(profit) as total_profit,
            SUM(bet_amount) as total_wagered
        FROM bets
        WHERE {where_clause}
        GROUP BY period
        ORDER BY period
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        # Calculate metrics
        df['win_rate'] = df['wins'] / df['bet_count']
        df['roi'] = df['total_profit'] / df['total_wagered'].replace(0, 1)

        # Running totals
        df['cumulative_profit'] = df['total_profit'].cumsum()
        df['cumulative_wagered'] = df['total_wagered'].cumsum()
        df['cumulative_roi'] = df['cumulative_profit'] / df['cumulative_wagered'].replace(0, 1)

        return df

    def get_edge_bucket_attribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get P&L breakdown by edge bucket.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with P&L by edge bucket
        """
        conn = self._get_connection()

        where_clauses = ["outcome IS NOT NULL"]
        params = []

        if start_date:
            where_clauses.append("DATE(settled_at) >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("DATE(settled_at) <= ?")
            params.append(end_date)

        where_clause = " AND ".join(where_clauses)

        query = f"""
        SELECT
            edge_bucket,
            COUNT(*) as bet_count,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(profit) as total_profit,
            SUM(bet_amount) as total_wagered,
            AVG(edge) as avg_edge
        FROM bets
        WHERE {where_clause}
            AND edge_bucket IS NOT NULL
        GROUP BY edge_bucket
        ORDER BY edge_bucket
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        df['win_rate'] = df['wins'] / df['bet_count']
        df['roi'] = df['total_profit'] / df['total_wagered'].replace(0, 1)

        return df

    def __repr__(self) -> str:
        return f"RiskAttributionEngine(db_path={self.db_path})"
