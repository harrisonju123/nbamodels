"""
FastAPI Backend for Real-time Dashboard Data

Provides REST API endpoints for:
- Real-time bet tracking
- Performance metrics
- Live odds updates
- Model predictions

Run with: uvicorn api.dashboard_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import secrets

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bet_tracker import (
    get_bet_history,
    get_performance_summary,
    get_enhanced_clv_summary,
)
from src.utils.constants import BETS_DB_PATH

# Initialize FastAPI app
app = FastAPI(
    title="NBA Betting Analytics API",
    description="Real-time API for NBA betting model analytics",
    version="1.0.0"
)

# Enable CORS for frontend
# Configure allowed origins based on environment
import os

ALLOWED_ORIGINS = [
    "http://localhost:8501",  # Streamlit default
    "http://localhost:3000",  # Common React dev port
]

# Add production origins from environment
if os.getenv("PRODUCTION_ORIGIN"):
    ALLOWED_ORIGINS.append(os.getenv("PRODUCTION_ORIGIN"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Whitelist only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
)

# API Authentication
security = HTTPBearer()
API_KEY = os.getenv("DASHBOARD_API_KEY")

# Warn if API key not configured
if not API_KEY:
    import logging
    logging.warning("⚠️  DASHBOARD_API_KEY not set - API authentication is DISABLED!")
    logging.warning("⚠️  Set environment variable to enable security: export DASHBOARD_API_KEY=your-secret-key")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key from Authorization header."""
    if not API_KEY:
        # Allow unauthenticated access if API key not configured (development only)
        return "unauthenticated"

    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Optional: Dependency for endpoints (can be disabled in development)
def get_api_auth():
    """Get API authentication dependency based on configuration."""
    if API_KEY:
        return Depends(verify_api_key)
    return None

# Pydantic models for API responses
class BetSummary(BaseModel):
    total_bets: int
    settled_bets: int
    pending_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_profit: float
    total_wagered: float
    roi: float
    avg_clv: float

class PerformanceMetrics(BaseModel):
    win_rate: float
    roi: float
    total_profit: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    avg_bet_size: float
    profit_factor: float

class BetRecord(BaseModel):
    id: str
    game_id: str
    bet_type: str
    bet_side: str
    line: float
    odds: float
    bet_amount: float
    edge: float
    model_prob: float
    outcome: Optional[str]
    profit: Optional[float]
    clv: Optional[float]
    logged_at: str
    settled_at: Optional[str]

class CLVAnalysis(BaseModel):
    avg_clv: float
    median_clv: float
    positive_clv_rate: float
    clv_by_outcome: Dict[str, float]
    best_clv: float
    worst_clv: float

class LiveUpdate(BaseModel):
    timestamp: str
    active_bets: int
    todays_profit: float
    todays_win_rate: float
    pending_games: int

# Helper functions
def calculate_performance_metrics(df: pd.DataFrame) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    if df.empty:
        return PerformanceMetrics(
            win_rate=0.0,
            roi=0.0,
            total_profit=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            avg_bet_size=0.0,
            profit_factor=0.0
        )

    settled = df[df['outcome'].notna()]

    if len(settled) == 0:
        return PerformanceMetrics(
            win_rate=0.0,
            roi=0.0,
            total_profit=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            avg_bet_size=0.0,
            profit_factor=0.0
        )

    wins = len(settled[settled['outcome'] == 'win'])
    losses = len(settled[settled['outcome'] == 'loss'])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    total_profit = settled['profit'].sum()
    total_wagered = settled['bet_amount'].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # Calculate max drawdown
    settled_sorted = settled.sort_values('logged_at')
    cumulative = settled_sorted['profit'].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Calculate profit factor
    total_wins = settled[settled['profit'] > 0]['profit'].sum()
    total_losses = abs(settled[settled['profit'] < 0]['profit'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Sharpe ratio (daily returns)
    if len(settled) > 5:
        daily_returns = settled.groupby(pd.to_datetime(settled['logged_at']).dt.date)['profit'].sum()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else None
    else:
        sharpe = None

    return PerformanceMetrics(
        win_rate=round(win_rate, 2),
        roi=round(roi, 2),
        total_profit=round(total_profit, 2),
        sharpe_ratio=round(sharpe, 2) if sharpe else None,
        max_drawdown=round(max_drawdown, 2),
        avg_bet_size=round(settled['bet_amount'].mean(), 2),
        profit_factor=round(profit_factor, 2)
    )

# API Endpoints

@app.get("/")
def root():
    """API root endpoint."""
    return {
        "message": "NBA Betting Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "/api/summary": "Get betting summary",
            "/api/performance": "Get performance metrics",
            "/api/bets": "Get all bets",
            "/api/bets/recent": "Get recent bets",
            "/api/clv": "Get CLV analysis",
            "/api/live": "Get live updates",
            "/api/stats/daily": "Get daily statistics",
        }
    }

@app.get("/api/summary", response_model=BetSummary)
def get_summary(
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter by last N days (1-365)"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get betting summary statistics."""
    try:
        df = get_bet_history()

        if df.empty:
            return BetSummary(
                total_bets=0,
                settled_bets=0,
                pending_bets=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0.0,
                total_profit=0.0,
                total_wagered=0.0,
                roi=0.0,
                avg_clv=0.0
            )

        # Filter by date if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)
            df = df[df['logged_at'] >= cutoff]

        total_bets = len(df)
        settled = df[df['outcome'].notna()]
        pending = df[df['outcome'].isna()]

        wins = len(settled[settled['outcome'] == 'win'])
        losses = len(settled[settled['outcome'] == 'loss'])
        pushes = len(settled[settled['outcome'] == 'push'])

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        total_profit = settled['profit'].sum()
        total_wagered = settled['bet_amount'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

        avg_clv = settled['clv'].mean() if 'clv' in settled.columns and not settled['clv'].isna().all() else 0

        return BetSummary(
            total_bets=total_bets,
            settled_bets=len(settled),
            pending_bets=len(pending),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=round(win_rate, 2),
            total_profit=round(total_profit, 2),
            total_wagered=round(total_wagered, 2),
            roi=round(roi, 2),
            avg_clv=round(avg_clv, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance", response_model=PerformanceMetrics)
def get_performance(
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter by last N days (1-365)"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get detailed performance metrics."""
    try:
        df = get_bet_history()

        if days and not df.empty:
            cutoff = datetime.now() - timedelta(days=days)
            df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)
            df = df[df['logged_at'] >= cutoff]

        return calculate_performance_metrics(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bets", response_model=List[BetRecord])
def get_bets(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of bets (1-1000)"),
    offset: int = Query(0, ge=0, description="Number of bets to skip"),
    bet_type: Optional[str] = Query(None, regex="^(spread|totals|moneyline)$", description="Filter by bet type"),
    outcome: Optional[str] = Query(None, regex="^(win|loss|push|pending)$", description="Filter by outcome"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get all bets with optional filtering."""
    try:
        df = get_bet_history()

        if df.empty:
            return []

        # Apply filters
        if bet_type:
            df = df[df['bet_type'] == bet_type]

        if outcome:
            df = df[df['outcome'] == outcome]

        # Sort by most recent
        df = df.sort_values('logged_at', ascending=False)

        # Apply pagination
        df = df.iloc[offset:offset+limit]

        # Convert to BetRecord objects
        bets = []
        for _, row in df.iterrows():
            bets.append(BetRecord(
                id=str(row['id']),
                game_id=str(row['game_id']),
                bet_type=row['bet_type'],
                bet_side=row['bet_side'],
                line=float(row['line']),
                odds=float(row['odds']),
                bet_amount=float(row['bet_amount']),
                edge=float(row['edge']) if 'edge' in row and pd.notna(row['edge']) else 0.0,
                model_prob=float(row['model_prob']) if 'model_prob' in row and pd.notna(row['model_prob']) else 0.0,
                outcome=row['outcome'] if pd.notna(row['outcome']) else None,
                profit=float(row['profit']) if pd.notna(row['profit']) else None,
                clv=float(row['clv']) if 'clv' in row and pd.notna(row['clv']) else None,
                logged_at=str(row['logged_at']),
                settled_at=str(row['settled_at']) if pd.notna(row.get('settled_at')) else None
            ))

        return bets

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bets/recent", response_model=List[BetRecord])
def get_recent_bets(
    count: int = Query(10, ge=1, le=100, description="Number of recent bets (1-100)"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get most recent bets."""
    return get_bets(limit=count, offset=0)

@app.get("/api/clv", response_model=CLVAnalysis)
def get_clv(
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter by last N days (1-365)"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get CLV analysis."""
    try:
        df = get_bet_history()

        if df.empty or 'clv' not in df.columns:
            return CLVAnalysis(
                avg_clv=0.0,
                median_clv=0.0,
                positive_clv_rate=0.0,
                clv_by_outcome={},
                best_clv=0.0,
                worst_clv=0.0
            )

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)
            df = df[df['logged_at'] >= cutoff]

        df_clv = df[df['clv'].notna()]

        if len(df_clv) == 0:
            return CLVAnalysis(
                avg_clv=0.0,
                median_clv=0.0,
                positive_clv_rate=0.0,
                clv_by_outcome={},
                best_clv=0.0,
                worst_clv=0.0
            )

        avg_clv = df_clv['clv'].mean()
        median_clv = df_clv['clv'].median()
        positive_clv_rate = (df_clv['clv'] > 0).sum() / len(df_clv) * 100

        # CLV by outcome
        clv_by_outcome = {}
        if 'outcome' in df_clv.columns:
            for outcome in ['win', 'loss', 'push']:
                outcome_df = df_clv[df_clv['outcome'] == outcome]
                if len(outcome_df) > 0:
                    clv_by_outcome[outcome] = round(outcome_df['clv'].mean(), 2)

        return CLVAnalysis(
            avg_clv=round(avg_clv, 2),
            median_clv=round(median_clv, 2),
            positive_clv_rate=round(positive_clv_rate, 2),
            clv_by_outcome=clv_by_outcome,
            best_clv=round(df_clv['clv'].max(), 2),
            worst_clv=round(df_clv['clv'].min(), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live", response_model=LiveUpdate)
def get_live_update(_auth: str = Depends(verify_api_key) if API_KEY else None):
    """Get real-time live update."""
    try:
        df = get_bet_history()

        if df.empty:
            return LiveUpdate(
                timestamp=datetime.now().isoformat(),
                active_bets=0,
                todays_profit=0.0,
                todays_win_rate=0.0,
                pending_games=0
            )

        # Today's data
        today = datetime.now().date()
        df['placed_date'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True).dt.date

        todays_bets = df[df['placed_date'] == today]
        todays_settled = todays_bets[todays_bets['outcome'].notna()]

        todays_profit = todays_settled['profit'].sum() if len(todays_settled) > 0 else 0.0

        todays_wins = len(todays_settled[todays_settled['outcome'] == 'win'])
        todays_total = len(todays_settled[todays_settled['outcome'] != 'push'])
        todays_win_rate = (todays_wins / todays_total * 100) if todays_total > 0 else 0.0

        pending_games = len(df[df['outcome'].isna()])

        return LiveUpdate(
            timestamp=datetime.now().isoformat(),
            active_bets=len(todays_bets),
            todays_profit=round(todays_profit, 2),
            todays_win_rate=round(todays_win_rate, 2),
            pending_games=pending_games
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/daily")
def get_daily_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to return (1-365)"),
    _auth: str = Depends(verify_api_key) if API_KEY else None
):
    """Get daily aggregated statistics."""
    try:
        df = get_bet_history()

        if df.empty:
            return []

        # Convert to datetime
        df['logged_at'] = pd.to_datetime(df['logged_at'], format='ISO8601', utc=True)
        df['date'] = df['logged_at'].dt.date

        # Filter to last N days
        cutoff = datetime.now().date() - timedelta(days=days)
        df = df[df['date'] >= cutoff]

        # Group by date
        daily_stats = []
        for date, group in df.groupby('date'):
            settled = group[group['outcome'].notna()]

            if len(settled) == 0:
                continue

            wins = len(settled[settled['outcome'] == 'win'])
            losses = len(settled[settled['outcome'] == 'loss'])
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

            daily_stats.append({
                'date': str(date),
                'bets': len(settled),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'profit': round(settled['profit'].sum(), 2),
                'roi': round((settled['profit'].sum() / settled['bet_amount'].sum() * 100), 2) if settled['bet_amount'].sum() > 0 else 0
            })

        return sorted(daily_stats, key=lambda x: x['date'])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
