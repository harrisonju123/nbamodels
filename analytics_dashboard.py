"""
Advanced Analytics Dashboard for NBA Betting Model

Features:
- Real-time performance tracking
- CLV (Closing Line Value) analysis
- Model calibration and accuracy metrics
- Interactive visualizations
- Bankroll management analytics

Run with: streamlit run analytics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bet_tracker import (
    get_bet_history,
    get_performance_summary,
    get_enhanced_clv_summary,
    get_performance_decay_metrics,
)
from src.utils.constants import BETS_DB_PATH

# Page configuration
st.set_page_config(
    page_title="NBA Betting Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1f2937;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    # Date range filter
    st.subheader("Filters")

    date_range = st.selectbox(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"],
        index=1
    )

    if date_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
    else:
        if date_range == "Last 7 Days":
            days = 7
        elif date_range == "Last 30 Days":
            days = 30
        elif date_range == "Last 90 Days":
            days = 90
        else:
            days = None

        if days:
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
        else:
            start_date = None
            end_date = None

    # Bet type filter
    bet_types = st.multiselect(
        "Bet Types",
        ["spread", "totals", "moneyline"],
        default=["spread"]
    )

    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Helper functions
@st.cache_data(ttl=60)
def load_bets_data(start_date=None, end_date=None):
    """Load bets from database with caching."""
    df = get_bet_history()

    if df.empty:
        return df

    # Convert date strings to datetime
    df['logged_at'] = pd.to_datetime(df['logged_at'])

    # Filter by date range
    if start_date:
        df = df[df['logged_at'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['logged_at'] <= pd.to_datetime(end_date)]

    return df

def calculate_cumulative_profit(df):
    """Calculate cumulative profit over time."""
    if df.empty:
        return pd.DataFrame()

    df_sorted = df.sort_values('logged_at').copy()
    df_sorted['cumulative_profit'] = df_sorted['profit'].fillna(0).cumsum()
    df_sorted['cumulative_roi'] = (df_sorted['cumulative_profit'] / df_sorted['bet_amount'].cumsum() * 100)

    return df_sorted

def create_performance_summary_cards(df):
    """Create metric cards for key performance indicators."""
    if df.empty:
        st.warning("No bets found for the selected period.")
        return

    # Calculate metrics
    settled_bets = df[df['outcome'].notna()]
    total_bets = len(settled_bets)

    if total_bets == 0:
        st.info("No settled bets in this period.")
        return

    wins = len(settled_bets[settled_bets['outcome'] == 'win'])
    losses = len(settled_bets[settled_bets['outcome'] == 'loss'])
    pushes = len(settled_bets[settled_bets['outcome'] == 'push'])

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    total_wagered = settled_bets['bet_amount'].sum()
    total_profit = settled_bets['profit'].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    avg_clv = settled_bets['clv'].mean() if 'clv' in settled_bets.columns else 0

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{wins}W-{losses}L-{pushes}P"
        )

    with col2:
        profit_color = "normal" if total_profit >= 0 else "inverse"
        st.metric(
            label="Total Profit",
            value=f"${total_profit:,.2f}",
            delta=f"{total_bets} bets",
            delta_color=profit_color
        )

    with col3:
        roi_color = "normal" if roi >= 0 else "inverse"
        st.metric(
            label="ROI",
            value=f"{roi:.2f}%",
            delta=f"${total_wagered:,.0f} wagered",
            delta_color=roi_color
        )

    with col4:
        clv_color = "normal" if avg_clv >= 0 else "inverse"
        st.metric(
            label="Avg CLV",
            value=f"{avg_clv:.2f}%",
            delta="Closing Line Value",
            delta_color=clv_color
        )

def create_profit_curve_chart(df):
    """Create cumulative profit curve."""
    df_cum = calculate_cumulative_profit(df)

    if df_cum.empty:
        return None

    fig = go.Figure()

    # Cumulative profit line
    fig.add_trace(go.Scatter(
        x=df_cum['logged_at'],
        y=df_cum['cumulative_profit'],
        mode='lines',
        name='Cumulative Profit',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))

    # Break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Cumulative Profit Over Time",
        xaxis_title="Date",
        yaxis_title="Profit ($)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig

def create_win_rate_trend_chart(df):
    """Create rolling win rate trend chart."""
    if df.empty or len(df) < 10:
        return None

    df_sorted = df[df['outcome'].notna()].sort_values('logged_at').copy()

    # Calculate rolling win rate (20-bet window)
    window = min(20, len(df_sorted) // 5)
    df_sorted['is_win'] = (df_sorted['outcome'] == 'win').astype(int)
    df_sorted['rolling_win_rate'] = df_sorted['is_win'].rolling(window=window, min_periods=1).mean() * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sorted['logged_at'],
        y=df_sorted['rolling_win_rate'],
        mode='lines',
        name=f'{window}-Bet Rolling Win Rate',
        line=dict(color='#3b82f6', width=2)
    ))

    # Break-even line (52.38% for -110 odds)
    fig.add_hline(y=52.38, line_dash="dash", line_color="yellow",
                  annotation_text="Break-even (52.38%)", opacity=0.7)

    fig.update_layout(
        title=f"Win Rate Trend ({window}-Bet Rolling Average)",
        xaxis_title="Date",
        yaxis_title="Win Rate (%)",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig

def create_clv_distribution_chart(df):
    """Create CLV distribution histogram."""
    if df.empty or 'clv' not in df.columns:
        return None

    df_clv = df[df['clv'].notna()].copy()

    if len(df_clv) < 5:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_clv['clv'],
        nbinsx=30,
        name='CLV Distribution',
        marker_color='#8b5cf6',
        opacity=0.7
    ))

    # Add mean line
    mean_clv = df_clv['clv'].mean()
    fig.add_vline(x=mean_clv, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean_clv:.2f}%", opacity=0.8)

    fig.update_layout(
        title="Closing Line Value (CLV) Distribution",
        xaxis_title="CLV (%)",
        yaxis_title="Count",
        template="plotly_dark",
        showlegend=False,
        height=400
    )

    return fig

def create_clv_vs_profit_scatter(df):
    """Create scatter plot of CLV vs actual profit."""
    if df.empty or 'clv' not in df.columns:
        return None

    df_settled = df[(df['outcome'].notna()) & (df['clv'].notna())].copy()

    if len(df_settled) < 10:
        return None

    # Color by outcome
    df_settled['color'] = df_settled['outcome'].map({
        'win': 'green',
        'loss': 'red',
        'push': 'gray'
    })

    fig = go.Figure()

    for outcome in ['win', 'loss', 'push']:
        df_outcome = df_settled[df_settled['outcome'] == outcome]
        if len(df_outcome) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=df_outcome['clv'],
            y=df_outcome['profit'],
            mode='markers',
            name=outcome.capitalize(),
            marker=dict(
                size=8,
                color=df_outcome['color'],
                opacity=0.6
            )
        ))

    fig.update_layout(
        title="CLV vs Profit Relationship",
        xaxis_title="CLV (%)",
        yaxis_title="Profit ($)",
        template="plotly_dark",
        hovermode='closest',
        height=400
    )

    return fig

def create_performance_by_edge_chart(df):
    """Create performance breakdown by model edge."""
    if df.empty or 'edge' not in df.columns:
        return None

    df_settled = df[df['outcome'].notna()].copy()

    if len(df_settled) < 10:
        return None

    # Create edge bins
    df_settled['edge_bin'] = pd.cut(
        df_settled['edge'] * 100,
        bins=[0, 5, 7, 10, 15, 100],
        labels=['5-7%', '7-10%', '10-15%', '15%+', '20%+'][:4]
    )

    # Calculate win rate by edge bin
    edge_performance = df_settled.groupby('edge_bin', observed=True).agg({
        'outcome': lambda x: (x == 'win').sum() / len(x) * 100,
        'profit': 'sum',
        'bet_amount': 'count'
    }).reset_index()

    edge_performance.columns = ['Edge Bin', 'Win Rate (%)', 'Profit', 'Bet Count']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=edge_performance['Edge Bin'],
            y=edge_performance['Win Rate (%)'],
            name='Win Rate',
            marker_color='#3b82f6'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=edge_performance['Edge Bin'],
            y=edge_performance['Bet Count'],
            name='Bet Count',
            mode='lines+markers',
            marker_color='#f59e0b',
            line=dict(width=3)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Performance by Model Edge",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Bet Count", secondary_y=True)

    return fig

def create_daily_performance_heatmap(df):
    """Create heatmap of performance by day of week and hour."""
    if df.empty:
        return None

    df_settled = df[df['outcome'].notna()].copy()

    if len(df_settled) < 20:
        return None

    df_settled['day_of_week'] = pd.to_datetime(df_settled['logged_at']).dt.day_name()
    df_settled['hour'] = pd.to_datetime(df_settled['logged_at']).dt.hour

    # Calculate win rate by day and hour
    pivot = df_settled.groupby(['day_of_week', 'hour']).agg({
        'outcome': lambda x: (x == 'win').sum() / len(x) * 100 if len(x) > 0 else 0
    }).reset_index()

    pivot_table = pivot.pivot(index='day_of_week', columns='hour', values='outcome')

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex([d for d in day_order if d in pivot_table.index])

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=52.38,  # Break-even point
        text=pivot_table.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Win Rate (%)")
    ))

    fig.update_layout(
        title="Win Rate Heatmap by Day & Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_dark",
        height=400
    )

    return fig

# Main dashboard
st.title("📊 Advanced NBA Betting Analytics")
st.markdown("Real-time performance tracking and comprehensive analytics")

# Load data
df_bets = load_bets_data(start_date, end_date)

# Filter by bet type
if bet_types and not df_bets.empty:
    df_bets = df_bets[df_bets['bet_type'].isin(bet_types)]

# Performance summary cards
st.markdown("### 📈 Performance Overview")
create_performance_summary_cards(df_bets)

# Create tabs for different analytics sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance Analytics",
    "💰 CLV Analysis",
    "🎯 Model Performance",
    "📅 Time-based Analytics"
])

with tab1:
    st.markdown("### Performance Trends")

    col1, col2 = st.columns(2)

    with col1:
        profit_chart = create_profit_curve_chart(df_bets)
        if profit_chart:
            st.plotly_chart(profit_chart, use_container_width=True)
        else:
            st.info("Not enough data for profit curve")

    with col2:
        winrate_chart = create_win_rate_trend_chart(df_bets)
        if winrate_chart:
            st.plotly_chart(winrate_chart, use_container_width=True)
        else:
            st.info("Not enough data for win rate trend")

    # Performance by edge
    st.markdown("### Performance by Model Edge")
    edge_chart = create_performance_by_edge_chart(df_bets)
    if edge_chart:
        st.plotly_chart(edge_chart, use_container_width=True)
    else:
        st.info("Not enough data for edge analysis")

with tab2:
    st.markdown("### CLV (Closing Line Value) Analysis")

    col1, col2 = st.columns(2)

    with col1:
        clv_dist_chart = create_clv_distribution_chart(df_bets)
        if clv_dist_chart:
            st.plotly_chart(clv_dist_chart, use_container_width=True)
        else:
            st.info("No CLV data available")

    with col2:
        clv_scatter = create_clv_vs_profit_scatter(df_bets)
        if clv_scatter:
            st.plotly_chart(clv_scatter, use_container_width=True)
        else:
            st.info("Not enough CLV data for scatter plot")

    # CLV statistics
    if not df_bets.empty and 'clv' in df_bets.columns:
        df_clv = df_bets[df_bets['clv'].notna()]

        if len(df_clv) > 0:
            st.markdown("### CLV Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                positive_clv_pct = (df_clv['clv'] > 0).sum() / len(df_clv) * 100
                st.metric("Positive CLV Rate", f"{positive_clv_pct:.1f}%")

            with col2:
                st.metric("Median CLV", f"{df_clv['clv'].median():.2f}%")

            with col3:
                st.metric("Best CLV", f"{df_clv['clv'].max():.2f}%")

            with col4:
                st.metric("Worst CLV", f"{df_clv['clv'].min():.2f}%")

with tab3:
    st.markdown("### Model Calibration & Accuracy")

    if not df_bets.empty and 'model_prob' in df_bets.columns:
        df_settled = df_bets[df_bets['outcome'].notna()].copy()

        if len(df_settled) >= 20:
            # Calibration plot
            df_settled['prob_bin'] = pd.cut(df_settled['model_prob'], bins=10)
            calibration = df_settled.groupby('prob_bin', observed=True).agg({
                'outcome': lambda x: (x == 'win').sum() / len(x),
                'model_prob': 'mean',
                'bet_amount': 'count'
            }).reset_index()

            fig = go.Figure()

            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ))

            # Actual calibration
            fig.add_trace(go.Scatter(
                x=calibration['model_prob'],
                y=calibration['outcome'],
                mode='markers+lines',
                name='Model Calibration',
                marker=dict(size=10, color='#10b981'),
                line=dict(width=2, color='#10b981')
            ))

            fig.update_layout(
                title="Model Calibration Plot",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Win Rate",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough settled bets for calibration analysis")
    else:
        st.info("Model probability data not available")

with tab4:
    st.markdown("### Time-based Performance Analysis")

    heatmap = create_daily_performance_heatmap(df_bets)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Not enough data for time-based analysis")

    # Recent bets table
    if not df_bets.empty:
        st.markdown("### Recent Bets")

        recent_bets = df_bets.sort_values('logged_at', ascending=False).head(20)

        # Format for display
        display_cols = ['logged_at', 'bet_type', 'bet_side', 'line', 'odds',
                       'bet_amount', 'outcome', 'profit', 'clv']

        available_cols = [col for col in display_cols if col in recent_bets.columns]

        st.dataframe(
            recent_bets[available_cols],
            use_container_width=True,
            height=400
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>NBA Betting Analytics Dashboard | Data updates every 60 seconds</p>
        <p>🔄 Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)
