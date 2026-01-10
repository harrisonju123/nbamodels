#!/usr/bin/env python3
"""
Feature Importance Analysis

Analyzes which features drive model predictions and evaluates alternative data impact.

Usage:
    python scripts/analyze_feature_importance.py

    # Save detailed report
    python scripts/analyze_feature_importance.py --output reports/feature_importance.md

    # Skip SHAP analysis (faster)
    python scripts/analyze_feature_importance.py --skip-shap
"""

import sys
sys.path.insert(0, '.')

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def load_model_and_data():
    """Load trained model and test data."""
    logger.info("Loading model and metadata...")

    # Load model
    with open('models/spread_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load metadata
    with open('models/spread_model.metadata.json', 'r') as f:
        metadata = json.load(f)

    logger.info(f"Model trained: {metadata['trained_at']}")
    logger.info(f"Features: {metadata['n_features']}")
    logger.info(f"Test accuracy: {metadata['metrics']['accuracy']:.4f}")

    return model, metadata


def load_test_data(split_date='2023-10-01'):
    """Load test data for analysis."""
    logger.info("Loading test data...")

    # Load games and odds
    games = pd.read_parquet('data/raw/games.parquet')
    odds = pd.read_csv('data/raw/historical_odds.csv')

    # Team mapping
    TEAM_MAP = {
        'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
        'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
        'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
        'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'nop': 'NOP',
        'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI',
        'phx': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sa': 'SAS', 'tor': 'TOR',
        'uta': 'UTA', 'was': 'WAS',
    }

    odds['home_team'] = odds['home'].map(TEAM_MAP)
    odds['away_team'] = odds['away'].map(TEAM_MAP)
    odds['date'] = pd.to_datetime(odds['date'])

    # Merge
    games['date'] = pd.to_datetime(games['date'])
    games_with_odds = games.merge(
        odds[['date', 'home_team', 'away_team', 'spread', 'total']],
        on=['date', 'home_team', 'away_team'],
        how='left'
    ).rename(columns={'spread': 'spread_home'})

    # Build features
    from src.features.game_features import GameFeatureBuilder
    from src.features.team_features import TeamFeatureBuilder

    # Disable Four Factors (has data leakage)
    original_add_four_factors = TeamFeatureBuilder.add_four_factors
    TeamFeatureBuilder.add_four_factors = lambda self, df: df

    builder = GameFeatureBuilder()
    features = builder.build_game_features(games_with_odds.copy())

    # Add targets
    features['spread_home'] = games_with_odds['spread_home']
    features['total'] = games_with_odds['total']
    features['point_diff'] = games_with_odds['home_score'] - games_with_odds['away_score']
    features['home_covers'] = (features['point_diff'] + features['spread_home'] > 0).astype(int)

    # Filter
    features = features[
        features['point_diff'].notna() &
        features['spread_home'].notna()
    ].copy()

    # Split
    split_date = pd.to_datetime(split_date)
    test_df = features[features['date'] >= split_date].copy()

    logger.info(f"Test data: {len(test_df)} games")

    return test_df


def extract_xgboost_importance(model, feature_cols):
    """Extract feature importance from XGBoost model."""
    logger.info("\nExtracting XGBoost feature importances...")

    # Get XGBoost model from dual model
    xgb_model = model.xgb

    # Get importance scores
    importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
    importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
    importance_cover = xgb_model.get_booster().get_score(importance_type='cover')

    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'feature': list(importance_gain.keys()),
        'gain': list(importance_gain.values()),
    })

    # Add other importance types
    importance_df['weight'] = importance_df['feature'].map(importance_weight).fillna(0)
    importance_df['cover'] = importance_df['feature'].map(importance_cover).fillna(0)

    # Convert feature names (f0, f1, etc.) to actual names
    feature_map = {f'f{i}': name for i, name in enumerate(feature_cols)}
    importance_df['feature_name'] = importance_df['feature'].map(feature_map)

    # Sort by gain
    importance_df = importance_df.sort_values('gain', ascending=False)

    # Normalize
    importance_df['gain_pct'] = importance_df['gain'] / importance_df['gain'].sum() * 100
    importance_df['weight_pct'] = importance_df['weight'] / importance_df['weight'].sum() * 100
    importance_df['cover_pct'] = importance_df['cover'] / importance_df['cover'].sum() * 100

    logger.info(f"Extracted importance for {len(importance_df)} features")

    return importance_df


def categorize_features(feature_names):
    """Categorize features by type."""
    categories = {
        'Team Stats (Rolling)': [],
        'Team Context': [],
        'Differentials': [],
        'Lineup/Injury': [],
        'Matchup (H2H)': [],
        'Elo': [],
        'Referee': [],
        'News': [],
        'Sentiment': [],
        'Schedule': [],
        'Odds': [],
        'Season': [],
    }

    for feat in feature_names:
        if 'ref_' in feat:
            categories['Referee'].append(feat)
        elif 'news_' in feat:
            categories['News'].append(feat)
        elif 'sentiment' in feat:
            categories['Sentiment'].append(feat)
        elif 'lineup' in feat or 'injury' in feat or 'injured' in feat or 'missing' in feat or 'impact' in feat:
            categories['Lineup/Injury'].append(feat)
        elif 'elo' in feat:
            categories['Elo'].append(feat)
        elif feat.startswith('diff_'):
            categories['Differentials'].append(feat)
        elif 'h2h_' in feat or 'division' in feat or 'conference' in feat or 'rivalry' in feat:
            categories['Matchup (H2H)'].append(feat)
        elif 'b2b' in feat or 'rest' in feat:
            categories['Schedule'].append(feat)
        elif feat in ['spread_home', 'total']:
            categories['Odds'].append(feat)
        elif 'season' in feat or 'allstar' in feat:
            categories['Season'].append(feat)
        elif 'travel' in feat or 'home_season_win_pct' in feat or 'away_season_win_pct' in feat:
            categories['Team Context'].append(feat)
        else:
            categories['Team Stats (Rolling)'].append(feat)

    return categories


def analyze_category_importance(importance_df, feature_cols):
    """Analyze importance by feature category."""
    logger.info("\nAnalyzing importance by category...")

    # Categorize
    categories = categorize_features(importance_df['feature_name'].tolist())

    # Calculate category importance
    category_importance = []

    for category, features in categories.items():
        if not features:
            continue

        cat_df = importance_df[importance_df['feature_name'].isin(features)]

        category_importance.append({
            'category': category,
            'n_features': len(features),
            'total_gain': cat_df['gain'].sum(),
            'total_gain_pct': cat_df['gain_pct'].sum(),
            'avg_gain': cat_df['gain'].mean(),
            'max_gain': cat_df['gain'].max(),
            'top_feature': cat_df.iloc[0]['feature_name'] if len(cat_df) > 0 else None,
            'top_feature_gain_pct': cat_df.iloc[0]['gain_pct'] if len(cat_df) > 0 else 0,
        })

    category_df = pd.DataFrame(category_importance)
    category_df = category_df.sort_values('total_gain_pct', ascending=False)

    return category_df


def analyze_correlations(test_df, feature_cols):
    """Analyze feature correlations."""
    logger.info("\nAnalyzing feature correlations...")

    X = test_df[feature_cols].fillna(0)

    # Calculate correlation matrix
    corr_matrix = X.corr()

    # Find highly correlated pairs
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:  # High correlation threshold
                high_corr.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val,
                })

    high_corr_df = pd.DataFrame(high_corr)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values('correlation', ascending=False)

    logger.info(f"Found {len(high_corr_df)} highly correlated pairs (r > 0.8)")

    return corr_matrix, high_corr_df


def evaluate_alternative_data_impact(model, test_df, feature_cols):
    """Evaluate impact of removing alternative data features."""
    logger.info("\nEvaluating alternative data impact...")

    # Identify alternative data features
    alt_data_features = [f for f in feature_cols if
                        'ref_' in f or 'news_' in f or
                        'sentiment' in f or 'lineup' in f]

    # Features without alt data
    core_features = [f for f in feature_cols if f not in alt_data_features]

    logger.info(f"Alternative data features: {len(alt_data_features)}")
    logger.info(f"Core features: {len(core_features)}")

    # Prepare data
    X_full = test_df[feature_cols].fillna(0)
    X_core = test_df[core_features].fillna(0)
    y = test_df['home_covers']

    # Predictions with full model
    pred_full = model.predict_proba(X_full)
    if isinstance(pred_full, dict):
        pred_full = pred_full['ensemble']

    # Predictions with core features only (zero out alt data)
    X_no_alt = X_full.copy()
    for feat in alt_data_features:
        X_no_alt[feat] = 0

    pred_no_alt = model.predict_proba(X_no_alt)
    if isinstance(pred_no_alt, dict):
        pred_no_alt = pred_no_alt['ensemble']

    # Calculate metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    metrics_full = {
        'accuracy': accuracy_score(y, (pred_full > 0.5).astype(int)),
        'auc': roc_auc_score(y, pred_full),
        'log_loss': log_loss(y, pred_full),
    }

    metrics_no_alt = {
        'accuracy': accuracy_score(y, (pred_no_alt > 0.5).astype(int)),
        'auc': roc_auc_score(y, pred_no_alt),
        'log_loss': log_loss(y, pred_no_alt),
    }

    # Calculate impact
    impact = {
        'accuracy_change': metrics_full['accuracy'] - metrics_no_alt['accuracy'],
        'auc_change': metrics_full['auc'] - metrics_no_alt['auc'],
        'log_loss_change': metrics_full['log_loss'] - metrics_no_alt['log_loss'],
    }

    logger.info(f"Full model accuracy: {metrics_full['accuracy']:.4f}")
    logger.info(f"No alt data accuracy: {metrics_no_alt['accuracy']:.4f}")
    logger.info(f"Impact: {impact['accuracy_change']:+.4f} ({impact['accuracy_change']*100:+.2f}%)")

    return {
        'metrics_full': metrics_full,
        'metrics_no_alt': metrics_no_alt,
        'impact': impact,
        'alt_data_features': alt_data_features,
        'core_features': core_features,
    }


def create_report(importance_df, category_df, high_corr_df, alt_data_impact, output_path=None):
    """Create feature importance report."""
    logger.info("\nCreating feature importance report...")

    report = []
    report.append("# Feature Importance Analysis")
    report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Model:** models/spread_model.pkl")
    report.append("\n---\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"**Total Features:** {len(importance_df)}")
    report.append(f"**Alternative Data Features:** {len(alt_data_impact['alt_data_features'])}")
    report.append(f"**Core Features:** {len(alt_data_impact['core_features'])}")
    report.append("")

    # Alternative Data Impact
    report.append("### Alternative Data Impact\n")
    impact = alt_data_impact['impact']
    report.append(f"- **Accuracy Change:** {impact['accuracy_change']:+.4f} ({impact['accuracy_change']*100:+.2f}%)")
    report.append(f"- **AUC Change:** {impact['auc_change']:+.4f}")
    report.append(f"- **Log Loss Change:** {impact['log_loss_change']:+.4f} (lower is better)")
    report.append("")

    if impact['accuracy_change'] > 0:
        report.append("✅ **Alternative data improves model accuracy**")
    else:
        report.append("⚠️ **Alternative data does not improve accuracy** (may be neutral/default values)")

    report.append("\n---\n")

    # Category Importance
    report.append("## Feature Category Importance\n")
    report.append("| Category | Features | Total Gain % | Top Feature | Top Gain % |")
    report.append("|----------|----------|--------------|-------------|------------|")

    for _, row in category_df.head(15).iterrows():
        report.append(
            f"| {row['category']} | {row['n_features']} | "
            f"{row['total_gain_pct']:.2f}% | {row['top_feature'] or 'N/A'} | "
            f"{row['top_feature_gain_pct']:.2f}% |"
        )

    report.append("\n---\n")

    # Top 30 Features
    report.append("## Top 30 Most Important Features\n")
    report.append("| Rank | Feature | Gain % | Category |")
    report.append("|------|---------|--------|----------|")

    # Categorize for display
    all_categories = categorize_features(importance_df['feature_name'].tolist())
    feature_to_category = {}
    for cat, features in all_categories.items():
        for feat in features:
            feature_to_category[feat] = cat

    for idx, row in importance_df.head(30).iterrows():
        feat_name = row['feature_name']
        category = feature_to_category.get(feat_name, 'Other')
        report.append(
            f"| {len([r for r in importance_df.head(30).itertuples() if r[0] <= idx])} | "
            f"{feat_name} | {row['gain_pct']:.2f}% | {category} |"
        )

    report.append("\n---\n")

    # Alternative Data Features
    report.append("## Alternative Data Feature Rankings\n")

    alt_features = importance_df[importance_df['feature_name'].isin(alt_data_impact['alt_data_features'])]

    if not alt_features.empty:
        report.append("| Feature | Gain % | Overall Rank |")
        report.append("|---------|--------|--------------|")

        for _, row in alt_features.iterrows():
            overall_rank = importance_df[importance_df['feature_name'] == row['feature_name']].index[0] + 1
            report.append(f"| {row['feature_name']} | {row['gain_pct']:.2f}% | #{overall_rank} |")
    else:
        report.append("*No alternative data features found in importance rankings.*")

    report.append("\n---\n")

    # High Correlations
    report.append("## Highly Correlated Features (r > 0.8)\n")

    if not high_corr_df.empty and len(high_corr_df) > 0:
        report.append("| Feature 1 | Feature 2 | Correlation |")
        report.append("|-----------|-----------|-------------|")

        for _, row in high_corr_df.head(20).iterrows():
            report.append(f"| {row['feature1']} | {row['feature2']} | {row['correlation']:.3f} |")

        report.append(f"\n**Total correlated pairs:** {len(high_corr_df)}")
    else:
        report.append("*No highly correlated feature pairs found.*")

    report.append("\n---\n")

    # Insights
    report.append("## Key Insights\n")

    # Top category
    top_category = category_df.iloc[0]
    report.append(f"1. **{top_category['category']}** is the most important category ({top_category['total_gain_pct']:.1f}% of total gain)")

    # Alternative data
    alt_cat_df = category_df[category_df['category'].isin(['Referee', 'News', 'Sentiment', 'Lineup/Injury'])]
    if not alt_cat_df.empty:
        total_alt_gain = alt_cat_df['total_gain_pct'].sum()
        report.append(f"2. Alternative data features contribute **{total_alt_gain:.2f}%** of total model gain")

    # Most important feature
    top_feature = importance_df.iloc[0]
    report.append(f"3. Most important single feature: **{top_feature['feature_name']}** ({top_feature['gain_pct']:.2f}% gain)")

    # Feature concentration
    top_10_gain = importance_df.head(10)['gain_pct'].sum()
    report.append(f"4. Top 10 features account for **{top_10_gain:.1f}%** of total gain")

    report.append("\n---\n")

    # Save report
    report_text = "\n".join(report)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.success(f"Report saved to {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--output', type=str, help='Output path for report')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)

    try:
        # Load model and data
        model, metadata = load_model_and_data()
        feature_cols = metadata['feature_columns']

        # Load test data
        test_df = load_test_data(split_date=metadata['split_date'])

        # Extract importance
        importance_df = extract_xgboost_importance(model, feature_cols)

        # Category analysis
        category_df = analyze_category_importance(importance_df, feature_cols)

        # Correlation analysis
        corr_matrix, high_corr_df = analyze_correlations(test_df, feature_cols)

        # Alternative data impact
        alt_data_impact = evaluate_alternative_data_impact(model, test_df, feature_cols)

        # Create report
        output_path = args.output or 'FEATURE_IMPORTANCE_ANALYSIS.md'
        report = create_report(importance_df, category_df, high_corr_df, alt_data_impact, output_path)

        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nReport saved to: {output_path}")
        print(f"\nTop 5 Categories by Importance:")
        for idx, row in category_df.head(5).iterrows():
            print(f"  {idx+1}. {row['category']}: {row['total_gain_pct']:.2f}%")

        print(f"\nTop 5 Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature_name']}: {row['gain_pct']:.2f}%")

        print("\n" + "=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
