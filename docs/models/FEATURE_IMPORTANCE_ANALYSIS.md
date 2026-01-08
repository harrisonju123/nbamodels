# Feature Importance Analysis

**Date:** 2026-01-04 18:21
**Model:** models/spread_model.pkl

---

## Executive Summary

**Total Features:** 58
**Alternative Data Features:** 19
**Core Features:** 80

### Alternative Data Impact

- **Accuracy Change:** -0.0350 (-3.50%)
- **AUC Change:** -0.0074
- **Log Loss Change:** -0.0998 (lower is better)

⚠️ **Alternative data does not improve accuracy** (may be neutral/default values)

---

## Feature Category Importance

| Category | Features | Total Gain % | Top Feature | Top Gain % |
|----------|----------|--------------|-------------|------------|
| Team Stats (Rolling) | 21 | 33.94% | away_win_rate_5g | 2.40% |
| Differentials | 11 | 17.46% | diff_net_rating_5g | 1.87% |
| Matchup (H2H) | 6 | 9.51% | h2h_home_win_rate | 2.17% |
| Schedule | 4 | 7.34% | b2b_advantage | 2.32% |
| Odds | 2 | 7.20% | spread_home | 5.53% |
| Season | 3 | 5.56% | home_home_season_win_pct | 2.04% |
| Elo | 3 | 5.53% | away_elo | 1.92% |
| Team Context | 3 | 5.43% | travel_diff | 2.20% |
| Lineup/Injury | 3 | 4.89% | lineup_impact_diff | 1.81% |
| News | 2 | 3.15% | home_news_volume_24h | 1.83% |

---

## Top 30 Most Important Features

| Rank | Feature | Gain % | Category |
|------|---------|--------|----------|
| 29 | spread_home | 5.53% | Odds |
| 9 | away_win_rate_5g | 2.40% | Team Stats (Rolling) |
| 11 | away_pace_5g | 2.36% | Team Stats (Rolling) |
| 28 | b2b_advantage | 2.32% | Schedule |
| 18 | rest_diff | 2.21% | Schedule |
| 19 | travel_diff | 2.20% | Team Context |
| 22 | h2h_home_win_rate | 2.17% | Matchup (H2H) |
| 3 | home_win_rate_20g | 2.13% | Team Stats (Rolling) |
| 7 | home_home_season_win_pct | 2.04% | Season |
| 8 | home_away_season_win_pct | 1.97% | Season |
| 25 | away_elo | 1.92% | Elo |
| 23 | h2h_recency_weighted_margin | 1.92% | Matchup (H2H) |
| 14 | diff_net_rating_5g | 1.87% | Differentials |
| 6 | home_pace_20g | 1.86% | Team Stats (Rolling) |
| 27 | home_news_volume_24h | 1.83% | News |
| 26 | elo_diff | 1.82% | Elo |
| 4 | home_pts_against_20g | 1.81% | Team Stats (Rolling) |
| 21 | lineup_impact_diff | 1.81% | Lineup/Injury |
| 16 | diff_pts_for_20g | 1.79% | Differentials |
| 24 | home_elo | 1.78% | Elo |
| 12 | away_rest_days | 1.77% | Schedule |
| 17 | diff_pace_20g | 1.70% | Differentials |
| 20 | away_lineup_impact | 1.69% | Lineup/Injury |
| 30 | total | 1.68% | Odds |
| 13 | away_travel_distance | 1.68% | Team Context |
| 5 | home_net_rating_20g | 1.67% | Team Stats (Rolling) |
| 1 | home_net_rating_5g | 1.67% | Team Stats (Rolling) |
| 2 | home_pace_5g | 1.67% | Team Stats (Rolling) |
| 10 | away_pts_against_5g | 1.65% | Team Stats (Rolling) |
| 15 | diff_pace_5g | 1.64% | Differentials |

---

## Alternative Data Feature Rankings

| Feature | Gain % | Overall Rank |
|---------|--------|--------------|
| home_news_volume_24h | 1.83% | #54 |
| lineup_impact_diff | 1.81% | #44 |
| away_lineup_impact | 1.69% | #43 |
| home_lineup_impact | 1.39% | #42 |
| news_volume_diff | 1.32% | #55 |

---

## Highly Correlated Features (r > 0.8)

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| home_away_season_win_pct | away_away_season_win_pct | 1.000 |
| home_is_b2b | home_b2b | 1.000 |
| home_lineup_impact | home_total_impact | 1.000 |
| home_season_progress | away_season_progress | 1.000 |
| diff_win_streak_5g | diff_win_streak_20g | 1.000 |
| home_win_streak_5g | home_win_streak_20g | 1.000 |
| away_lineup_impact | away_total_impact | 1.000 |
| away_is_b2b | away_b2b | 1.000 |
| home_is_post_allstar | away_is_post_allstar | 1.000 |
| away_win_streak_5g | away_win_streak_20g | 1.000 |
| home_home_season_win_pct | away_home_season_win_pct | 1.000 |
| h2h_home_margin | h2h_recency_weighted_margin | 0.979 |
| elo_diff | elo_prob | 0.979 |
| rest_diff | rest_advantage | 0.955 |
| home_win_rate_20g | home_net_rating_20g | 0.920 |
| diff_net_rating_20g | diff_win_rate_20g | 0.920 |
| away_pts_against_20g | away_pace_20g | 0.918 |
| away_win_rate_20g | away_net_rating_20g | 0.918 |
| away_pts_for_20g | away_pace_20g | 0.917 |
| diff_win_rate_20g | elo_diff | 0.910 |

**Total correlated pairs:** 52

---

## Key Insights

1. **Team Stats (Rolling)** is the most important category (33.9% of total gain)
2. Alternative data features contribute **8.03%** of total model gain
3. Most important single feature: **spread_home** (5.53% gain)
4. Top 10 features account for **25.3%** of total gain

---
