"""
Multi-Strategy Configuration Loader

Loads and validates configuration from YAML file.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from dataclasses import dataclass
from loguru import logger


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    enabled: bool
    min_edge: float
    # Additional strategy-specific parameters stored in dict


@dataclass
class MultiStrategyConfig:
    """Complete multi-strategy configuration."""
    # Global settings
    bankroll: float
    kelly_fraction: float
    max_daily_exposure: float
    min_bet_size: float
    paper_trading: bool

    # Allocations
    allocation: Dict[str, float]
    daily_limits: Dict[str, int]

    # Strategy configs
    strategies: Dict[str, Dict]

    # Risk settings
    risk: Dict

    # Reporting
    reporting: Dict

    # Dev settings
    dev: Dict


def load_config(config_path: Optional[str] = None) -> MultiStrategyConfig:
    """
    Load multi-strategy configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config/multi_strategy_config.yaml)

    Returns:
        MultiStrategyConfig object
    """
    if config_path is None:
        # Default path
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "multi_strategy_config.yaml"

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return get_default_config()

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = MultiStrategyConfig(
            bankroll=config_dict['global']['bankroll'],
            kelly_fraction=config_dict['global']['kelly_fraction'],
            max_daily_exposure=config_dict['global']['max_daily_exposure'],
            min_bet_size=config_dict['global']['min_bet_size'],
            paper_trading=config_dict['global']['paper_trading'],
            allocation=config_dict['allocation'],
            daily_limits=config_dict['daily_limits'],
            strategies=config_dict['strategies'],
            risk=config_dict['risk'],
            reporting=config_dict['reporting'],
            dev=config_dict['dev'],
        )

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        return get_default_config()


def get_default_config() -> MultiStrategyConfig:
    """
    Get default configuration if config file not found.

    Returns:
        MultiStrategyConfig with sensible defaults
    """
    return MultiStrategyConfig(
        bankroll=1000.0,
        kelly_fraction=0.25,
        max_daily_exposure=0.15,
        min_bet_size=10.0,
        paper_trading=True,
        allocation={
            'totals': 0.30,
            'live': 0.20,
            'arbitrage': 0.25,
            'props': 0.15,
            'spread': 0.10,
        },
        daily_limits={
            'totals': 5,
            'live': 3,
            'arbitrage': 10,
            'props': 8,
            'spread': 3,
        },
        strategies={
            'totals': {'enabled': True, 'min_edge': 0.05},
            'live': {'enabled': False, 'min_edge': 0.05},
            'arbitrage': {'enabled': True, 'min_arb_profit': 0.01},
            'props': {'enabled': False, 'min_edge': 0.05},
            'spread': {'enabled': False, 'min_edge': 0.05},
        },
        risk={
            'drawdown_warning_threshold': 0.10,
            'drawdown_pause_threshold': 0.20,
            'drawdown_hard_stop': 0.30,
            'max_same_team_exposure': 0.15,
            'max_same_game_exposure': 0.10,
        },
        reporting={
            'send_daily_report': True,
            'track_by_strategy': True,
        },
        dev={
            'use_mock_data': False,
            'verbose': False,
            'dry_run': False,
        }
    )


def get_enabled_strategies(config: MultiStrategyConfig) -> List[str]:
    """
    Get list of enabled strategy names.

    Args:
        config: MultiStrategyConfig object

    Returns:
        List of enabled strategy names
    """
    enabled = []
    for name, settings in config.strategies.items():
        if settings.get('enabled', False):
            enabled.append(name)

    return enabled
