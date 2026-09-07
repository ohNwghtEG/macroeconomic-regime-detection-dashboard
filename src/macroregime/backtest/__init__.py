"""Walk-forward backtesting with costs, benchmarks and significance testing."""

from .engine import backtest, benchmark_weights, probability_weighted_allocation
from .stats import bootstrap_sharpe_difference, deflated_sharpe_ratio, performance_summary

__all__ = [
    "backtest",
    "probability_weighted_allocation",
    "benchmark_weights",
    "performance_summary",
    "bootstrap_sharpe_difference",
    "deflated_sharpe_ratio",
]
