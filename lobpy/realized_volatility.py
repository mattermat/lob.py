"""
Realized volatility computation from trade data.

Realized volatility = sqrt(sum of squared log returns) over a rolling time window.
"""

import numpy as np
import pandas as pd


def realized_vol_scalar(trades):
    """
    Compute realized volatility from a list of Trade objects.

    Realized volatility = sqrt(sum of squared log returns), where log returns
    are computed on trade prices sorted by timestamp.

    Returns nan if fewer than 2 trades.
    """
    if len(trades) < 2:
        return float("nan")
    prices = [t.price for t in sorted(trades, key=lambda t: t.timestamp)]
    log_returns = np.diff(np.log(prices))
    return float(np.sqrt(np.sum(log_returns**2)))


def realized_vol(trades, window_size=None, rolling_items=None):
    """
    Compute realized volatility from trade prices.

    Realized volatility = sqrt(sum of squared log returns) over the
    sequence of trade prices sorted by timestamp.

    Args:
        trades:        List of Trade objects.
        window_size:   If None, returns a scalar over all trades.
                       If given, returns a pd.Series of realized vol values
                       computed over rolling windows of the given size.
        rolling_items: Callable that takes window_size and yields
                       (timestamp, trades_list) pairs. Required if
                       window_size is given.

    Returns:
        float (scalar) or pd.Series indexed by end timestamp.
    """
    if window_size is None:
        return realized_vol_scalar(trades)
    vals = {}
    for ts, window_trades in rolling_items(window_size):
        vals[ts] = realized_vol_scalar(window_trades)
    return pd.Series(vals, name="realized_vol")
