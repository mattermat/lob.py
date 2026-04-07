"""
VPIN (Volume-Synchronized Probability of Informed Trading) computation.
"""

import pandas as pd

_DEFAULT_NUM_BUCKETS = 50


def _fill_buckets(trades, bucket_size, include_partial=False):
    """
    Partition a sorted list of Trade objects into fixed-volume buckets.

    Each bucket accumulates exactly bucket_size total volume. Trades are split
    across bucket boundaries when a single trade straddles two buckets.

    Args:
        trades:          List of Trade objects, sorted by timestamp.
        bucket_size:     Target volume per bucket.
        include_partial: If True, append the last partial bucket even if its
                         total volume is less than bucket_size.

    Returns:
        pd.DataFrame indexed by bucket number with columns:
            buy_volume  – volume from buy-aggressor trades
            sell_volume – volume from sell-aggressor trades
    """
    buy_vols, sell_vols = [], []
    current_buy = current_sell = 0.0
    remaining = bucket_size

    for trade in trades:
        vol = trade.volume
        while vol > 0:
            fill = min(vol, remaining)
            if trade.side == "b":
                current_buy += fill
            else:
                current_sell += fill
            vol -= fill
            remaining -= fill
            if remaining < 1e-12:
                buy_vols.append(current_buy)
                sell_vols.append(current_sell)
                current_buy = current_sell = 0.0
                remaining = bucket_size

    if include_partial and (current_buy + current_sell) > 0:
        buy_vols.append(current_buy)
        sell_vols.append(current_sell)

    return pd.DataFrame({"buy_volume": buy_vols, "sell_volume": sell_vols})


def _vpin_from_buckets(df, bucket_size):
    """Compute VPIN scalar from a volume-buckets DataFrame."""
    if df.empty:
        return float("nan")
    n = len(df)
    imbalance = (df["buy_volume"] - df["sell_volume"]).abs().sum()
    return float(imbalance / (n * bucket_size))


def volume_buckets(trades, bucket_size=None, include_partial=False):
    """
    Partition trades into fixed-volume buckets.

    Trades are processed in timestamp order. Each bucket accumulates exactly
    bucket_size total volume; a trade is split across bucket boundaries when
    it would overflow the current bucket.

    Args:
        trades:          List of Trade objects.
        bucket_size:     Volume per bucket. If None, computed as
                         total_volume / 50.
        include_partial: If True, the last partial bucket (volume < bucket_size)
                         is included. Default False.

    Returns:
        pd.DataFrame indexed by bucket number (0-based) with columns:
            buy_volume  – volume from buy-aggressor trades
            sell_volume – volume from sell-aggressor trades
    """
    trades = sorted(trades, key=lambda t: t.timestamp)
    if not trades:
        return pd.DataFrame(columns=["buy_volume", "sell_volume"])
    if bucket_size is None:
        total_vol = sum(t.volume for t in trades)
        bucket_size = total_vol / _DEFAULT_NUM_BUCKETS
    return _fill_buckets(trades, bucket_size, include_partial)


def vpin(trades, window_size=None, bucket_size=None, rolling_items=None):
    """
    Compute VPIN (Volume-Synchronized Probability of Informed Trading).

    VPIN = Σ|V_buy[i] - V_sell[i]| / (n × bucket_size)

    where the sum runs over all complete volume buckets.

    Args:
        trades:       List of Trade objects.
        window_size:  If None, returns a scalar over all trades.
                      If given, returns a pd.Series of VPIN values computed
                      over rolling time windows of the given size.
        bucket_size:  Volume per bucket. If None, computed once as
                      total_volume / 50 and reused across all windows.
        rolling_items: Callable that takes window_size and yields
                       (timestamp, trades_list) pairs. Required if
                       window_size is given.

    Returns:
        float (scalar) or pd.Series indexed by end timestamp.
    """
    if bucket_size is None:
        total_vol = sum(t.volume for t in trades)
        bucket_size = total_vol / _DEFAULT_NUM_BUCKETS if total_vol > 0 else 1.0

    if window_size is None:
        df = volume_buckets(trades, bucket_size)
        return _vpin_from_buckets(df, bucket_size)

    vals = {}
    for ts, window_trades in rolling_items(window_size):
        df = volume_buckets(window_trades, bucket_size)
        vals[ts] = _vpin_from_buckets(df, bucket_size)
    return pd.Series(vals, name="vpin")
