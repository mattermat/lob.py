"""
OHLC (Open, High, Low, Close) candle computation from trade data.
"""

import pandas as pd

OHLC_PERIODS = {
    "1s": 1,
    "5s": 5,
    "1m": 60,
    "15m": 900,
    "1h": 3_600,
    "24h": 86_400,
}

TS_UNITS = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
    "ns": 1_000_000_000,
}


def ohlc(trades, period, timestamp_unit="ns"):
    """
    Compute OHLC candles from trade data.

    Trades are bucketed into fixed time windows of the given period.

    Args:
        trades:          List of Trade objects.
        period:          One of '1s', '5s', '1m', '15m', '1h', '24h'.
        timestamp_unit:  Unit of timestamps. One of 's', 'ms', 'us', 'ns'.

    Returns:
        pd.DataFrame indexed by candle-open timestamp with columns:
        open, high, low, close, volume, count.
    """
    if period not in OHLC_PERIODS:
        raise ValueError(f"Unknown period '{period}'. Accepted: {list(OHLC_PERIODS)}")
    if not trades:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "count"])

    period_ts = OHLC_PERIODS[period] * TS_UNITS[timestamp_unit]
    buckets = {}
    for trade in sorted(trades, key=lambda t: t.timestamp):
        bucket = (trade.timestamp // period_ts) * period_ts
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(trade)

    rows = {}
    for bucket_ts in sorted(buckets):
        bucket_trades = buckets[bucket_ts]
        prices = [t.price for t in bucket_trades]
        rows[bucket_ts] = {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(t.volume for t in bucket_trades),
            "count": len(bucket_trades),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "timestamp"
    return df
