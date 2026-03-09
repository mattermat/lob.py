"""
TimeLine (TL) - combines LOBts with trade events.
"""

import numpy as np
import pandas as pd

from .lobts import LOBts
from .gueant import GueantAccessor

# Mapping from period string to seconds
_OHLC_PERIODS = {
    '1s':  1,
    '5s':  5,
    '1m':  60,
    '15m': 900,
    '1h':  3_600,
    '24h': 86_400,
}

# Timestamp unit → number of units per second
def _realized_vol(trades):
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
    return float(np.sqrt(np.sum(log_returns ** 2)))


_TS_UNITS = {
    's':  1,
    'ms': 1_000,
    'us': 1_000_000,
    'ns': 1_000_000_000,
}


class Trade:
    """A single trade event."""

    __slots__ = ("timestamp", "side", "price", "volume")

    def __init__(self, timestamp, side, price, volume):
        self.timestamp = timestamp
        self.side = side      # 'b' (buy aggressor) or 's' (sell aggressor)
        self.price = price
        self.volume = volume

    def __repr__(self):
        return f"<Trade ts={self.timestamp} {self.side} {self.volume}@{self.price}>"


class TL:
    """
    TimeLine: a unified container for LOB time series and trade events.

    Combines LOBts (order book snapshots/updates) with trade records to
    enable analysis that mixes order book and execution data.

    Args:
        name: Optional identifier for this timeline
        tick_size: Minimum price increment
        lob_mode: 'delta' (incremental updates) or
                  'snapshot' (full snapshots at each update)
        update_type: 'realtime' (sparse updates/snapshots) or
                     'fixed' (update/snapshots at regular intervals)
        timestamp_unit: Unit of all timestamps. One of 's', 'ms', 'us', 'ns'.
                        Defaults to 'ns' (nanoseconds).
    """

    def __init__(self, name=None, tick_size=1, lob_mode='delta', update_type='realtime',
                 timestamp_unit='ns'):
        if timestamp_unit not in _TS_UNITS:
            raise ValueError(
                f"Unknown timestamp_unit '{timestamp_unit}'. Accepted: {list(_TS_UNITS)}"
            )
        if name is None:
            name = f"tl{id(self)}"
        self.name = name
        self.tick_size = tick_size
        self.lob_mode = lob_mode
        self.update_type = update_type
        self.timestamp_unit = timestamp_unit
        _lobts_mode = 'latest' if lob_mode == 'snapshot' else 'delta'
        self._lobts = LOBts(name=name, tick_size=tick_size, mode=_lobts_mode)
        self._trades = []

    # ------------------------------------------------------------------
    # LOB methods
    # ------------------------------------------------------------------

    @property
    def lob(self):
        """Return the underlying LOBts, indexable by timestamp."""
        return self._lobts

    def add_lob_snapshot(self, timestamp, bids, asks):
        """
        Record a full LOB snapshot at the given timestamp.

        Args:
            timestamp: Event timestamp
            bids: List of (price, quantity) tuples, bid side
            asks: List of (price, quantity) tuples, ask side
        """
        self._lobts.set_snapshot(bids, asks, timestamp)

    def add_lob_update(self, timestamp, updates):
        """
        Apply LOB updates at the given timestamp.

        In 'delta' mode, updates are incremental changes on top of the previous snapshot.
        In 'snapshot' mode, updates replace the book entirely (full snapshot semantics).

        Args:
            timestamp: Event timestamp
            updates: List of (side, price, quantity) tuples.
                     side is 'b'/'bid' or 'a'/'ask'.
                     quantity=0 removes the level.
        """
        self._lobts.set_updates(updates, timestamp)

    # ------------------------------------------------------------------
    # Trade methods
    # ------------------------------------------------------------------

    @property
    def trades(self):
        """Return list of Trade objects in insertion order."""
        return self._trades

    def add_trade(self, timestamp, side, price, volume):
        """
        Record a single trade event.

        Args:
            timestamp: Event timestamp
            side: 'b' (buy aggressor, takes from asks) or
                  's' (sell aggressor, takes from bids)
            price: Execution price
            volume: Trade size
        """
        self._trades.append(Trade(timestamp, side, price, volume))

    def add_trades(self, timestamp, trades):
        """
        Record multiple trade events at the same timestamp.

        Args:
            timestamp: Event timestamp
            trades: List of (side, price, volume) tuples
        """
        for side, price, volume in trades:
            self._trades.append(Trade(timestamp, side, price, volume))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _iter_rows(self):
        """Yield (timestamp, type, side, level, price, size) rows sorted by timestamp."""
        lob_rows = []
        for ts in self._lobts.timestamps:
            lob = self._lobts[ts]
            for level, (price, size) in enumerate(lob._bids.items()):
                lob_rows.append((ts, "lob", "b", level, price, size))
            for level, (price, size) in enumerate(lob._asks.items()):
                lob_rows.append((ts, "lob", "a", level, price, size))

        trade_rows = [
            (t.timestamp, "trade", t.side, float("nan"), t.price, t.volume)
            for t in self._trades
        ]

        return sorted(lob_rows + trade_rows, key=lambda r: r[0])

    def to_pd(self):
        """
        Export to pandas DataFrame.

        Returns:
            DataFrame with columns: timestamp, type, side, level, price, size.
            type is 'lob' or 'trade'; level is NaN for trade rows.
        """
        return pd.DataFrame(
            self._iter_rows(),
            columns=["timestamp", "type", "side", "level", "price", "size"],
        )

    def to_np(self):
        """
        Export to numpy array.

        Returns:
            Object array with columns: timestamp, type, side, level, price, size.
            type is 'lob' or 'trade'; level is NaN for trade rows.
        """
        rows = self._iter_rows()
        if not rows:
            return np.empty((0, 6), dtype=object)
        return np.array(rows, dtype=object)

    def __getitem__(self, ts_slice):
        """
        Return a new TL sliced to the given timestamp range.

        Args:
            ts_slice: slice object with start/stop timestamps (both inclusive)

        Returns:
            New TL containing only events in [start, stop]
        """
        if not isinstance(ts_slice, slice):
            raise TypeError("TL indexing requires a slice (tl[start:stop])")
        start, stop = ts_slice.start, ts_slice.stop
        result = TL(
            name=f"{self.name}_slice",
            tick_size=self.tick_size,
            lob_mode=self.lob_mode,
            update_type=self.update_type,
            timestamp_unit=self.timestamp_unit,
        )
        result._lobts = self._lobts.get_range(start, stop)
        result._trades = [
            t for t in self._trades
            if (start is None or t.timestamp >= start)
            and (stop is None or t.timestamp <= stop)
        ]
        return result

    def __len__(self):
        """Return total number of events (LOB snapshots + trades)."""
        return len(self._lobts) + len(self._trades)

    @property
    def timestamps(self):
        """Return sorted list of all event timestamps (LOB and trades)."""
        ts = sorted(
            set(self._lobts.timestamps) | {t.timestamp for t in self._trades}
        )
        return ts

    def rolling(self, window_size):
        """
        Yield TL slices over a rolling window.

        For each event timestamp ts, yields tl[ts-window_size : ts].

        Args:
            window_size: window size in time units (same units as timestamps)
        """
        for ts in self.timestamps:
            yield self[ts - window_size:ts]

    def _rolling_items(self, window_size):
        """Yield (end_timestamp, TL slice) pairs for rolling windows."""
        for ts in self.timestamps:
            yield ts, self[ts - window_size:ts]

    def ohlc(self, period):
        """
        Compute OHLC candles from trade data.

        Trades are bucketed into fixed time windows of the given period.
        Uses the timestamp_unit set at construction time.

        Args:
            period: One of '1s', '5s', '1m', '15m', '1h', '24h'.

        Returns:
            pd.DataFrame indexed by candle-open timestamp (ns) with columns:
            open, high, low, close, volume, count.
        """
        if period not in _OHLC_PERIODS:
            raise ValueError(
                f"Unknown period '{period}'. Accepted: {list(_OHLC_PERIODS)}"
            )
        if not self._trades:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "count"]
            )

        period_ts = _OHLC_PERIODS[period] * _TS_UNITS[self.timestamp_unit]
        buckets = {}  # bucket_start -> list of Trade (insertion order = time order)
        for trade in sorted(self._trades, key=lambda t: t.timestamp):
            bucket = (trade.timestamp // period_ts) * period_ts
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(trade)

        rows = {}
        for bucket_ts in sorted(buckets):
            trades = buckets[bucket_ts]
            prices = [t.price for t in trades]
            rows[bucket_ts] = {
                "open":   prices[0],
                "high":   max(prices),
                "low":    min(prices),
                "close":  prices[-1],
                "volume": sum(t.volume for t in trades),
                "count":  len(trades),
            }

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "timestamp"
        return df

    def realized_vol(self, window_size=None):
        """
        Compute realized volatility from trade prices.

        Realized volatility = sqrt(sum of squared log returns) over the
        sequence of trade prices sorted by timestamp.

        Args:
            window_size: If None, returns a scalar over all trades.
                         If given, returns a pd.Series of realized vol values
                         computed over rolling windows of the given size
                         (same units as timestamps).

        Returns:
            float (scalar) or pd.Series indexed by end timestamp.
        """
        if window_size is None:
            return _realized_vol(self._trades)
        vals = {}
        for ts, window in self._rolling_items(window_size):
            vals[ts] = _realized_vol(window._trades)
        return pd.Series(vals, name="realized_vol")

    @property
    def gueant(self):
        """Accessor for Guéant intensity function parameters (λ(δ) = A·exp(-k·δ))."""
        return GueantAccessor(self)

    def __repr__(self):
        return (
            f"<TL[{self.name}] lob_snapshots={len(self._lobts)}"
            f" trades={len(self._trades)}>"
        )
