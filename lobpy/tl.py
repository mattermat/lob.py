"""
TimeLine (TL) - combines LOBts with trade events.
"""

import numpy as np
import pandas as pd

from .gueant import GueantAccessor
from .lobts import LOBts
from .ohlc import TS_UNITS as _TS_UNITS
from .ohlc import ohlc as _ohlc_compute
from .realized_volatility import realized_vol as _realized_vol_compute
from .vpin import volume_buckets as _volume_buckets_compute
from .vpin import vpin as _vpin_compute


class Trade:
    """A single trade event."""

    __slots__ = ("timestamp", "side", "price", "volume")

    def __init__(self, timestamp, side, price, volume):
        self.timestamp = timestamp
        self.side = side  # 'b' (buy aggressor) or 's' (sell aggressor)
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

    def __init__(
        self, name=None, tick_size=1, lob_mode="delta", update_type="realtime", timestamp_unit="ns"
    ):
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
        _lobts_mode = "latest" if lob_mode == "snapshot" else "delta"
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
            for level, (price, size) in enumerate(lob._bids):
                lob_rows.append((ts, "lob", "b", level, price, size))
            for level, (price, size) in enumerate(lob._asks):
                lob_rows.append((ts, "lob", "a", level, price, size))

        trade_rows = [
            (t.timestamp, "trade", t.side, float("nan"), t.price, t.volume) for t in self._trades
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
            t
            for t in self._trades
            if (start is None or t.timestamp >= start) and (stop is None or t.timestamp <= stop)
        ]
        return result

    def __len__(self):
        """Return total number of events (LOB snapshots + trades)."""
        return len(self._lobts) + len(self._trades)

    @property
    def timestamps(self):
        """Return sorted list of all event timestamps (LOB and trades)."""
        ts = sorted(set(self._lobts.timestamps) | {t.timestamp for t in self._trades})
        return ts

    def rolling(self, window_size):
        """
        Yield TL slices over a rolling window.

        For each event timestamp ts, yields tl[ts-window_size : ts].

        Args:
            window_size: window size in time units (same units as timestamps)
        """
        for ts in self.timestamps:
            yield self[ts - window_size : ts]

    def _rolling_items(self, window_size):
        """Yield (end_timestamp, TL slice) pairs for rolling windows."""
        for ts in self.timestamps:
            yield ts, self[ts - window_size : ts]

    def _rolling_trades_iter(self, window_size):
        """
        Yield (end_timestamp, trades_in_window) pairs using a two-pointer
        sliding window over the sorted trade list.

        O(N_trades) amortised — does not touch LOB data at all.
        """
        trades = sorted(self._trades, key=lambda t: t.timestamp)
        left = 0
        for right in range(len(trades)):
            ts = trades[right].timestamp
            while trades[left].timestamp < ts - window_size:
                left += 1
            yield ts, trades[left : right + 1]

    def ohlc(self, period):
        """
        Compute OHLC candles from trade data.

        Trades are bucketed into fixed time windows of the given period.
        Uses the timestamp_unit set at construction time.

        Args:
            period: One of '1s', '5s', '1m', '15m', '1h', '24h'.

        Returns:
            pd.DataFrame indexed by candle-open timestamp with columns:
            open, high, low, close, volume, count.
        """
        return _ohlc_compute(self._trades, period, self.timestamp_unit)

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

        return _realized_vol_compute(
            self._trades,
            window_size=window_size,
            rolling_items=self._rolling_trades_iter,
        )

    def volume_buckets(self, bucket_size=None, include_partial=False):
        """
        Partition trades into fixed-volume buckets.

        Trades are processed in timestamp order. Each bucket accumulates exactly
        bucket_size total volume; a trade is split across bucket boundaries when
        it would overflow the current bucket.

        Args:
            bucket_size:     Volume per bucket. If None, computed as
                             total_volume / 50.
            include_partial: If True, the last partial bucket (volume < bucket_size)
                             is included. Default False.

        Returns:
            pd.DataFrame indexed by bucket number (0-based) with columns:
                buy_volume  – volume from buy-aggressor trades
                sell_volume – volume from sell-aggressor trades
        """
        return _volume_buckets_compute(self._trades, bucket_size, include_partial)

    def vpin(self, window_size=None, bucket_size=None):
        """
        Compute VPIN (Volume-Synchronized Probability of Informed Trading).

        VPIN = Σ|V_buy[i] - V_sell[i]| / (n × bucket_size)

        where the sum runs over all complete volume buckets.

        Args:
            window_size: If None, returns a scalar over all trades.
                         If given, returns a pd.Series of VPIN values computed
                         over rolling time windows of the given size (same units
                         as timestamps).
            bucket_size: Volume per bucket. If None, computed once as
                         total_volume / 50 and reused across all windows.

        Returns:
            float (scalar) or pd.Series indexed by end timestamp.
        """

        return _vpin_compute(
            self._trades,
            window_size=window_size,
            bucket_size=bucket_size,
            rolling_items=self._rolling_trades_iter,
        )

    @property
    def gueant(self):
        """Accessor for Guéant intensity function parameters (λ(δ) = A·exp(-k·δ))."""
        return GueantAccessor(self)

    def from_parquet(self, path, mode="eager"):
        """
        Load LOB and trade events from a parquet file into this TL instance.

        Expected columns: timestamp, event_type, side, price, quantity.

        event_type values:
            'book_level'  – rows belonging to a full LOB snapshot; all rows
                            sharing the same timestamp are treated as one snapshot.
            'book_update' – incremental LOB updates; rows sharing a timestamp
                            are applied as a single update batch.
            'trade'       – individual trade events.

        side values:
            LOB rows:   'bid' / 'ask'
            Trade rows: 'buy' / 'sell'

        Events are processed in timestamp order. Within the same timestamp,
        book_level rows are applied before book_update rows, which are applied
        before trade rows.

        Args:
            path: Path to the parquet file.
            mode: 'eager' (default) loads everything into memory immediately.
                  'lazy' stores only checkpoints + a delta log and reconstructs
                  LOB snapshots on demand — dramatically lower memory usage for
                  large files.
        """
        if mode == "eager":
            self._from_parquet_eager(path)
        elif mode == "lazy":
            self._from_parquet_lazy(path)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Accepted: 'eager', 'lazy'")

    def _from_parquet_eager(self, path):
        df = pd.read_parquet(path).sort_values("timestamp")

        _lob_side = {"bid": "b", "ask": "a"}
        _trade_side = {"buy": "b", "sell": "s"}

        for ts, group in df.groupby("timestamp", sort=True):
            levels = group[group["event_type"] == "book_level"]
            if not levels.empty:
                bids = [(r.price, r.quantity) for r in levels[levels["side"] == "bid"].itertuples()]
                asks = [(r.price, r.quantity) for r in levels[levels["side"] == "ask"].itertuples()]
                self.add_lob_snapshot(ts, bids, asks)

            updates = group[group["event_type"] == "book_update"]
            if not updates.empty:
                upd = [(_lob_side[r.side], r.price, r.quantity) for r in updates.itertuples()]
                self.add_lob_update(ts, upd)

            for r in group[group["event_type"] == "trade"].itertuples():
                self.add_trade(ts, _trade_side[r.side], r.price, r.quantity)

    def _from_parquet_lazy(self, path):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        from .lob import _make_ask_array, _make_bid_array
        from .lobts import _LAZY_DELTA_DTYPE, LOBts

        _lobts_mode = "latest" if self.lob_mode == "snapshot" else "lazy"
        lazy_lobts = LOBts(name=self.name, tick_size=self.tick_size, mode=_lobts_mode)

        table = pq.read_table(
            path,
            columns=["timestamp", "event_type", "side", "price", "quantity"],
        )

        # Sort by timestamp once upfront
        table = table.sort_by("timestamp")

        # Vectorised split by event type — no Python-level row iteration
        et_col = table.column("event_type")
        bl_table = table.filter(pc.equal(et_col, "book_level"))
        bu_table = table.filter(pc.equal(et_col, "book_update"))
        tr_table = table.filter(pc.equal(et_col, "trade"))

        # --- book_update → delta log (fully vectorised, no per-row boxing) ---
        if len(bu_table) > 0:
            side_is_bid = pc.equal(bu_table.column("side"), "bid").to_numpy()
            delta_log = np.empty(len(bu_table), dtype=_LAZY_DELTA_DTYPE)
            delta_log["ts"]    = bu_table.column("timestamp").to_numpy()
            delta_log["side"]  = np.where(side_is_bid, np.uint8(0), np.uint8(1))
            delta_log["price"] = bu_table.column("price").to_numpy()
            delta_log["qty"]   = bu_table.column("quantity").to_numpy()
        else:
            delta_log = np.empty(0, dtype=_LAZY_DELTA_DTYPE)

        # --- trades (numeric columns via numpy, side via vectorised comparison) ---
        if len(tr_table) > 0:
            tr_ts    = tr_table.column("timestamp").to_numpy()
            tr_buy   = pc.equal(tr_table.column("side"), "buy").to_numpy()
            tr_price = tr_table.column("price").to_numpy()
            tr_qty   = tr_table.column("quantity").to_numpy()
            trade_rows = [
                Trade(int(ts), "b" if buy else "s", float(p), float(q))
                for ts, buy, p, q in zip(tr_ts, tr_buy, tr_price, tr_qty)
            ]
        else:
            trade_rows = []

        # --- book_level → checkpoints ---
        # Group by timestamp using numpy (table is already timestamp-sorted so
        # np.unique preserves order and gives contiguous group boundaries).
        if len(bl_table) > 0:
            bl_ts    = bl_table.column("timestamp").to_numpy()
            bl_bid   = pc.equal(bl_table.column("side"), "bid").to_numpy()
            bl_price = bl_table.column("price").to_numpy()
            bl_qty   = bl_table.column("quantity").to_numpy()

            unique_ts, first_idx = np.unique(bl_ts, return_index=True)
            end_idx = np.append(first_idx[1:], len(bl_ts))

            ckpts = {}
            for ts_val, lo, hi in zip(unique_ts, first_idx, end_idx):
                bid_mask = bl_bid[lo:hi]
                bids = list(zip(bl_price[lo:hi][bid_mask].tolist(),
                                bl_qty[lo:hi][bid_mask].tolist()))
                asks = list(zip(bl_price[lo:hi][~bid_mask].tolist(),
                                bl_qty[lo:hi][~bid_mask].tolist()))
                ckpts[int(ts_val)] = (_make_bid_array(bids), _make_ask_array(asks))

            # Assign in bulk — avoids O(N²) np.sort(np.append(...)) from set_snapshot
            lazy_lobts._ckpts   = ckpts
            lazy_lobts._ckpt_ts = unique_ts.astype(np.int64)  # np.unique output is sorted

        lazy_lobts._delta_log = delta_log
        if len(delta_log) > 0:
            lazy_lobts.build_checkpoints()

        self._lobts = lazy_lobts
        self._trades = trade_rows

    def __repr__(self):
        return f"<TL[{self.name}] lob_snapshots={len(self._lobts)}" f" trades={len(self._trades)}>"
