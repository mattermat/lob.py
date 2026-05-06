"""
TimeLine (TL) - combines LOBts with trade events.
"""

import numpy as np
import pandas as pd

from lobpy._cext import ffi, lib

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
        _lobts_mode = "latest" if lob_mode == "snapshot" else "lazy"
        self._lobts = LOBts(
            name=name,
            tick_size=tick_size,
            mode=_lobts_mode,
            timestamp_unit=timestamp_unit,
        )
        self._ptr_trades = lib.trades_create()
        self._trades_cache = None  # invalidated on mutation, rebuilt lazily
        self._exchange_timestamps: np.ndarray = np.empty(0, dtype=np.int64)
        self._sequences: np.ndarray = np.empty(0, dtype=np.int64)

    def __del__(self):
        ptr = getattr(self, "_ptr_trades", None)
        if ptr is not None and ptr != ffi.NULL:
            lib.trades_destroy(ptr)

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
        if self._trades_cache is None:
            n = lib.trades_len(self._ptr_trades)
            ts_p = ffi.new("long long *")
            side_p = ffi.new("int *")
            price_p = ffi.new("double *")
            vol_p = ffi.new("double *")
            result = []
            for i in range(n):
                lib.trades_get_at(self._ptr_trades, i, ts_p, side_p, price_p, vol_p)
                result.append(Trade(ts_p[0], chr(side_p[0]), price_p[0], vol_p[0]))
            self._trades_cache = result
        return self._trades_cache

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
        lib.trades_append(self._ptr_trades, timestamp, ord(side), price, volume)
        self._trades_cache = None

    def add_trades(self, timestamp, trades):
        """
        Record multiple trade events at the same timestamp.

        Args:
            timestamp: Event timestamp
            trades: List of (side, price, volume) tuples
        """
        for side, price, volume in trades:
            lib.trades_append(self._ptr_trades, timestamp, ord(side), price, volume)
        self._trades_cache = None

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
            (t.timestamp, "trade", t.side, float("nan"), t.price, t.volume) for t in self.trades
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
        for t in self.trades:
            if (start is None or t.timestamp >= start) and (stop is None or t.timestamp <= stop):
                lib.trades_append(result._ptr_trades, t.timestamp, ord(t.side), t.price, t.volume)
        return result

    def __len__(self):
        """Return total number of events (LOB snapshots + trades)."""
        return len(self._lobts) + lib.trades_len(self._ptr_trades)

    @property
    def timestamps(self):
        """Return sorted list of all event timestamps (LOB and trades)."""
        ts = sorted(set(self._lobts.timestamps) | {t.timestamp for t in self.trades})
        return ts

    @property
    def exchange_timestamps(self):
        """Return exchange timestamps as ndarray, aligned with timestamps."""
        return self._exchange_timestamps

    @property
    def sequences(self):
        """Return sequence IDs as ndarray, aligned with timestamps."""
        return self._sequences

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
        trades = sorted(self.trades, key=lambda t: t.timestamp)
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
        return _ohlc_compute(self.trades, period, self.timestamp_unit)

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
            self.trades,
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
        return _volume_buckets_compute(self.trades, bucket_size, include_partial)

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
            self.trades,
            window_size=window_size,
            bucket_size=bucket_size,
            rolling_items=self._rolling_trades_iter,
        )

    def _duration_seconds(self):
        """Return timeline duration in seconds using the configured timestamp_unit."""
        ts = self.timestamps
        if len(ts) < 2:
            return 0.0
        return (ts[-1] - ts[0]) / _TS_UNITS[self.timestamp_unit]

    @property
    def trade_frequency(self):
        """Total trades per second over the full timeline."""
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return len(self.trades) / dur

    @property
    def ask_trade_frequency(self):
        """
        Buy-aggressor trades per second (trades that hit the ask side).

        These are trades with side='b' where the buyer is the aggressor.
        """
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(1 for t in self.trades if t.side == "b") / dur

    @property
    def bid_trade_frequency(self):
        """
        Sell-aggressor trades per second (trades that hit the bid side).

        These are trades with side='s' where the seller is the aggressor.
        """
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(1 for t in self.trades if t.side == "s") / dur

    @property
    def trade_volume_imbalance(self):
        """
        Normalized trade volume imbalance: (buy_vol − sell_vol) / (buy_vol + sell_vol).

        Ranges from −1 (all sell-aggressor volume) to +1 (all buy-aggressor volume).
        Returns 0.0 if there are no trades.
        """
        buy_vol = sum(t.volume for t in self.trades if t.side == "b")
        sell_vol = sum(t.volume for t in self.trades if t.side == "s")
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def kyle_lambda(self, interval=None, window_size=None):
        """
        Estimate Kyle's lambda: price impact per unit of signed order flow.

        Fits ΔP_mid = λ · Q_signed + α via OLS, where Q_signed is net signed
        volume (buy volume − sell volume) and ΔP_mid is the midprice change over
        the same interval. Zero-flow intervals are excluded from the regression.

        Args:
            interval:    None → one observation per consecutive LOB-update pair
                             (t₁, t₂): Q_signed = trades in (t₁, t₂],
                             ΔP_mid = mid(t₂) − mid(t₁).
                         N   → non-overlapping fixed-width buckets [t, t+N):
                             Q_signed = trades in the bucket,
                             ΔP_mid = mid(t+N) − mid(t).
                             N is in the same units as timestamps.
            window_size: None → scalar λ over all observations (float).
                         W   → rolling λ at each observation end timestamp;
                             returns pd.Series indexed by end timestamps.
                             W is in the same units as timestamps.

        Returns:
            float when window_size is None.
            pd.Series indexed by observation end timestamps when window_size is given.
            nan / empty Series if fewer than 2 non-zero-flow observations exist.
        """
        lob_ts_list = self._lobts.timestamps
        if len(lob_ts_list) < 2:
            return float("nan") if window_size is None else pd.Series(dtype=float)

        lob_ts = np.array(lob_ts_list, dtype=np.int64)
        mid_prices = self._lobts.midprice_ts().to_numpy(dtype=np.float64)

        trades_sorted = sorted(self.trades, key=lambda t: t.timestamp)
        if not trades_sorted:
            return float("nan") if window_size is None else pd.Series(dtype=float)

        trade_ts = np.array([t.timestamp for t in trades_sorted], dtype=np.int64)
        trade_sv = np.array(
            [t.volume if t.side == "b" else -t.volume for t in trades_sorted],
            dtype=np.float64,
        )

        def _mid_at(ts):
            idx = np.searchsorted(lob_ts, ts, side="right") - 1
            return float("nan") if idx < 0 else float(mid_prices[idx])

        def _sum_sv(lo, hi, lo_incl):
            # (lo, hi] when lo_incl=False;  [lo, hi) when lo_incl=True
            side_lo = "left" if lo_incl else "right"
            side_hi = "left" if lo_incl else "right"
            a = np.searchsorted(trade_ts, lo, side=side_lo)
            b = np.searchsorted(trade_ts, hi, side=side_hi)
            return float(np.sum(trade_sv[a:b])) if a < b else 0.0

        def _ols_slope(qs, dps):
            if len(qs) < 2:
                return float("nan")
            x_mat = np.column_stack([qs, np.ones(len(qs))])
            try:
                coefs, _, _, _ = np.linalg.lstsq(x_mat, dps, rcond=None)
                return float(coefs[0])
            except Exception:
                return float("nan")

        obs_q: list = []
        obs_dp: list = []
        obs_ts: list = []

        if interval is None:
            for i in range(len(lob_ts) - 1):
                t1, t2 = int(lob_ts[i]), int(lob_ts[i + 1])
                m1, m2 = float(mid_prices[i]), float(mid_prices[i + 1])
                if np.isnan(m1) or np.isnan(m2):
                    continue
                q = _sum_sv(t1, t2, lo_incl=False)
                if q == 0.0:
                    continue
                obs_q.append(q)
                obs_dp.append(m2 - m1)
                obs_ts.append(t2)
        else:
            iv = int(interval)
            t = int(lob_ts[0])
            t_end = int(lob_ts[-1])
            while t < t_end:
                t_next = t + iv
                m1, m2 = _mid_at(t), _mid_at(t_next)
                if not (np.isnan(m1) or np.isnan(m2)):
                    q = _sum_sv(t, t_next, lo_incl=True)
                    if q != 0.0:
                        obs_q.append(q)
                        obs_dp.append(m2 - m1)
                        obs_ts.append(t_next)
                t = t_next

        obs_q_arr = np.asarray(obs_q, dtype=np.float64)
        obs_dp_arr = np.asarray(obs_dp, dtype=np.float64)
        obs_ts_arr = np.asarray(obs_ts, dtype=np.int64)

        if window_size is None:
            return _ols_slope(obs_q_arr, obs_dp_arr)

        if len(obs_q_arr) == 0:
            return pd.Series(dtype=float)

        n = len(obs_q_arr)
        lambdas = np.full(n, float("nan"))
        left = 0
        for right in range(n):
            while obs_ts_arr[left] < obs_ts_arr[right] - window_size:
                left += 1
            lambdas[right] = _ols_slope(obs_q_arr[left : right + 1], obs_dp_arr[left : right + 1])

        return pd.Series(lambdas, index=obs_ts_arr)

    def fill_rate(self, holding_time, side="a", buckets=None):
        """
        Fill probability for a passive limit order at each tick distance δ.

        Uses the empirical Gueant trade arrival rate λ̂(δ) = N(δ) / T(δ) and
        applies the Poisson fill-probability formula:

            P(fill within holding_time | δ) = 1 − exp(−λ̂(δ) · holding_time)

        λ̂(δ) is the rate of trades arriving at distance δ from the best quote,
        estimated directly from the LOB and trade history (no model fitting).
        holding_time is in the same units as all timestamps in this TL.

        Args:
            holding_time: order resting duration before cancellation (timestamp units).
            side:         'a' (ask side — filled by buy-aggressor trades) or
                          'b' (bid side — filled by sell-aggressor trades).
            buckets:      optional list of δ thresholds to aggregate integer-δ bins;
                          same semantics as tl.gueant.buckets().

        Returns:
            pd.DataFrame with columns:
                delta     – tick distance from best quote (int or float threshold)
                N         – number of trades observed at this δ
                T         – total book-time (timestamp units) the book exposed liquidity at δ
                lambda    – empirical arrival rate N/T; nan if no exposure or no trades
                fill_rate – P(fill within holding_time); nan where lambda is nan
        """
        df = self.gueant.buckets(side, buckets).copy()
        ht = float(holding_time)
        df["fill_rate"] = df["lambda"].apply(
            lambda lam: float("nan") if np.isnan(lam) else 1.0 - np.exp(-lam * ht)
        )
        return df

    def hawkes(self, side=None, window_size=None):
        """
        Fit a univariate exponential Hawkes process to trade timestamps.

        λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ))

        Parameters are always in SI units — μ and α in events/second, β in 1/second —
        regardless of the TL's timestamp_unit.

        Args:
            side:        None = all trades, 'b' = buy-aggressors only,
                         's' = sell-aggressors only.
            window_size: None  → scalar fit over all trades; returns dict.
                         N     → rolling fit at each trade timestamp; returns
                                 pd.DataFrame. N is in the same timestamp units
                                 as all other TL methods.

        Returns:
            dict with keys 'mu', 'alpha', 'beta', 'branching_ratio' (all floats)
            when window_size is None.

            pd.DataFrame with those four columns, indexed by trade timestamps
            (in original timestamp units), when window_size is given.

            Returns nan / empty DataFrame if fewer than 3 trades are available.

        Notes:
            branching_ratio = α / β.  Values ≥ 1 indicate a non-stationary process
            (the optimizer enforces < 1; fits near 1 suggest the model is a poor fit
            or the data is too short).
            half_life = ln(2) / β  (in seconds).
        """
        from .hawkes import _fit

        nan = float("nan")
        trades_list = sorted(
            (t for t in self.trades if side is None or t.side == side),
            key=lambda t: t.timestamp,
        )

        scale = _TS_UNITS[self.timestamp_unit]  # ticks per second

        def _result(mu, alpha, beta):
            br = alpha / beta if (np.isfinite(alpha) and np.isfinite(beta) and beta > 0.0) else nan
            return {"mu": mu, "alpha": alpha, "beta": beta, "branching_ratio": br}

        if window_size is None:
            if len(trades_list) < 3:
                return _result(nan, nan, nan)
            ts_s = np.array([t.timestamp for t in trades_list], dtype=np.float64) / scale
            ts_s -= ts_s[0]
            mu, alpha, beta = _fit(ts_s)
            return _result(mu, alpha, beta)

        # Rolling fit at each trade timestamp
        if not trades_list:
            return pd.DataFrame(columns=["mu", "alpha", "beta", "branching_ratio"])

        ts_s = np.array([t.timestamp for t in trades_list], dtype=np.float64) / scale
        ts_orig = np.array([t.timestamp for t in trades_list], dtype=np.int64)
        ws_s = window_size / scale

        n = len(ts_s)
        mus = np.full(n, nan)
        alphas = np.full(n, nan)
        betas = np.full(n, nan)
        brs = np.full(n, nan)

        left = 0
        for right in range(n):
            while ts_s[left] < ts_s[right] - ws_s:
                left += 1
            window = ts_s[left : right + 1] - ts_s[left]
            mu, alpha, beta = _fit(window)
            mus[right] = mu
            alphas[right] = alpha
            betas[right] = beta
            if np.isfinite(alpha) and np.isfinite(beta) and beta > 0.0:
                brs[right] = alpha / beta

        return pd.DataFrame(
            {"mu": mus, "alpha": alphas, "beta": betas, "branching_ratio": brs},
            index=ts_orig,
        )

    @property
    def gueant(self):
        """Accessor for Guéant intensity function parameters (λ(δ) = A·exp(-k·δ))."""
        return GueantAccessor(self)

    def from_parquet(self, path, mode="lazy"):
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
            mode: 'lazy' (default) stores only checkpoints + a delta log and reconstructs
                  LOB snapshots on demand — dramatically lower memory usage for large files.
                  'eager' loads everything into memory immediately.
        """
        if mode == "lazy":
            self._from_parquet_lazy(path)
        elif mode == "eager":
            self._from_parquet_eager(path)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Accepted: 'lazy', 'eager'")

    def _from_parquet_eager(self, path):
        df = pd.read_parquet(path).sort_values("timestamp")

        has_exch_ts = "exchange_timestamp" in df.columns
        has_seq = "sequence" in df.columns

        _lob_side = {"bid": "b", "ask": "a"}
        _trade_side = {"buy": "b", "sell": "s"}

        tl_exch_ts = []
        tl_seqs = []
        lob_exch_ts = []
        lob_seqs = []

        for ts, group in df.groupby("timestamp", sort=True):
            exch_val = int(group.iloc[0]["exchange_timestamp"]) if has_exch_ts else None
            seq_val = int(group.iloc[0]["sequence"]) if has_seq else None

            has_lob = False

            levels = group[group["event_type"] == "book_level"]
            if not levels.empty:
                has_lob = True
                bids = [(r.price, r.quantity) for r in levels[levels["side"] == "bid"].itertuples()]
                asks = [(r.price, r.quantity) for r in levels[levels["side"] == "ask"].itertuples()]
                self.add_lob_snapshot(ts, bids, asks)

            updates = group[group["event_type"] == "book_update"]
            if not updates.empty:
                has_lob = True
                upd = [(_lob_side[r.side], r.price, r.quantity) for r in updates.itertuples()]
                self.add_lob_update(ts, upd)

            for r in group[group["event_type"] == "trade"].itertuples():
                self.add_trade(ts, _trade_side[r.side], r.price, r.quantity)

            if has_exch_ts or has_seq:
                tl_exch_ts.append(exch_val if has_exch_ts else 0)
                tl_seqs.append(seq_val if has_seq else 0)
                if has_lob:
                    lob_exch_ts.append(exch_val if has_exch_ts else 0)
                    lob_seqs.append(seq_val if has_seq else 0)

        if has_exch_ts or has_seq:
            self._exchange_timestamps = np.array(tl_exch_ts, dtype=np.int64)
            self._sequences = np.array(tl_seqs, dtype=np.int64)
            self._lobts._exchange_timestamps = np.array(lob_exch_ts, dtype=np.int64)
            self._lobts._sequences = np.array(lob_seqs, dtype=np.int64)

    def _from_parquet_lazy(self, path):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        from .lob import _make_ask_array, _make_bid_array
        from .lobts import _LAZY_DELTA_DTYPE, LOBts

        _lobts_mode = "latest" if self.lob_mode == "snapshot" else "lazy"
        lazy_lobts = LOBts(
            name=self.name,
            tick_size=self.tick_size,
            mode=_lobts_mode,
            timestamp_unit=self.timestamp_unit,
        )

        schema = pq.read_schema(path)
        base_cols = ["timestamp", "event_type", "side", "price", "quantity"]
        has_exch_ts = "exchange_timestamp" in schema.names
        has_seq = "sequence" in schema.names
        read_cols = list(base_cols)
        if has_exch_ts:
            read_cols.append("exchange_timestamp")
        if has_seq:
            read_cols.append("sequence")

        table = pq.read_table(path, columns=read_cols)

        # Sort by timestamp once upfront
        table = table.sort_by("timestamp")

        # Build per-unique-timestamp exchange_timestamp / sequence arrays
        all_ts = table.column("timestamp").to_numpy().astype(np.int64)
        if has_exch_ts or has_seq:
            exch_ts_col = (
                table.column("exchange_timestamp").to_numpy().astype(np.int64)
                if has_exch_ts
                else np.zeros(len(table), dtype=np.int64)
            )
            seq_col = (
                table.column("sequence").to_numpy().astype(np.int64)
                if has_seq
                else np.zeros(len(table), dtype=np.int64)
            )
            unique_ts_vals, first_idx = np.unique(all_ts, return_index=True)
            tl_exch_ts = exch_ts_col[first_idx]
            tl_seqs = seq_col[first_idx]
        else:
            tl_exch_ts = np.empty(0, dtype=np.int64)
            tl_seqs = np.empty(0, dtype=np.int64)

        # Vectorised split by event type — no Python-level row iteration
        et_col = table.column("event_type")
        bl_table = table.filter(pc.equal(et_col, "book_level"))
        bu_table = table.filter(pc.equal(et_col, "book_update"))
        tr_table = table.filter(pc.equal(et_col, "trade"))

        # Build per-LOB-timestamp exchange_timestamp / sequence for LOBts
        if has_exch_ts or has_seq:
            lob_mask = np.isin(
                all_ts,
                np.concatenate(
                    [
                        bl_table.column("timestamp").to_numpy(),
                        bu_table.column("timestamp").to_numpy(),
                    ]
                ),
            )
            lob_unique_ts, lob_first_idx = np.unique(all_ts[lob_mask], return_index=True)
            lob_exch_ts = (
                exch_ts_col[lob_mask][lob_first_idx] if has_exch_ts else np.empty(0, dtype=np.int64)
            )
            lob_seqs = seq_col[lob_mask][lob_first_idx] if has_seq else np.empty(0, dtype=np.int64)
        else:
            lob_exch_ts = np.empty(0, dtype=np.int64)
            lob_seqs = np.empty(0, dtype=np.int64)

        # --- book_update → delta log (fully vectorised, no per-row boxing) ---
        if len(bu_table) > 0:
            side_is_bid = pc.equal(bu_table.column("side"), "bid").to_numpy()
            delta_log = np.empty(len(bu_table), dtype=_LAZY_DELTA_DTYPE)
            delta_log["ts"] = bu_table.column("timestamp").to_numpy()
            delta_log["side"] = np.where(side_is_bid, np.uint8(0), np.uint8(1))
            delta_log["price"] = bu_table.column("price").to_numpy()
            delta_log["qty"] = bu_table.column("quantity").to_numpy()
        else:
            delta_log = np.empty(0, dtype=_LAZY_DELTA_DTYPE)

        # --- trades: bulk-load into C array without boxing Python objects ---
        lib.trades_clear(self._ptr_trades)
        if len(tr_table) > 0:
            tr_ts = tr_table.column("timestamp").to_numpy().astype(np.int64)
            tr_buy = pc.equal(tr_table.column("side"), "buy").to_numpy()
            tr_price = tr_table.column("price").to_numpy().astype(np.float64)
            tr_qty = tr_table.column("quantity").to_numpy().astype(np.float64)
            tr_sides = np.where(tr_buy, np.uint8(ord("b")), np.uint8(ord("s"))).astype(np.uint8)
            lib.trades_append_bulk(
                self._ptr_trades,
                ffi.cast("long long *", tr_ts.ctypes.data),
                ffi.cast("uint8_t *", tr_sides.ctypes.data),
                ffi.cast("double *", tr_price.ctypes.data),
                ffi.cast("double *", tr_qty.ctypes.data),
                len(tr_ts),
            )
        self._trades_cache = None

        # --- book_level → checkpoints ---
        # Group by timestamp using numpy (table is already timestamp-sorted so
        # np.unique preserves order and gives contiguous group boundaries).
        if len(bl_table) > 0:
            bl_ts = bl_table.column("timestamp").to_numpy()
            bl_bid = pc.equal(bl_table.column("side"), "bid").to_numpy()
            bl_price = bl_table.column("price").to_numpy()
            bl_qty = bl_table.column("quantity").to_numpy()

            unique_ts, first_idx = np.unique(bl_ts, return_index=True)
            end_idx = np.append(first_idx[1:], len(bl_ts))

            ckpts = {}
            for ts_val, lo, hi in zip(unique_ts, first_idx, end_idx):
                bid_mask = bl_bid[lo:hi]
                bids = list(
                    zip(bl_price[lo:hi][bid_mask].tolist(), bl_qty[lo:hi][bid_mask].tolist())
                )
                asks = list(
                    zip(bl_price[lo:hi][~bid_mask].tolist(), bl_qty[lo:hi][~bid_mask].tolist())
                )
                ckpts[int(ts_val)] = (_make_bid_array(bids), _make_ask_array(asks))

            # Assign in bulk — avoids O(N²) np.sort(np.append(...)) from set_snapshot
            lazy_lobts._ckpts = ckpts
            lazy_lobts._ckpt_ts = unique_ts.astype(np.int64)  # np.unique output is sorted

        lazy_lobts._delta_log = delta_log
        if len(delta_log) > 0:
            lazy_lobts.build_checkpoints()

        lazy_lobts._exchange_timestamps = lob_exch_ts
        lazy_lobts._sequences = lob_seqs

        self._lobts = lazy_lobts
        self._exchange_timestamps = tl_exch_ts
        self._sequences = tl_seqs

    def __repr__(self):
        return (
            f"<TL[{self.name}] lob_snapshots={len(self._lobts)}"
            f" trades={lib.trades_len(self._ptr_trades)}>"
        )
