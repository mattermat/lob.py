"""
Time Series LOB (Limit Order Book)
"""

from collections import OrderedDict

import numpy as np

from lobpy._cext import ffi, lib

from .lob import LOB, _get_sidebook_data, _levels_to_arrays, _make_ask_array, _make_bid_array
from .ohlc import TS_UNITS as _TS_UNITS

_LAZY_DELTA_DTYPE: np.dtype = np.dtype(
    [("ts", "i8"), ("side", "u1"), ("price", "f8"), ("qty", "f8")]
)
_CACHE_MAXSIZE = 32
_AUTO_CHECKPOINTS = 100


class LOBts:
    """Time series of LOB objects indexed by timestamp."""

    def __init__(self, name=None, tick_size=1, mode="lazy", timestamp_unit="ns") -> None:
        """
        Initialize LOBts.

        Args:
            name: Optional identifier for the time series
            tick_size: Minimum price increment
            mode: 'lazy' (default) stores only checkpoints + delta log and reconstructs on demand,
                  'eager' stores all snapshots as full LobBook objects in C memory,
                  'latest' keeps only the most recent snapshot in C memory
            timestamp_unit: Unit of all timestamps. One of 's', 'ms', 'us', 'ns'. Defaults to 'ns'.
        """
        if name is None:
            name = f"lobts{id(self)}"
        self.name = name
        self.tick_size = tick_size
        self.timestamp_unit = timestamp_unit
        if mode not in ("lazy", "eager", "latest"):
            raise ValueError(
                f"Unknown mode '{mode}'. Accepted: 'lazy' (default), 'eager', 'latest'."
            )
        self._mode = mode

        if mode == "lazy":
            self._ptr = None
            self._delta_log = np.empty(0, dtype=_LAZY_DELTA_DTYPE)
            self._ckpt_ts: np.ndarray = np.empty(0, dtype=np.int64)
            self._ckpts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            self._cache: OrderedDict[int, LOB] = OrderedDict()
            self._ts_lo = None  # inclusive lower bound for timestamps view
            self._ts_hi = None  # inclusive upper bound for timestamps view
            self._exchange_timestamps: np.ndarray = np.empty(0, dtype=np.int64)
            self._sequences: np.ndarray = np.empty(0, dtype=np.int64)
        else:
            mode_int = 0 if mode == "eager" else 1  # LOBTS_DELTA=0, LOBTS_LATEST=1
            self._ptr = lib.lobts_create(name.encode(), float(tick_size), mode_int)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr is not None:
            lib.lobts_destroy(self._ptr)

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def mode(self):
        """Return mode."""
        return self._mode

    @property
    def timestamps(self):
        """Return sorted timestamps."""
        if self._mode == "lazy":
            ts_set = set(self._ckpt_ts.tolist())
            if len(self._delta_log) > 0:
                ts_set |= set(np.unique(self._delta_log["ts"]).tolist())
            if self._ts_lo is not None:
                ts_set = {ts for ts in ts_set if ts >= self._ts_lo}
            if self._ts_hi is not None:
                ts_set = {ts for ts in ts_set if ts <= self._ts_hi}
            return sorted(ts_set)
        ts_pp = ffi.new("const long long **")
        len_p = ffi.new("int *")
        lib.lobts_get_timestamps(self._ptr, ts_pp, len_p)
        n = len_p[0]
        if n == 0:
            return []
        buf = ffi.buffer(ffi.cast("long long *", ts_pp[0]), n * 8)
        return np.frombuffer(buf, dtype=np.int64).copy().tolist()

    @property
    def exchange_timestamps(self):
        """Return exchange timestamps aligned with timestamps."""
        if self._mode == "lazy":
            return self._exchange_timestamps
        return np.empty(0, dtype=np.int64)

    @property
    def sequences(self):
        """Return sequence IDs aligned with timestamps."""
        if self._mode == "lazy":
            return self._sequences
        return np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_snapshot(self, bids, asks, timestamp=0, force=False):
        """
        Create a LOB snapshot at the given timestamp.

        Args:
            bids: List of (price, size) tuples for bid side
            asks: List of (price, size) tuples for ask side
            timestamp: Timestamp for this snapshot
            force: If True and timestamp exists, replace it
        """
        if self._mode == "lazy":
            ts = int(timestamp)
            if ts in self._ckpts and not force:
                raise ValueError(f"Timestamp {ts} already exists. Use force=True to overwrite.")
            self._ckpts[ts] = (_make_bid_array(bids), _make_ask_array(asks))
            if ts not in self._ckpt_ts:
                self._ckpt_ts = np.sort(np.append(self._ckpt_ts, np.int64(ts)))
            self._cache.pop(ts, None)
            return

        b_prices, b_qtys = _levels_to_arrays(bids)
        a_prices, a_qtys = _levels_to_arrays(asks)
        bn, an = len(b_prices), len(a_prices)
        bp = ffi.cast("double *", b_prices.ctypes.data) if bn else ffi.NULL
        bq = ffi.cast("double *", b_qtys.ctypes.data) if bn else ffi.NULL
        ap = ffi.cast("double *", a_prices.ctypes.data) if an else ffi.NULL
        aq = ffi.cast("double *", a_qtys.ctypes.data) if an else ffi.NULL
        ret = lib.lobts_set_snapshot(
            self._ptr, int(timestamp), bp, bq, bn, ap, aq, an, 1 if force else 0
        )
        if ret < 0:
            raise ValueError(f"Timestamp {timestamp} already exists. Use force=True to overwrite.")

    def set_updates(self, updates, timestamp=0):
        """
        Apply updates at timestamp.

        In eager mode, copies the previous LOB state, applies deltas, and stores a
        new full snapshot.  In lazy mode, appends raw delta rows to the delta log.

        Args:
            updates: List of (side, price, size) tuples. Side can be 'bid'/'b' or 'ask'/'a'
            timestamp: Timestamp for this snapshot

        Returns:
            The new LOB object (eager mode only; lazy mode returns None)
        """
        if self._mode == "lazy":
            if not updates:
                return None
            ts = int(timestamp)
            rows = []
            for side, price, size in updates:
                side_int = np.uint8(0) if side in ("b", "bid") else np.uint8(1)
                rows.append((np.int64(ts), side_int, np.float64(price), np.float64(size)))
            new_rows = np.array(rows, dtype=_LAZY_DELTA_DTYPE)
            self._delta_log = np.concatenate([self._delta_log, new_rows])
            self._cache.pop(ts, None)
            return None

        if not updates:
            return None
        n = len(updates)
        sides = np.empty(n, dtype=np.uint8)
        prices = np.empty(n, dtype=np.float64)
        qtys = np.empty(n, dtype=np.float64)
        for i, (side, price, size) in enumerate(updates):
            sides[i] = 0 if side in ("b", "bid") else 1
            prices[i] = float(price)
            qtys[i] = float(size)
        lib.lobts_set_updates(
            self._ptr,
            int(timestamp),
            ffi.cast("uint8_t *", sides.ctypes.data),
            ffi.cast("double *", prices.ctypes.data),
            ffi.cast("double *", qtys.ctypes.data),
            n,
        )
        return self[int(timestamp)]

    def update(self, side, price_level, size, timestamp=0):
        """
        Apply a single update, creating a new LOB snapshot.

        Args:
            side: 'bid'/'b' or 'ask'/'a'
            price_level: Price level
            size: Quantity (0 to delete)
            timestamp: Timestamp for this snapshot

        Returns:
            The new LOB object (eager mode only)
        """
        if side == "b":
            side = "bid"
        elif side == "a":
            side = "ask"
        return self.set_updates([(side, price_level, size)], timestamp)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def __getitem__(self, timestamp_or_slice):
        """
        Return LOB at specific timestamp or slice.

        Args:
            timestamp_or_slice: Timestamp to retrieve, or slice object

        Returns:
            LOB object at timestamp, new LOBts for slice, or None if not found
        """
        if isinstance(timestamp_or_slice, slice):
            return self.get_range(timestamp_or_slice.start, timestamp_or_slice.stop)

        t = timestamp_or_slice

        if self._mode == "lazy":
            t = int(t)
            if t not in self:
                return None
            return self._reconstruct(t)

        lob_ptr = lib.lobts_get(self._ptr, int(t))
        if lob_ptr == ffi.NULL:
            return None
        return LOB._from_ptr(lob_ptr)

    def _reconstruct(self, t):
        """Reconstruct LOB at timestamp t from nearest checkpoint + delta replay."""
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]

        # Find the nearest checkpoint at or before t.
        # If none exists, start from an empty book (matching eager-mode behaviour when
        # set_updates is called before any set_snapshot).
        if len(self._ckpt_ts) == 0:
            t0 = None
            bids_arr = np.empty((0, 2), dtype=np.float64)
            asks_arr = np.empty((0, 2), dtype=np.float64)
        else:
            idx = int(np.searchsorted(self._ckpt_ts, t, side="right")) - 1
            if idx < 0:
                t0 = None
                bids_arr = np.empty((0, 2), dtype=np.float64)
                asks_arr = np.empty((0, 2), dtype=np.float64)
            else:
                t0 = int(self._ckpt_ts[idx])
                bids_arr, asks_arr = self._ckpts[t0]

        # Slice delta log from t0 (or the beginning if no checkpoint) to t (both inclusive)
        if len(self._delta_log) > 0:
            lo = (
                int(np.searchsorted(self._delta_log["ts"], t0, side="left"))
                if t0 is not None
                else 0
            )
            hi = int(np.searchsorted(self._delta_log["ts"], t, side="right"))
            deltas = self._delta_log[lo:hi]
        else:
            deltas = self._delta_log

        # Replay onto a working dict copy of the checkpoint
        bid_dict = {float(p): float(q) for p, q in bids_arr}
        ask_dict = {float(p): float(q) for p, q in asks_arr}
        for ts_d, side_int, price, qty in deltas:
            price = float(price)
            qty = float(qty)
            if side_int == 0:  # bid
                if qty == 0.0:
                    bid_dict.pop(price, None)
                else:
                    bid_dict[price] = qty
            else:  # ask
                if qty == 0.0:
                    ask_dict.pop(price, None)
                else:
                    ask_dict[price] = qty

        lob = LOB(
            name=f"{self.name}_t{t}",
            tick_size=self.tick_size,
            bids=list(bid_dict.items()),
            asks=list(ask_dict.items()),
        )
        lob.timestamp = t

        self._cache[t] = lob
        self._cache.move_to_end(t)
        if len(self._cache) > _CACHE_MAXSIZE:
            self._cache.popitem(last=False)

        return lob

    def get_at_timestamp(self, timestamp):
        """
        Return LOB at specific timestamp.

        Args:
            timestamp: Timestamp to retrieve

        Returns:
            LOB object at timestamp, or None if not found
        """
        return self[timestamp]

    def get_range(self, start_ts, end_ts):
        """
        Return new LOBts containing only snapshots in time range [start_ts, end_ts].

        In lazy mode the nearest preceding checkpoint is included as the reconstruction
        anchor so that all timestamps in the requested range can be reconstructed.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            New LOBts with filtered data
        """
        if self._mode == "lazy":
            result = LOBts(
                name=f"{self.name}_range",
                tick_size=self.tick_size,
                mode="lazy",
                timestamp_unit=self.timestamp_unit,
            )

            # Anchor: nearest checkpoint at or before start_ts
            if start_ts is not None and len(self._ckpt_ts) > 0:
                anchor_idx = int(np.searchsorted(self._ckpt_ts, start_ts, side="right")) - 1
                anchor_ts = int(self._ckpt_ts[anchor_idx]) if anchor_idx >= 0 else None
            else:
                anchor_ts = None

            lo_ts = anchor_ts if anchor_ts is not None else start_ts

            # Copy checkpoints from anchor onwards (up to end_ts)
            ckpt_mask = np.ones(len(self._ckpt_ts), dtype=bool)
            if lo_ts is not None:
                ckpt_mask &= self._ckpt_ts >= lo_ts
            if end_ts is not None:
                ckpt_mask &= self._ckpt_ts <= end_ts
            result._ckpt_ts = self._ckpt_ts[ckpt_mask].copy()
            result._ckpts = {int(ts): self._ckpts[int(ts)] for ts in result._ckpt_ts}

            # Copy delta rows from anchor onwards (up to end_ts)
            if len(self._delta_log) > 0:
                lo = int(np.searchsorted(self._delta_log["ts"], lo_ts)) if lo_ts is not None else 0
                hi = (
                    int(np.searchsorted(self._delta_log["ts"], end_ts, side="right"))
                    if end_ts is not None
                    else len(self._delta_log)
                )
                result._delta_log = self._delta_log[lo:hi].copy()

            # Restrict what timestamps reports to the user-requested range
            result._ts_lo = start_ts
            result._ts_hi = end_ts

            # Slice exchange_timestamps / sequences to the user-requested range
            if len(self._exchange_timestamps) > 0:
                result._exchange_timestamps = self._exchange_timestamps.copy()
                result._sequences = self._sequences.copy()

            return result

        has_start = 1 if start_ts is not None else 0
        has_end = 1 if end_ts is not None else 0
        new_ptr = lib.lobts_get_range(
            self._ptr,
            int(start_ts) if start_ts is not None else 0,
            int(end_ts) if end_ts is not None else 0,
            has_start,
            has_end,
        )
        result = LOBts.__new__(LOBts)
        result.name = f"{self.name}_range"
        result.tick_size = self.tick_size
        result._mode = self._mode
        result._ptr = new_ptr
        result.timestamp_unit = self.timestamp_unit
        return result

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Return number of LOB timestamps stored."""
        if self._mode == "lazy":
            return len(self.timestamps)
        return lib.lobts_len(self._ptr)

    @property
    def len(self):
        """Return number of timestamps (for compatibility with example)."""
        return len(self)

    @property
    def len_ts(self):
        """Return duration: last timestamp - first timestamp."""
        ts = list(self.timestamps)
        if len(ts) < 2:
            return 0
        return ts[-1] - ts[0]

    def __contains__(self, timestamp):
        """Check if timestamp exists in the series."""
        if self._mode == "lazy":
            ts = int(timestamp)
            if ts in self._ckpts:
                return True
            if len(self._delta_log) == 0:
                return False
            idx = int(np.searchsorted(self._delta_log["ts"], ts))
            return idx < len(self._delta_log) and self._delta_log["ts"][idx] == ts
        return lib.lobts_get(self._ptr, int(timestamp)) != ffi.NULL

    def __iter__(self):
        """Iterate over LOB objects in timestamp order."""
        if self._mode == "lazy":
            for ts in self.timestamps:
                yield self[ts]
            return
        for i in range(lib.lobts_len(self._ptr)):
            yield LOB._from_ptr(lib.lobts_get_at(self._ptr, i))

    # ------------------------------------------------------------------
    # Time series analytics — mode-agnostic via self.timestamps / self[ts]
    # ------------------------------------------------------------------

    def _seq_extract_best(self, start_ts=None, end_ts=None):
        """
        Single forward pass through checkpoints + deltas via C extension.

        Returns (timestamps, best_bids, best_asks, best_bidqs, best_askqs).
        Only valid in lazy mode; O(N + D) vs the naive O(N * D/C).
        timestamps is a Python list; the four price/qty outputs are numpy float64 arrays.
        """
        all_ts = self.timestamps  # sorted
        if start_ts is not None:
            all_ts = [t for t in all_ts if t >= start_ts]
        if end_ts is not None:
            all_ts = [t for t in all_ts if t <= end_ts]
        if not all_ts:
            return [], [], [], [], []

        n_ts = len(all_ts)
        all_ts_arr = np.asarray(all_ts, dtype=np.int64)
        first_ts = all_ts[0]
        last_ts = all_ts[-1]

        # Anchor: nearest checkpoint at or before first_ts
        n_ckpts = len(self._ckpt_ts)
        anchor_idx = (
            int(np.searchsorted(self._ckpt_ts, first_ts, side="right")) - 1 if n_ckpts > 0 else -1
        )
        anchor_ts = int(self._ckpt_ts[anchor_idx]) if anchor_idx >= 0 else None

        # Slice delta log to [anchor_ts_or_first_ts, last_ts]
        if len(self._delta_log) > 0:
            lo = int(
                np.searchsorted(
                    self._delta_log["ts"],
                    anchor_ts if anchor_ts is not None else first_ts,
                    side="left",
                )
            )
            hi = int(np.searchsorted(self._delta_log["ts"], last_ts, side="right"))
            deltas = self._delta_log[lo:hi]
        else:
            deltas = self._delta_log
        n_deltas = len(deltas)

        # Flatten all checkpoints into contiguous arrays + offset tables.
        # bid_offs / ask_offs have length n_ckpts+1 (CSR-style);
        # ckpt_bids_flat / ckpt_asks_flat are interleaved [price, qty, ...].
        if n_ckpts > 0:
            bid_offs = np.empty(n_ckpts + 1, dtype=np.int32)
            ask_offs = np.empty(n_ckpts + 1, dtype=np.int32)
            bid_offs[0] = ask_offs[0] = 0
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                bid_offs[k + 1] = bid_offs[k] + len(bids_a)
                ask_offs[k + 1] = ask_offs[k] + len(asks_a)
            total_bid = int(bid_offs[n_ckpts])
            total_ask = int(ask_offs[n_ckpts])
            ckpt_bids_flat = np.empty(total_bid * 2, dtype=np.float64)
            ckpt_asks_flat = np.empty(total_ask * 2, dtype=np.float64)
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                b0 = int(bid_offs[k]) * 2
                b1 = int(bid_offs[k + 1]) * 2
                if b1 > b0:
                    ckpt_bids_flat[b0:b1] = bids_a.ravel()
                a0 = int(ask_offs[k]) * 2
                a1 = int(ask_offs[k + 1]) * 2
                if a1 > a0:
                    ckpt_asks_flat[a0:a1] = asks_a.ravel()

        # Extract contiguous field arrays from the structured delta log.
        # Structured-array field views are strided (stride = record size), so
        # np.ascontiguousarray is required before passing raw pointers to C.
        if n_deltas > 0:
            dl_ts = np.ascontiguousarray(deltas["ts"])
            dl_side = np.ascontiguousarray(deltas["side"])
            dl_price = np.ascontiguousarray(deltas["price"])
            dl_qty = np.ascontiguousarray(deltas["qty"])

        # Pre-allocate output arrays; C fills them in one pass.
        out_bids = np.empty(n_ts, dtype=np.float64)
        out_asks = np.empty(n_ts, dtype=np.float64)
        out_bidqs = np.empty(n_ts, dtype=np.float64)
        out_askqs = np.empty(n_ts, dtype=np.float64)

        def _p(arr, ct):
            return ffi.cast(ct, arr.ctypes.data)

        lib.lobts_seq_extract_best(
            _p(all_ts_arr, "const long long *"),
            n_ts,
            _p(self._ckpt_ts, "const long long *") if n_ckpts > 0 else ffi.NULL,
            n_ckpts,
            _p(bid_offs, "const int *") if n_ckpts > 0 else ffi.NULL,
            _p(ckpt_bids_flat, "const double *") if n_ckpts > 0 and total_bid > 0 else ffi.NULL,
            _p(ask_offs, "const int *") if n_ckpts > 0 else ffi.NULL,
            _p(ckpt_asks_flat, "const double *") if n_ckpts > 0 and total_ask > 0 else ffi.NULL,
            _p(dl_ts, "const long long *") if n_deltas > 0 else ffi.NULL,
            _p(dl_side, "const uint8_t *") if n_deltas > 0 else ffi.NULL,
            _p(dl_price, "const double *") if n_deltas > 0 else ffi.NULL,
            _p(dl_qty, "const double *") if n_deltas > 0 else ffi.NULL,
            n_deltas,
            anchor_idx,
            _p(out_bids, "double *"),
            _p(out_asks, "double *"),
            _p(out_bidqs, "double *"),
            _p(out_askqs, "double *"),
        )

        return all_ts, out_bids, out_asks, out_bidqs, out_askqs

    def _iter_seq_states(self, start_ts=None, end_ts=None):
        """
        Yield (ts, bid_dict, ask_dict) for every LOB timestamp via a single forward pass.

        bid_dict / ask_dict are plain {price: qty} dicts (copies, safe to read after yield).
        In lazy mode this uses the C extension; in other modes it iterates stored LOB objects.
        """
        if self._mode != "lazy":
            for i in range(lib.lobts_len(self._ptr)):
                ts = lib.lobts_ts_at(self._ptr, i)
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    break
                lob_ptr = lib.lobts_get_at(self._ptr, i)
                bids_arr = _get_sidebook_data(lob_ptr, lib.lob_get_bids)
                asks_arr = _get_sidebook_data(lob_ptr, lib.lob_get_asks)
                yield (
                    int(ts),
                    {float(p): float(q) for p, q in bids_arr},
                    {float(p): float(q) for p, q in asks_arr},
                )
            return

        # Lazy mode — delegate the forward pass to C.
        all_ts = self.timestamps
        if start_ts is not None:
            all_ts = [t for t in all_ts if t >= start_ts]
        if end_ts is not None:
            all_ts = [t for t in all_ts if t <= end_ts]
        if not all_ts:
            return

        n_ts = len(all_ts)
        all_ts_arr = np.asarray(all_ts, dtype=np.int64)
        first_ts = all_ts[0]
        last_ts = all_ts[-1]

        n_ckpts = len(self._ckpt_ts)
        anchor_idx = (
            int(np.searchsorted(self._ckpt_ts, first_ts, side="right")) - 1 if n_ckpts > 0 else -1
        )
        anchor_ts = int(self._ckpt_ts[anchor_idx]) if anchor_idx >= 0 else None

        if len(self._delta_log) > 0:
            lo = int(
                np.searchsorted(
                    self._delta_log["ts"],
                    anchor_ts if anchor_ts is not None else first_ts,
                    side="left",
                )
            )
            hi = int(np.searchsorted(self._delta_log["ts"], last_ts, side="right"))
            deltas = self._delta_log[lo:hi]
        else:
            deltas = self._delta_log
        n_deltas = len(deltas)

        # Flatten checkpoint arrays (same logic as _seq_extract_best)
        if n_ckpts > 0:
            bid_offs = np.empty(n_ckpts + 1, dtype=np.int32)
            ask_offs = np.empty(n_ckpts + 1, dtype=np.int32)
            bid_offs[0] = ask_offs[0] = 0
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                bid_offs[k + 1] = bid_offs[k] + len(bids_a)
                ask_offs[k + 1] = ask_offs[k] + len(asks_a)
            total_bid_ckpt = int(bid_offs[n_ckpts])
            total_ask_ckpt = int(ask_offs[n_ckpts])
            ckpt_bids_flat = np.empty(total_bid_ckpt * 2, dtype=np.float64)
            ckpt_asks_flat = np.empty(total_ask_ckpt * 2, dtype=np.float64)
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                b0 = int(bid_offs[k]) * 2
                b1 = int(bid_offs[k + 1]) * 2
                if b1 > b0:
                    ckpt_bids_flat[b0:b1] = bids_a.ravel()
                a0 = int(ask_offs[k]) * 2
                a1 = int(ask_offs[k + 1]) * 2
                if a1 > a0:
                    ckpt_asks_flat[a0:a1] = asks_a.ravel()

        if n_deltas > 0:
            dl_ts = np.ascontiguousarray(deltas["ts"])
            dl_side = np.ascontiguousarray(deltas["side"])
            dl_price = np.ascontiguousarray(deltas["price"])
            dl_qty = np.ascontiguousarray(deltas["qty"])

        def _p(arr, ct):
            return ffi.cast(ct, arr.ctypes.data)

        # Shared CFFI args (used by both passes)
        c_all_ts = _p(all_ts_arr, "const long long *")
        c_ckpt_ts = _p(self._ckpt_ts, "const long long *") if n_ckpts > 0 else ffi.NULL
        c_bid_offs = _p(bid_offs, "const int *") if n_ckpts > 0 else ffi.NULL
        c_ckbids = (
            _p(ckpt_bids_flat, "const double *") if n_ckpts > 0 and total_bid_ckpt > 0 else ffi.NULL
        )
        c_ask_offs = _p(ask_offs, "const int *") if n_ckpts > 0 else ffi.NULL
        c_ckasks = (
            _p(ckpt_asks_flat, "const double *") if n_ckpts > 0 and total_ask_ckpt > 0 else ffi.NULL
        )
        c_dl_ts = _p(dl_ts, "const long long *") if n_deltas > 0 else ffi.NULL
        c_dl_side = _p(dl_side, "const uint8_t *") if n_deltas > 0 else ffi.NULL
        c_dl_price = _p(dl_price, "const double *") if n_deltas > 0 else ffi.NULL
        c_dl_qty = _p(dl_qty, "const double *") if n_deltas > 0 else ffi.NULL

        out_ts_arr = np.empty(n_ts, dtype=np.int64)
        out_bid_offs = np.empty(n_ts + 1, dtype=np.int32)
        out_ask_offs = np.empty(n_ts + 1, dtype=np.int32)
        total_bid_p = ffi.new("int *")
        total_ask_p = ffi.new("int *")

        def _call(bid_data_ptr, ask_data_ptr):
            lib.lobts_iter_seq_states(
                c_all_ts,
                n_ts,
                c_ckpt_ts,
                n_ckpts,
                c_bid_offs,
                c_ckbids,
                c_ask_offs,
                c_ckasks,
                c_dl_ts,
                c_dl_side,
                c_dl_price,
                c_dl_qty,
                n_deltas,
                anchor_idx,
                _p(out_ts_arr, "long long *"),
                _p(out_bid_offs, "int *"),
                _p(out_ask_offs, "int *"),
                bid_data_ptr,
                ask_data_ptr,
                total_bid_p,
                total_ask_p,
            )

        # Pass 1: count only — fills offset tables and totals, data pointers are NULL
        _call(ffi.NULL, ffi.NULL)
        total_bid = total_bid_p[0]
        total_ask = total_ask_p[0]

        # Pass 2: fill — allocate output arrays sized from pass 1 and fill them
        bid_data = np.empty(total_bid * 2, dtype=np.float64)
        ask_data = np.empty(total_ask * 2, dtype=np.float64)
        _call(
            _p(bid_data, "double *") if total_bid > 0 else ffi.NULL,
            _p(ask_data, "double *") if total_ask > 0 else ffi.NULL,
        )

        # Reconstruct (ts, bid_dict, ask_dict) from flat CSR arrays.
        # Use arr[::2].tolist() + arr[1::2].tolist() instead of element-by-element
        # float(arr[j]) indexing: .tolist() converts a whole numpy column to Python
        # floats in one C-speed pass, making dict construction ~10× faster for deep books.
        for i in range(n_ts):
            t = int(out_ts_arr[i])
            b0 = int(out_bid_offs[i]) * 2
            b1 = int(out_bid_offs[i + 1]) * 2
            a0 = int(out_ask_offs[i]) * 2
            a1 = int(out_ask_offs[i + 1]) * 2
            bf = bid_data[b0:b1]
            af = ask_data[a0:a1]
            yield (
                t,
                dict(zip(bf[::2].tolist(), bf[1::2].tolist())),
                dict(zip(af[::2].tolist(), af[1::2].tolist())),
            )

    def _seq_states_csr(self):
        """
        (Lazy mode only) Run a single C forward pass and return raw CSR LOB state arrays.

        Returns
        -------
        lob_ts    : np.ndarray[int64,   shape=(n,)]    LOB-change timestamps
        bid_offs  : np.ndarray[int32,   shape=(n+1,)]  CSR row offsets for bids
        ask_offs  : np.ndarray[int32,   shape=(n+1,)]  CSR row offsets for asks
        bid_data  : np.ndarray[float64, shape=(2*B,)]  flat [price,qty,...] bids (descending)
        ask_data  : np.ndarray[float64, shape=(2*A,)]  flat [price,qty,...] asks (ascending)

        Each LOB state i uses bid_data[bid_offs[i]*2 : bid_offs[i+1]*2] (similarly for asks).
        bid_data[bid_offs[i]*2] is the best bid (highest); ask_data[ask_offs[i]*2] is the best ask.
        """
        all_ts = self.timestamps
        if not all_ts:
            empty_offs = np.zeros(1, dtype=np.int32)
            return (
                np.empty(0, np.int64),
                empty_offs,
                empty_offs.copy(),
                np.empty(0, np.float64),
                np.empty(0, np.float64),
            )

        n_ts = len(all_ts)
        all_ts_arr = np.asarray(all_ts, dtype=np.int64)
        first_ts = all_ts[0]
        last_ts = all_ts[-1]

        n_ckpts = len(self._ckpt_ts)
        anchor_idx = (
            int(np.searchsorted(self._ckpt_ts, first_ts, side="right")) - 1 if n_ckpts > 0 else -1
        )
        anchor_ts = int(self._ckpt_ts[anchor_idx]) if anchor_idx >= 0 else None

        if len(self._delta_log) > 0:
            lo = int(
                np.searchsorted(
                    self._delta_log["ts"],
                    anchor_ts if anchor_ts is not None else first_ts,
                    side="left",
                )
            )
            hi = int(np.searchsorted(self._delta_log["ts"], last_ts, side="right"))
            deltas = self._delta_log[lo:hi]
        else:
            deltas = self._delta_log
        n_deltas = len(deltas)

        if n_ckpts > 0:
            bid_offs_c = np.empty(n_ckpts + 1, dtype=np.int32)
            ask_offs_c = np.empty(n_ckpts + 1, dtype=np.int32)
            bid_offs_c[0] = ask_offs_c[0] = 0
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                bid_offs_c[k + 1] = bid_offs_c[k] + len(bids_a)
                ask_offs_c[k + 1] = ask_offs_c[k] + len(asks_a)
            total_bid_ckpt = int(bid_offs_c[n_ckpts])
            total_ask_ckpt = int(ask_offs_c[n_ckpts])
            ckpt_bids_flat = np.empty(total_bid_ckpt * 2, dtype=np.float64)
            ckpt_asks_flat = np.empty(total_ask_ckpt * 2, dtype=np.float64)
            for k, cts in enumerate(self._ckpt_ts):
                bids_a, asks_a = self._ckpts[int(cts)]
                b0 = int(bid_offs_c[k]) * 2
                b1 = int(bid_offs_c[k + 1]) * 2
                if b1 > b0:
                    ckpt_bids_flat[b0:b1] = bids_a.ravel()
                a0 = int(ask_offs_c[k]) * 2
                a1 = int(ask_offs_c[k + 1]) * 2
                if a1 > a0:
                    ckpt_asks_flat[a0:a1] = asks_a.ravel()

        if n_deltas > 0:
            dl_ts = np.ascontiguousarray(deltas["ts"])
            dl_side = np.ascontiguousarray(deltas["side"])
            dl_price = np.ascontiguousarray(deltas["price"])
            dl_qty = np.ascontiguousarray(deltas["qty"])

        def _p(arr, ct):
            return ffi.cast(ct, arr.ctypes.data)

        c_all_ts = _p(all_ts_arr, "const long long *")
        c_ckpt_ts = _p(self._ckpt_ts, "const long long *") if n_ckpts > 0 else ffi.NULL
        c_bid_offs = _p(bid_offs_c, "const int *") if n_ckpts > 0 else ffi.NULL
        c_ckbids = (
            _p(ckpt_bids_flat, "const double *") if n_ckpts > 0 and total_bid_ckpt > 0 else ffi.NULL
        )
        c_ask_offs = _p(ask_offs_c, "const int *") if n_ckpts > 0 else ffi.NULL
        c_ckasks = (
            _p(ckpt_asks_flat, "const double *") if n_ckpts > 0 and total_ask_ckpt > 0 else ffi.NULL
        )
        c_dl_ts = _p(dl_ts, "const long long *") if n_deltas > 0 else ffi.NULL
        c_dl_side = _p(dl_side, "const uint8_t *") if n_deltas > 0 else ffi.NULL
        c_dl_price = _p(dl_price, "const double *") if n_deltas > 0 else ffi.NULL
        c_dl_qty = _p(dl_qty, "const double *") if n_deltas > 0 else ffi.NULL

        out_ts_arr = np.empty(n_ts, dtype=np.int64)
        out_bid_offs = np.empty(n_ts + 1, dtype=np.int32)
        out_ask_offs = np.empty(n_ts + 1, dtype=np.int32)
        total_bid_p = ffi.new("int *")
        total_ask_p = ffi.new("int *")

        def _call(bid_data_ptr, ask_data_ptr):
            lib.lobts_iter_seq_states(
                c_all_ts,
                n_ts,
                c_ckpt_ts,
                n_ckpts,
                c_bid_offs,
                c_ckbids,
                c_ask_offs,
                c_ckasks,
                c_dl_ts,
                c_dl_side,
                c_dl_price,
                c_dl_qty,
                n_deltas,
                anchor_idx,
                _p(out_ts_arr, "long long *"),
                _p(out_bid_offs, "int *"),
                _p(out_ask_offs, "int *"),
                bid_data_ptr,
                ask_data_ptr,
                total_bid_p,
                total_ask_p,
            )

        _call(ffi.NULL, ffi.NULL)
        total_bid = total_bid_p[0]
        total_ask = total_ask_p[0]

        bid_data = np.empty(total_bid * 2, dtype=np.float64)
        ask_data = np.empty(total_ask * 2, dtype=np.float64)
        _call(
            _p(bid_data, "double *") if total_bid > 0 else ffi.NULL,
            _p(ask_data, "double *") if total_ask > 0 else ffi.NULL,
        )

        return out_ts_arr, out_bid_offs, out_ask_offs, bid_data, ask_data

    def build_checkpoints(self, n=_AUTO_CHECKPOINTS):
        """
        Generate n evenly spaced checkpoints from the current delta log.

        Makes random access via self[ts] O(D/n) instead of O(D).
        Called automatically after loading from parquet in lazy mode.

        Checkpoints are inserted at timestamps that divide the delta log into n
        roughly equal segments.  Each checkpoint stores the full LOB state
        *before* any deltas at that timestamp, which matches the convention
        expected by _reconstruct.
        """
        if self._mode != "lazy" or len(self._delta_log) == 0:
            return

        # Pick n evenly spaced positions across the delta log
        total = len(self._delta_log)
        positions = np.unique(np.linspace(0, total - 1, n + 1, dtype=int)[1:])  # skip pos 0

        # Find the earliest existing checkpoint to start the forward pass
        if len(self._ckpt_ts) > 0:
            start_ts = int(self._ckpt_ts[0])
            bids_arr, asks_arr = self._ckpts[start_ts]
            bid_dict: dict = {float(p): float(q) for p, q in bids_arr}
            ask_dict: dict = {float(p): float(q) for p, q in asks_arr}
            delta_start = int(np.searchsorted(self._delta_log["ts"], start_ts, side="left"))
        else:
            bid_dict = {}
            ask_dict = {}
            delta_start = 0

        new_ckpts: list = []

        pos_iter = iter(positions)
        next_pos = next(pos_iter, None)

        # Extract column views once — avoids per-iteration structured-array field lookup
        dl_ts = self._delta_log["ts"]
        dl_side = self._delta_log["side"]
        dl_price = self._delta_log["price"]
        dl_qty = self._delta_log["qty"]
        # Cache the anchor timestamp to avoid repeated len() + indexing inside the loop
        first_ckpt_ts = int(self._ckpt_ts[0]) if len(self._ckpt_ts) > 0 else None

        for i in range(delta_start, total):
            # If we hit a later existing checkpoint, reset state (skip forward pass overhead)
            cur_ts = int(dl_ts[i])
            if cur_ts in self._ckpts and cur_ts != first_ckpt_ts:
                bids_arr, asks_arr = self._ckpts[cur_ts]
                bid_dict = {float(p): float(q) for p, q in bids_arr}
                ask_dict = {float(p): float(q) for p, q in asks_arr}

            # Before applying delta i, check if i is a target checkpoint position
            while next_pos is not None and i == next_pos:
                ckpt_ts = cur_ts
                if ckpt_ts not in self._ckpts:
                    new_ckpts.append((ckpt_ts, dict(bid_dict), dict(ask_dict)))
                next_pos = next(pos_iter, None)

            # Apply delta i
            side_int = int(dl_side[i])
            price = float(dl_price[i])
            qty = float(dl_qty[i])
            if side_int == 0:
                if qty == 0.0:
                    bid_dict.pop(price, None)
                else:
                    bid_dict[price] = qty
            else:
                if qty == 0.0:
                    ask_dict.pop(price, None)
                else:
                    ask_dict[price] = qty

        # Store collected checkpoints
        for ckpt_ts, bd, ad in new_ckpts:
            self._ckpts[ckpt_ts] = (
                _make_bid_array(list(bd.items())),
                _make_ask_array(list(ad.items())),
            )
        if new_ckpts:
            new_ts = np.array([c[0] for c in new_ckpts], dtype=np.int64)
            self._ckpt_ts = np.sort(np.concatenate([self._ckpt_ts, new_ts]))

    def spread_ts(self, start_ts=None, end_ts=None):
        """
        Return spread time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and spread values
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, bids, asks, _, _ = self._seq_extract_best(start_ts, end_ts)
            import math

            spreads = [
                a - b if not math.isnan(a) and not math.isnan(b) else float("nan")
                for b, a in zip(bids, asks)
            ]
            return pd.Series(spreads, index=ts_list, name="spread")

        spreads = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            spreads.append(self[ts].spread)
            timestamps.append(ts)

        return pd.Series(spreads, index=timestamps, name="spread")

    def midprice_ts(self, start_ts=None, end_ts=None):
        """
        Return mid-price time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and mid-price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, bids, asks, _, _ = self._seq_extract_best(start_ts, end_ts)
            import math

            midprices = [
                (b + a) / 2 if not math.isnan(b) and not math.isnan(a) else float("nan")
                for b, a in zip(bids, asks)
            ]
            return pd.Series(midprices, index=ts_list, name="midprice")

        midprices = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self[ts]
            midprices.append((lob.bid[0] + lob.ask[0]) / 2)
            timestamps.append(ts)

        return pd.Series(midprices, index=timestamps, name="midprice")

    def bid_ts(self, start_ts=None, end_ts=None):
        """
        Return best bid time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and bid price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, bids, _, _, _ = self._seq_extract_best(start_ts, end_ts)
            return pd.Series(bids, index=ts_list, name="bid")

        bids = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            bids.append(self[ts].bid[0])
            timestamps.append(ts)

        return pd.Series(bids, index=timestamps, name="bid")

    def ask_ts(self, start_ts=None, end_ts=None):
        """
        Return best ask time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and ask price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, _, asks, _, _ = self._seq_extract_best(start_ts, end_ts)
            return pd.Series(asks, index=ts_list, name="ask")

        asks = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            asks.append(self[ts].ask[0])
            timestamps.append(ts)

        return pd.Series(asks, index=timestamps, name="ask")

    def bidq_ts(self, start_ts=None, end_ts=None):
        """
        Return best bid quantity time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and bid quantity values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, _, _, bidqs, _ = self._seq_extract_best(start_ts, end_ts)
            return pd.Series(bidqs, index=ts_list, name="bidq")

        bidqs = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            bidqs.append(self[ts].bidq[0])
            timestamps.append(ts)

        return pd.Series(bidqs, index=timestamps, name="bidq")

    def askq_ts(self, start_ts=None, end_ts=None):
        """
        Return best ask quantity time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and ask quantity values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, _, _, _, askqs = self._seq_extract_best(start_ts, end_ts)
            return pd.Series(askqs, index=ts_list, name="askq")

        askqs = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            askqs.append(self[ts].askq[0])
            timestamps.append(ts)

        return pd.Series(askqs, index=timestamps, name="askq")

    @property
    def spread(self):
        """Return spread time series as property."""
        return self.spread_ts()

    @property
    def bid(self):
        """Return bid time series as property."""
        return self.bid_ts()

    @property
    def ask(self):
        """Return ask time series as property."""
        return self.ask_ts()

    @property
    def midprice(self):
        """Return mid-price time series as property."""
        return self.midprice_ts()

    @property
    def bidq(self):
        """Return bid quantity time series as property."""
        return self.bidq_ts()

    @property
    def askq(self):
        """Return ask quantity time series as property."""
        return self.askq_ts()

    def vw_midprice_ts(self, start_ts=None, end_ts=None):
        """
        Return volume-weighted mid-price time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and vw_midprice values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, bids, asks, bidqs, askqs = self._seq_extract_best(start_ts, end_ts)
            import math

            vw_midprices = [
                (b * bq + a * aq) / (bq + aq)
                if not math.isnan(b) and not math.isnan(a) and (bq + aq) != 0
                else float("nan")
                for b, a, bq, aq in zip(bids, asks, bidqs, askqs)
            ]
            return pd.Series(vw_midprices, index=ts_list, name="vw_midprice")

        vw_midprices = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self[ts]
            bid_price = lob.bid[0]
            bid_size = lob.bidq[0]
            ask_price = lob.ask[0]
            ask_size = lob.askq[0]
            if bid_size + ask_size == 0:
                import math

                vw_midprices.append(math.nan)
            else:
                vw_midprices.append(
                    (bid_price * bid_size + ask_price * ask_size) / (bid_size + ask_size)
                )
            timestamps.append(ts)

        return pd.Series(vw_midprices, index=timestamps, name="vw_midprice")

    @property
    def vw_midprice(self):
        """Return volume-weighted mid-price time series as property."""
        return self.vw_midprice_ts()

    def vi_ts(self, start_ts=None, end_ts=None):
        """
        Return volume imbalance time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and vi values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        if self._mode == "lazy":
            ts_list, _, _, bidqs, askqs = self._seq_extract_best(start_ts, end_ts)

            vi_values = [
                (bq - aq) / (bq + aq) if (bq + aq) != 0 else 0.0
                for bq, aq in zip(bidqs, askqs)
            ]
            return pd.Series(vi_values, index=ts_list, name="vi")

        vi_values = []
        timestamps = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self[ts]
            bid_size = lob.bidq[0]
            ask_size = lob.askq[0]
            if bid_size + ask_size == 0:
                vi_values.append(0.0)
            else:
                vi_values.append((bid_size - ask_size) / (bid_size + ask_size))
            timestamps.append(ts)

        return pd.Series(vi_values, index=timestamps, name="vi")

    @property
    def vi(self):
        """Return volume imbalance time series as property."""
        return self.vi_ts()

    def _duration_seconds(self):
        """Return timeline duration in seconds using the configured timestamp_unit."""
        ts = self.len_ts
        return ts / _TS_UNITS[self.timestamp_unit] if ts > 0 else 0.0

    def _iter_transitions(self):
        """
        Yield per-LOB-transition statistics via a single forward pass.

        Yields:
            (ts, bid_arr_vol, bid_can_vol, ask_arr_vol, ask_can_vol,
                 bid_arr_cnt, bid_can_cnt, ask_arr_cnt, ask_can_cnt)

        *_vol: total quantity added/removed on that side.
        *_cnt: number of price levels that arrived/cancelled on that side.
        """
        it = self._iter_seq_states()
        try:
            _, prev_bid, prev_ask = next(it)
        except StopIteration:
            return
        for ts, curr_bid, curr_ask in it:
            bid_arr_vol = bid_can_vol = 0.0
            bid_arr_cnt = bid_can_cnt = 0
            ask_arr_vol = ask_can_vol = 0.0
            ask_arr_cnt = ask_can_cnt = 0

            for price, new_qty in curr_bid.items():
                if price not in prev_bid:
                    bid_arr_vol += new_qty
                    bid_arr_cnt += 1
                elif new_qty > prev_bid[price]:
                    bid_arr_vol += new_qty - prev_bid[price]
                    bid_arr_cnt += 1

            for price, old_qty in prev_bid.items():
                if price not in curr_bid:
                    bid_can_vol += old_qty
                    bid_can_cnt += 1
                elif curr_bid[price] < old_qty:
                    bid_can_vol += old_qty - curr_bid[price]
                    bid_can_cnt += 1

            for price, new_qty in curr_ask.items():
                if price not in prev_ask:
                    ask_arr_vol += new_qty
                    ask_arr_cnt += 1
                elif new_qty > prev_ask[price]:
                    ask_arr_vol += new_qty - prev_ask[price]
                    ask_arr_cnt += 1

            for price, old_qty in prev_ask.items():
                if price not in curr_ask:
                    ask_can_vol += old_qty
                    ask_can_cnt += 1
                elif curr_ask[price] < old_qty:
                    ask_can_vol += old_qty - curr_ask[price]
                    ask_can_cnt += 1

            prev_bid, prev_ask = curr_bid, curr_ask
            yield (
                ts,
                bid_arr_vol,
                bid_can_vol,
                ask_arr_vol,
                ask_can_vol,
                bid_arr_cnt,
                bid_can_cnt,
                ask_arr_cnt,
                ask_can_cnt,
            )

    @property
    def order_arrival_volume(self):
        """
        Total quantity added to the order book across all LOB transitions.

        Counts new price levels and quantity increases on both sides.
        """
        return sum(r[1] + r[3] for r in self._iter_transitions())

    @property
    def order_cancel_volume(self):
        """
        Total quantity removed from the order book across all LOB transitions.

        Counts full cancels and partial size decreases on both sides.
        """
        return sum(r[2] + r[4] for r in self._iter_transitions())

    def update_volume(self):
        """Total quantity changed in the order book (arrivals + cancels)."""
        return self.order_arrival_volume + self.order_cancel_volume

    @property
    def order_arrival_frequency(self):
        """
        Order arrival events per second (both sides combined).

        Each price level that appears new or increases in quantity counts as one event.
        """
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[5] + r[7] for r in self._iter_transitions()) / dur

    @property
    def order_cancel_frequency(self):
        """
        Order cancel events per second (both sides combined).

        Each price level that disappears or decreases in quantity counts as one event.
        """
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[6] + r[8] for r in self._iter_transitions()) / dur

    @property
    def bid_order_arrival_frequency(self):
        """Bid-side order arrival events per second."""
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[5] for r in self._iter_transitions()) / dur

    @property
    def ask_order_arrival_frequency(self):
        """Ask-side order arrival events per second."""
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[7] for r in self._iter_transitions()) / dur

    @property
    def bid_order_cancel_frequency(self):
        """Bid-side order cancel events per second."""
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[6] for r in self._iter_transitions()) / dur

    @property
    def ask_order_cancel_frequency(self):
        """Ask-side order cancel events per second."""
        dur = self._duration_seconds()
        if dur == 0:
            return 0.0
        return sum(r[8] for r in self._iter_transitions()) / dur

    def _ofi_from_deltas(self):
        """
        Compute OFI directly from the delta log (lazy mode only).

        Maintains bid/ask price→qty maps in a single forward pass over deltas,
        grouping by timestamp. At each timestamp, computes:
            OFI = (bid_arr - bid_can) - (ask_arr - ask_can)

        Returns:
            tuple of (timestamps: list[int], ofi_values: list[float])
        """
        # Initial state from earliest checkpoint, if any
        bid_state = {}
        ask_state = {}
        ckpt_ts = self._ckpt_ts
        n_ckpts = len(ckpt_ts)
        first_ckpt_ts = int(ckpt_ts[0]) if n_ckpts > 0 else None

        if first_ckpt_ts is not None:
            bids_arr, asks_arr = self._ckpts[first_ckpt_ts]
            for p, q in bids_arr:
                bid_state[float(p)] = float(q)
            for p, q in asks_arr:
                ask_state[float(p)] = float(q)

        deltas = self._delta_log
        n_d = len(deltas)
        if n_d == 0:
            return [], []

        # Find starting point: only process deltas at or after first checkpoint
        start_idx = 0
        if first_ckpt_ts is not None:
            start_idx = int(np.searchsorted(deltas["ts"], first_ckpt_ts, side="left"))

        timestamps = []
        ofi_vals = []

        i = start_idx
        while i < n_d:
            ts = int(deltas["ts"][i])
            bid_arr = bid_can = 0.0
            ask_arr = ask_can = 0.0

            # Process all deltas at this timestamp
            while i < n_d and int(deltas["ts"][i]) == ts:
                side = deltas["side"][i]
                price = float(deltas["price"][i])
                new_qty = float(deltas["qty"][i])

                if side == 0:  # bid
                    old_qty = bid_state.get(price, 0.0)
                    if new_qty > old_qty:
                        bid_arr += new_qty - old_qty
                    elif new_qty < old_qty:
                        bid_can += old_qty - new_qty
                    if new_qty > 0.0:
                        bid_state[price] = new_qty
                    else:
                        bid_state.pop(price, None)
                else:  # ask
                    old_qty = ask_state.get(price, 0.0)
                    if new_qty > old_qty:
                        ask_arr += new_qty - old_qty
                    elif new_qty < old_qty:
                        ask_can += old_qty - new_qty
                    if new_qty > 0.0:
                        ask_state[price] = new_qty
                    else:
                        ask_state.pop(price, None)

                i += 1

            timestamps.append(ts)
            ofi_vals.append((bid_arr - bid_can) - (ask_arr - ask_can))

        return timestamps, ofi_vals

    @property
    def order_flow_imbalance(self):
        """
        Order flow imbalance (OFI) time series.

        OFI(t) = (bid_arrival_vol − bid_cancel_vol) − (ask_arrival_vol − ask_cancel_vol)

        Positive values indicate the bid side is building faster than the ask side
        (bullish pressure). Negative values indicate ask-side dominance (bearish).

        Returns:
            pd.Series indexed by LOB transition timestamps.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for order_flow_imbalance")

        if self._mode == "lazy":
            ts_vals, ofi_vals = self._ofi_from_deltas()
            return pd.Series(ofi_vals, index=ts_vals, name="ofi")

        timestamps = []
        ofi = []
        for ts, bav, bcv, aav, acv, *_ in self._iter_transitions():
            timestamps.append(ts)
            ofi.append((bav - bcv) - (aav - acv))
        return pd.Series(ofi, index=timestamps, name="ofi")

    def diff(self, other):
        """
        Calculate differences between this LOBts and another LOBts.

        Args:
            other: Another LOBts object to compare with

        Returns:
            List of (timestamp, bid_deltas, ask_deltas) tuples
        """
        results = []
        for ts in self.timestamps:
            if ts not in other:
                continue

            lob1 = self[ts]
            lob2 = other[ts]

            bids1 = dict(lob1._bids)
            bids2 = dict(lob2._bids)
            asks1 = dict(lob1._asks)
            asks2 = dict(lob2._asks)

            bid_deltas = []
            ask_deltas = []

            for price, qty in bids2.items():
                if qty != bids1.get(price, 0.0):
                    bid_deltas.append((price, qty))
            for price in bids1:
                if price not in bids2:
                    bid_deltas.append((price, 0.0))

            for price, qty in asks2.items():
                if qty != asks1.get(price, 0.0):
                    ask_deltas.append((price, qty))
            for price in asks1:
                if price not in asks2:
                    ask_deltas.append((price, 0.0))

            results.append((ts, bid_deltas, ask_deltas))

        return results

    def to_pd(self, start_ts=None, end_ts=None):
        """
        Export to pandas DataFrame.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            DataFrame with columns: timestamp, side, level, price, size
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for export methods")

        data = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self[ts]
            for level, (price, size) in enumerate(lob._bids):
                data.append((ts, "b", level, price, size))
            for level, (price, size) in enumerate(lob._asks):
                data.append((ts, "a", level, price, size))

        return pd.DataFrame(data, columns=["timestamp", "side", "level", "price", "size"])

    def to_np(self, start_ts=None, end_ts=None):
        """
        Export to numpy array.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            numpy array with shape (n, 5): [timestamp, side, level, price, size]
        """
        data = []
        for ts in self.timestamps:
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self[ts]
            for level, (price, size) in enumerate(lob._bids):
                data.append([ts, "b", level, price, size])
            for level, (price, size) in enumerate(lob._asks):
                data.append([ts, "a", level, price, size])

        return np.array(data, dtype=object)

    def to_csv(self, path, start_ts=None, end_ts=None):
        """
        Export to CSV file.

        Args:
            path: File path for CSV output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        self.to_pd(start_ts, end_ts).to_csv(path, index=False)

    def to_xlsx(self, path, start_ts=None, end_ts=None):
        """
        Export to XLSX file.

        Args:
            path: File path for XLSX output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        self.to_pd(start_ts, end_ts).to_excel(path, index=False, engine="openpyxl")

    def to_parquet(self, path, start_ts=None, end_ts=None):
        """
        Export to Parquet file.

        Args:
            path: File path for Parquet output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        self.to_pd(start_ts, end_ts).to_parquet(path, engine="pyarrow")

    def __repr__(self) -> str:
        return f"<LOBts[{self.name}] mode={self._mode} snapshots={len(self)}>"
