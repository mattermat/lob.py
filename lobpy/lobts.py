"""
Time Series LOB (Limit Order Book)
"""

from collections import OrderedDict

import numpy as np

from .lob import LOB, _make_ask_array, _make_bid_array
from .sorteddict import SortedDict

_LAZY_DELTA_DTYPE = np.dtype([("ts", "i8"), ("side", "u1"), ("price", "f8"), ("qty", "f8")])
_CACHE_MAXSIZE = 32
_AUTO_CHECKPOINTS = 100


class LOBts:
    """Time series of LOB objects indexed by timestamp."""

    def __init__(self, name=None, tick_size=1, mode="delta") -> None:
        """
        Initialize LOBts.

        Args:
            name: Optional identifier for the time series
            tick_size: Minimum price increment
            mode: 'delta' to store all snapshots eagerly,
                  'latest' to keep only the current snapshot,
                  'lazy' to store only checkpoints + a delta log and reconstruct on demand
        """
        if name is None:
            name = f"lobts{id(self)}"
        self.name = name
        self.tick_size = tick_size
        self._mode = mode

        if mode == "lazy":
            self._delta_log = np.empty(0, dtype=_LAZY_DELTA_DTYPE)
            self._ckpt_ts = np.empty(0, dtype=np.int64)
            self._ckpts = {}
            self._cache = OrderedDict()
            self._ts_lo = None  # inclusive lower bound for timestamps view
            self._ts_hi = None  # inclusive upper bound for timestamps view
        else:
            self._lobs = SortedDict()
            self._timestamps = self._lobs.keys()

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
        return self._timestamps

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
            if timestamp in self._ckpts and not force:
                raise ValueError(
                    f"Timestamp {timestamp} already exists. Use force=True to overwrite."
                )
            self._ckpts[timestamp] = (_make_bid_array(bids), _make_ask_array(asks))
            if timestamp not in self._ckpt_ts:
                self._ckpt_ts = np.sort(np.append(self._ckpt_ts, np.int64(timestamp)))
            self._cache.pop(timestamp, None)
            return

        if self._mode == "latest":
            self._lobs.clear()

        lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size, bids=bids, asks=asks)
        lob.timestamp = timestamp

        if timestamp in self._lobs and not force:
            raise ValueError(f"Timestamp {timestamp} already exists. Use force=True to overwrite.")
        self._lobs[timestamp] = lob

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
            rows = []
            for side, price, size in updates:
                side_int = np.uint8(0) if side in ("b", "bid") else np.uint8(1)
                rows.append((np.int64(timestamp), side_int, np.float64(price), np.float64(size)))
            new_rows = np.array(rows, dtype=_LAZY_DELTA_DTYPE)
            self._delta_log = np.concatenate([self._delta_log, new_rows])
            self._cache.pop(timestamp, None)
            return None

        if len(self._lobs) > 0:
            prev_timestamp = self._lobs.keys()[-1]
            prev_lob = self._lobs[prev_timestamp]
            new_lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size)

            new_lob._bids = prev_lob._bids.copy()
            new_lob._asks = prev_lob._asks.copy()
            new_lob.timestamp = prev_timestamp

            for side, price, size in updates:
                if side == "b":
                    side = "bid"
                elif side == "a":
                    side = "ask"
                new_lob.update(side, price, size, 0)

            new_lob.timestamp = timestamp
            self._lobs[timestamp] = new_lob
        else:
            lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size)
            for side, price, size in updates:
                if side == "b":
                    side = "bid"
                elif side == "a":
                    side = "ask"
                lob.update(side, price, size, 0)
            lob.timestamp = timestamp
            self._lobs[timestamp] = lob

        return self._lobs[timestamp]

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
            return self._reconstruct(t)

        try:
            return self._lobs[t]
        except KeyError:
            return None

    def _reconstruct(self, t):
        """Reconstruct LOB at timestamp t from nearest checkpoint + delta replay."""
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]

        if len(self._ckpt_ts) == 0:
            return None

        idx = int(np.searchsorted(self._ckpt_ts, t, side="right")) - 1
        if idx < 0:
            return None

        t0 = int(self._ckpt_ts[idx])
        bids_arr, asks_arr = self._ckpts[t0]

        # Slice delta log from t0 to t (both inclusive)
        if len(self._delta_log) > 0:
            lo = int(np.searchsorted(self._delta_log["ts"], t0, side="left"))
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
            result = LOBts(name=f"{self.name}_range", tick_size=self.tick_size, mode="lazy")

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

            return result

        result = LOBts(name=f"{self.name}_range", tick_size=self.tick_size, mode=self._mode)
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            result._lobs[ts] = self._lobs[ts]
        return result

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Return number of LOB timestamps stored."""
        if self._mode == "lazy":
            return len(self.timestamps)
        return len(self._lobs)

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
            if timestamp in self._ckpts:
                return True
            if len(self._delta_log) == 0:
                return False
            idx = int(np.searchsorted(self._delta_log["ts"], timestamp))
            return idx < len(self._delta_log) and self._delta_log["ts"][idx] == timestamp
        return timestamp in self._lobs

    def __iter__(self):
        """Iterate over LOB objects in timestamp order."""
        if self._mode == "lazy":
            for ts in self.timestamps:
                yield self[ts]
            return
        return iter(self._lobs.values())

    # ------------------------------------------------------------------
    # Time series analytics — mode-agnostic via self.timestamps / self[ts]
    # ------------------------------------------------------------------

    def _seq_extract_best(self, start_ts=None, end_ts=None):
        """
        Single forward pass through checkpoints + deltas.

        Returns (timestamps, best_bids, best_asks, best_bidqs, best_askqs).
        Only valid in lazy mode; O(N + D) vs the naive O(N * D/C).
        """
        all_ts = self.timestamps  # sorted
        if start_ts is not None:
            all_ts = [t for t in all_ts if t >= start_ts]
        if end_ts is not None:
            all_ts = [t for t in all_ts if t <= end_ts]
        if not all_ts:
            return [], [], [], [], []

        first_ts = all_ts[0]
        last_ts = all_ts[-1]

        # Anchor: nearest checkpoint at or before first_ts
        if len(self._ckpt_ts) > 0:
            anchor_idx = int(np.searchsorted(self._ckpt_ts, first_ts, side="right")) - 1
        else:
            anchor_idx = -1

        if anchor_idx >= 0:
            anchor_ts = int(self._ckpt_ts[anchor_idx])
            bids_arr, asks_arr = self._ckpts[anchor_ts]
            bid_dict = {float(p): float(q) for p, q in bids_arr}
            ask_dict = {float(p): float(q) for p, q in asks_arr}
        else:
            anchor_ts = None
            bid_dict = {}
            ask_dict = {}

        # Slice the delta log to the relevant window
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

        ckpt_set = set(int(t) for t in self._ckpt_ts)

        # Batch-compute the delta_idx for every checkpoint that falls within the range.
        # Replaces per-checkpoint searchsorted calls inside the main loop.
        n_deltas = len(deltas)
        ckpt_delta_pos: dict = {}
        if n_deltas > 0:
            dl_ts = deltas["ts"]
            dl_side = deltas["side"]
            dl_price = deltas["price"]
            dl_qty = deltas["qty"]
            ckpts_in_range = [
                ts for ts in ckpt_set if ts != anchor_ts and first_ts <= ts <= last_ts
            ]
            if ckpts_in_range:
                positions = np.searchsorted(dl_ts, ckpts_in_range, side="left")
                ckpt_delta_pos = dict(zip(ckpts_in_range, positions.tolist()))
        else:
            dl_ts = dl_side = dl_price = dl_qty = None

        nan = float("nan")

        # Initialise running best bid/ask from the anchor state
        best_bid = max(bid_dict) if bid_dict else nan
        best_ask = min(ask_dict) if ask_dict else nan

        delta_idx = 0
        out_ts: list = []
        out_bids: list = []
        out_asks: list = []
        out_bidqs: list = []
        out_askqs: list = []

        for t in all_ts:
            if t in ckpt_set and t != anchor_ts:
                # New full checkpoint — reset state
                bids_arr, asks_arr = self._ckpts[t]
                bid_dict = {float(p): float(q) for p, q in bids_arr}
                ask_dict = {float(p): float(q) for p, q in asks_arr}
                delta_idx = ckpt_delta_pos.get(t, 0)
                best_bid = max(bid_dict) if bid_dict else nan
                best_ask = min(ask_dict) if ask_dict else nan

            # Apply all deltas at timestamp t; n_deltas cached to avoid len() per iteration
            while delta_idx < n_deltas and dl_ts[delta_idx] == t:
                price = float(dl_price[delta_idx])
                qty = float(dl_qty[delta_idx])
                if int(dl_side[delta_idx]) == 0:  # bid
                    if qty == 0.0:
                        bid_dict.pop(price, None)
                        if price == best_bid:
                            best_bid = max(bid_dict) if bid_dict else nan
                    else:
                        bid_dict[price] = qty
                        if best_bid != best_bid or price > best_bid:  # nan or new max
                            best_bid = price
                else:  # ask
                    if qty == 0.0:
                        ask_dict.pop(price, None)
                        if price == best_ask:
                            best_ask = min(ask_dict) if ask_dict else nan
                    else:
                        ask_dict[price] = qty
                        if best_ask != best_ask or price < best_ask:  # nan or new min
                            best_ask = price
                delta_idx += 1

            out_ts.append(t)
            out_bids.append(best_bid)
            out_asks.append(best_ask)
            out_bidqs.append(bid_dict.get(best_bid, nan))
            out_askqs.append(ask_dict.get(best_ask, nan))

        return out_ts, out_bids, out_asks, out_bidqs, out_askqs

    def _iter_seq_states(self, start_ts=None, end_ts=None):
        """
        Yield (ts, bid_dict, ask_dict) for every LOB timestamp via a single forward pass.

        bid_dict / ask_dict are plain {price: qty} dicts (copies, safe to read after yield).
        In lazy mode this is O(N + D); in other modes it iterates the stored LOB objects.
        """
        if self._mode != "lazy":
            for ts, lob in self._lobs.items():
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    break
                bids = {float(p): float(q) for p, q in lob._bids}
                asks = {float(p): float(q) for p, q in lob._asks}
                yield ts, bids, asks
            return

        # Lazy mode: same forward-pass structure as _seq_extract_best
        all_ts = self.timestamps
        if start_ts is not None:
            all_ts = [t for t in all_ts if t >= start_ts]
        if end_ts is not None:
            all_ts = [t for t in all_ts if t <= end_ts]
        if not all_ts:
            return

        first_ts = all_ts[0]
        last_ts = all_ts[-1]

        if len(self._ckpt_ts) > 0:
            anchor_idx = int(np.searchsorted(self._ckpt_ts, first_ts, side="right")) - 1
        else:
            anchor_idx = -1

        if anchor_idx >= 0:
            anchor_ts = int(self._ckpt_ts[anchor_idx])
            bids_arr, asks_arr = self._ckpts[anchor_ts]
            bid_dict: dict = {float(p): float(q) for p, q in bids_arr}
            ask_dict: dict = {float(p): float(q) for p, q in asks_arr}
        else:
            anchor_ts = None
            bid_dict = {}
            ask_dict = {}

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

        ckpt_set = set(int(t) for t in self._ckpt_ts)
        delta_idx = 0

        for t in all_ts:
            if t in ckpt_set and t != anchor_ts:
                bids_arr, asks_arr = self._ckpts[t]
                bid_dict = {float(p): float(q) for p, q in bids_arr}
                ask_dict = {float(p): float(q) for p, q in asks_arr}
                delta_idx = (
                    int(np.searchsorted(deltas["ts"], t, side="left")) if len(deltas) > 0 else 0
                )

            while delta_idx < len(deltas) and deltas["ts"][delta_idx] == t:
                side_int = int(deltas["side"][delta_idx])
                price = float(deltas["price"][delta_idx])
                qty = float(deltas["qty"][delta_idx])
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
                delta_idx += 1

            yield t, dict(bid_dict), dict(ask_dict)

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

    @property
    def vw_midprice(self):
        """Return volume-weighted mid-price time series as property."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        vw_midprices = []
        timestamps = []
        for ts in self.timestamps:
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
    def vi(self):
        """Return volume imbalance time series as property."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        vi_values = []
        timestamps = []
        for ts in self.timestamps:
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
    def arrival_frequency(self):
        """
        Return total arrival frequency (quantity added to order book across all snapshots).

        In L2 order books, an arrival can be:
        - New level: a price level that didn't exist before
        - Quantity increase: existing level size increases (X -> Y where Y > X)

        Returns total quantity added (positive changes) across all transitions.
        """
        timestamps_list = list(self.timestamps)
        if len(timestamps_list) <= 1:
            return 0

        total_arrivals = 0
        for i in range(1, len(timestamps_list)):
            prev_lob = self[timestamps_list[i - 1]]
            curr_lob = self[timestamps_list[i]]

            prev_bid_dict = dict(prev_lob._bids)
            prev_ask_dict = dict(prev_lob._asks)

            for price, new_qty in curr_lob._bids:
                if price not in prev_bid_dict:
                    total_arrivals += new_qty
                elif new_qty > prev_bid_dict[price]:
                    total_arrivals += new_qty - prev_bid_dict[price]

            for price, new_qty in curr_lob._asks:
                if price not in prev_ask_dict:
                    total_arrivals += new_qty
                elif new_qty > prev_ask_dict[price]:
                    total_arrivals += new_qty - prev_ask_dict[price]

        return total_arrivals

    @property
    def cancel_frequency(self):
        """
        Return total cancel frequency (quantity removed from order book across all snapshots).

        In L2 order books, a cancel can be:
        - Full cancel: level completely removed (size goes to X -> 0 or level disappears)
        - Partial cancel: level size decreases (size goes from X -> Y where Y < X)

        Returns total quantity removed (negative changes) across all transitions.
        """
        timestamps_list = list(self.timestamps)
        if len(timestamps_list) <= 1:
            return 0

        total_cancels = 0
        for i in range(1, len(timestamps_list)):
            prev_lob = self[timestamps_list[i - 1]]
            curr_lob = self[timestamps_list[i]]

            curr_bid_dict = dict(curr_lob._bids)
            curr_ask_dict = dict(curr_lob._asks)

            for price, old_qty in prev_lob._bids:
                if price not in curr_bid_dict:
                    total_cancels += old_qty
                elif curr_bid_dict[price] < old_qty:
                    total_cancels += old_qty - curr_bid_dict[price]

            for price, old_qty in prev_lob._asks:
                if price not in curr_ask_dict:
                    total_cancels += old_qty
                elif curr_ask_dict[price] < old_qty:
                    total_cancels += old_qty - curr_ask_dict[price]

        return total_cancels

    def update_frequency(self):
        """Calculate update frequency (arrivals + cancels)."""
        return self.arrival_frequency + self.cancel_frequency

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
