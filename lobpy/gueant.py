"""
Guéant intensity function estimation.

Model: λ(δ) = A * exp(-k * δ)
Reference: Guéant, Lehalle, Fernandez-Tapia (2013)
# TODO: add link to the paper
"""

from bisect import bisect_left
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _delta_contribs(bid_dict, ask_dict, side, tick_size):
    """
    Return {delta: level_count} for one LOB snapshot.

    ask side: δ = (ask_price - best_bid) / tick_size
    bid side: δ = (best_ask  - bid_price) / tick_size
    """
    if not bid_dict or not ask_dict:
        return {}
    best_bid = max(bid_dict)
    best_ask = min(ask_dict)
    if best_bid <= 0 or best_ask <= 0:
        return {}
    dc: dict = {}
    if side == "a":
        for price in ask_dict:
            d = int(round((price - best_bid) / tick_size))
            if d >= 0:
                dc[d] = dc.get(d, 0) + 1
    else:
        for price in bid_dict:
            d = int(round((best_ask - price) / tick_size))
            if d >= 0:
                dc[d] = dc.get(d, 0) + 1
    return dc


def _trade_delta(trade_price, bid_dict, ask_dict, side, tick_size):
    """Return integer δ for one trade, or None if book is invalid."""
    if not bid_dict or not ask_dict:
        return None
    best_bid = max(bid_dict)
    best_ask = min(ask_dict)
    if best_bid <= 0 or best_ask <= 0:
        return None
    if side == "a":
        d = int(round((trade_price - best_bid) / tick_size))
    else:
        d = int(round((best_ask - trade_price) / tick_size))
    return d if d >= 0 else None


def _precompute(tl, side):
    """
    Single forward pass over the LOB to build everything needed for Guéant.

    Returns
    -------
    lob_states : list of (ts, bid_dict, ask_dict)
    intervals  : list of (ts_start, ts_end, duration, delta_counts_dict)
    trade_deltas : list of (ts, delta_int) for trades on the given side, sorted by ts
    """
    tick_size = tl.tick_size
    trade_side_str = "b" if side == "a" else "s"

    lob_states = list(tl._lobts._iter_seq_states())
    if not lob_states:
        return [], [], []

    end_ts = max(
        lob_states[-1][0],
        max((t.timestamp for t in tl._trades), default=lob_states[-1][0]),
    )

    # LOB intervals
    intervals = []
    for i, (ts, bid_dict, ask_dict) in enumerate(lob_states):
        next_ts = lob_states[i + 1][0] if i + 1 < len(lob_states) else end_ts
        duration = next_ts - ts
        if duration <= 0:
            continue
        dc = _delta_contribs(bid_dict, ask_dict, side, tick_size)
        intervals.append((ts, next_ts, duration, dc))

    # Trade deltas (binary search into lob_states)
    lob_ts_arr = np.array([s[0] for s in lob_states])
    relevant_trades = sorted(
        (t for t in tl._trades if t.side == trade_side_str),
        key=lambda t: t.timestamp,
    )
    trade_deltas = []
    for trade in relevant_trades:
        idx = int(np.searchsorted(lob_ts_arr, trade.timestamp, side="right")) - 1
        if idx < 0:
            continue
        _, bid_dict, ask_dict = lob_states[idx]
        d = _trade_delta(trade.price, bid_dict, ask_dict, side, tick_size)
        if d is not None:
            trade_deltas.append((trade.timestamp, d))

    return lob_states, intervals, trade_deltas


def _assemble_raw(n_counts, t_vec, max_d):
    """Build raw λ̂(δ) DataFrame from N dict and T numpy vector."""
    all_d = sorted(set(n_counts.keys()) | set(int(x) for x in np.nonzero(t_vec)[0]))
    rows = []
    for d in all_d:
        n = n_counts.get(d, 0)
        t = float(t_vec[d]) if d < max_d else 0.0
        lam = n / t if t > 0 and n > 0 else float("nan")
        rows.append((d, n, t, lam))
    return pd.DataFrame(rows, columns=["delta", "N", "T", "lambda"])


def _aggregate_buckets(raw, thresholds):
    """
    Aggregate integer-δ buckets into custom threshold bins.

    thresholds: sorted list of upper bounds, e.g. [1, 3, 5, 10].
    Bin i covers (thresholds[i-1], thresholds[i]], with the first bin starting at 0.
    Each bin's λ̂ = N_sum / T_sum; the threshold is used as the representative δ.

    Args:
        raw:        pd.DataFrame from _compute_buckets (columns: delta, N, T, lambda)
        thresholds: list of int/float upper bounds

    Returns:
        pd.DataFrame with columns [delta, N, T, lambda], one row per threshold.
    """
    thresholds = sorted(thresholds)
    rows = []
    lower = -1  # exclusive lower bound; first bin covers delta >= 0
    for upper in thresholds:
        mask = (raw["delta"] > lower) & (raw["delta"] <= upper)
        subset = raw[mask]
        N_total = subset["N"].sum()
        T_total = subset["T"].sum()
        lam = N_total / T_total if T_total > 0 and N_total > 0 else float("nan")
        rows.append((upper, int(N_total), T_total, lam))
        lower = upper
    # overflow bin: all δ > last threshold (not used in fit, lambda=nan)
    overflow = raw[raw["delta"] > lower]
    if not overflow.empty:
        N_total = overflow["N"].sum()
        T_total = overflow["T"].sum()
        rows.append((float("inf"), int(N_total), T_total, float("nan")))
    return pd.DataFrame(rows, columns=["delta", "N", "T", "lambda"])


def _fit(buckets):
    """
    Fit λ(δ) = A * exp(-k * δ) via log-linear regression.

    Args:
        buckets: pd.DataFrame from _compute_buckets

    Returns:
        (A, k) floats, or (nan, nan) if fewer than 2 valid points.
    """
    valid = buckets.dropna(subset=["lambda"])
    valid = valid[valid["lambda"] > 0]
    if len(valid) < 2:
        return float("nan"), float("nan")
    x = valid["delta"].values.astype(float)
    y = np.log(valid["lambda"].values.astype(float))
    intercept, slope = np.polynomial.polynomial.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)


# ---------------------------------------------------------------------------
# Scalar (full-timeline) computation
# ---------------------------------------------------------------------------

def _compute_buckets(tl, side):
    """
    Compute empirical intensity buckets λ̂(δ) = N(δ) / T(δ) for one side.

    Uses a single forward pass over the LOB state sequence — O(N_lob + D + N_trades).

    For each integer tick distance δ:
      N(δ) = number of trades that occurred at distance δ
      T(δ) = total time the book exposed liquidity at distance δ

    Args:
        tl:   TL instance
        side: 'a' (ask side) or 'b' (bid side)

    Returns:
        pd.DataFrame with columns [delta, N, T, lambda]
    """
    _, intervals, trade_deltas = _precompute(tl, side)
    if not intervals:
        return pd.DataFrame(columns=["delta", "N", "T", "lambda"])

    T: dict = defaultdict(float)
    for _, _, duration, dc in intervals:
        for d, cnt in dc.items():
            T[d] += duration * cnt

    N: dict = defaultdict(int)
    for _, d in trade_deltas:
        N[d] += 1

    all_deltas = sorted(set(N) | set(T))
    rows = [
        (d, N.get(d, 0), T.get(d, 0.0),
         N[d] / T[d] if T.get(d, 0.0) > 0 and N.get(d, 0) > 0 else float("nan"))
        for d in all_deltas
    ]
    return pd.DataFrame(rows, columns=["delta", "N", "T", "lambda"])


# ---------------------------------------------------------------------------
# Rolling computation
# ---------------------------------------------------------------------------

def _compute_rolling_gueant(tl, side, window_size, buckets=None):
    """
    Compute rolling Guéant (A, k) at every trade timestamp.

    Uses:
    - A single forward pass to precompute LOB intervals and trade δ values
    - A numpy cumsum matrix for T(δ): O(MAX_D) per window lookup
    - A sliding deque for N(δ): O(1) amortised per trade

    Overall: O((N_lob + N_trades) × K_levels) instead of
             O(N_timestamps × N_lob_in_window × D/C).
    """
    _, intervals, trade_deltas = _precompute(tl, side)

    if not intervals or not trade_deltas:
        return pd.Series(name=f"gueant_A_{side}"), pd.Series(name=f"gueant_k_{side}")

    # Determine MAX_D for array sizing
    MAX_D = max(
        max((max(dc.keys(), default=0) for _, _, _, dc in intervals), default=0),
        max((d for _, d in trade_deltas), default=0),
    ) + 1

    # numpy arrays for fast binary search
    intervals_start = np.array([iv[0] for iv in intervals], dtype=np.int64)
    intervals_end   = np.array([iv[1] for iv in intervals], dtype=np.int64)

    # Cumsum matrix: T_cumsum[i] = Σ_{j < i} duration_j × delta_counts_j
    # T(δ) for intervals [l, r) ≈ T_cumsum[r] - T_cumsum[l]
    T_cumsum = np.zeros((len(intervals) + 1, MAX_D), dtype=np.float64)
    for i, (_, _, duration, dc) in enumerate(intervals):
        T_cumsum[i + 1] = T_cumsum[i]
        for d, cnt in dc.items():
            if d < MAX_D:
                T_cumsum[i + 1, d] += duration * cnt

    A_vals: dict = {}
    k_vals: dict = {}

    nan = float("nan")
    n_intervals = len(intervals)

    # Pre-compute bucket structure outside the loop so per-iteration work is minimal.
    # These are used by the numpy hot path; the scalar path keeps the DataFrame helpers.
    if buckets is not None:
        _thresholds: list = sorted(int(t) for t in buckets)
        _n_b: int = len(_thresholds)
        _thresholds_f = np.array(_thresholds, dtype=np.float64)
        # Inclusive integer ranges [_bkt_lo[i], _bkt_hi[i]) for T_vec slicing
        _bkt_lo = np.array([0] + [_thresholds[i] + 1 for i in range(_n_b - 1)], dtype=np.intp)
        _bkt_hi = np.array([t + 1 for t in _thresholds], dtype=np.intp)

    # All timestamps (LOB + trades), matching the old contract of one value per event
    all_ts = sorted(set(tl._lobts.timestamps) | {t.timestamp for t in tl._trades})

    # Sliding deque for N(δ) — advanced as all_ts progresses
    N_window: dict = defaultdict(int)
    trade_deque: deque = deque()
    trade_right = 0  # next trade_deltas index to add
    n_trade_deltas = len(trade_deltas)

    for ts in all_ts:
        lo = ts - window_size

        # Add trades up to ts
        while trade_right < n_trade_deltas and trade_deltas[trade_right][0] <= ts:
            t_ts, t_d = trade_deltas[trade_right]
            trade_deque.append((t_ts, t_d))
            N_window[t_d] += 1
            trade_right += 1

        # Remove trades before lo
        while trade_deque and trade_deque[0][0] < lo:
            old_ts, old_d = trade_deque.popleft()
            N_window[old_d] -= 1
            if N_window[old_d] == 0:
                del N_window[old_d]

        # T(δ) for window [lo, ts] via cumsum
        left = int(np.searchsorted(intervals_end, lo, side="right"))
        right = int(np.searchsorted(intervals_start, ts, side="right"))

        if left >= right:
            A_vals[ts] = nan
            k_vals[ts] = nan
            continue

        T_vec = T_cumsum[right] - T_cumsum[left]  # new array from subtraction

        # Correct boundary clipping for the left and right edge intervals
        for b in set([left, right - 1]):
            if 0 <= b < n_intervals:
                ts_s, ts_e, dur, dc = intervals[b]
                clipped = int(min(ts_e, ts)) - int(max(ts_s, lo))
                if dur > 0 and clipped != dur:
                    diff = clipped - dur
                    for d, cnt in dc.items():
                        if d < MAX_D:
                            T_vec[d] += diff * cnt

        # --- numpy hot path: no DataFrames ---
        if buckets is not None:
            # Accumulate N into threshold bins from the sparse N_window dict
            N_agg = np.zeros(_n_b)
            for d, count in N_window.items():
                idx = bisect_left(_thresholds, d)
                if idx < _n_b:
                    N_agg[idx] += count

            # Sum T_vec slices for each bucket
            T_agg = np.array([
                T_vec[lo_i:min(hi_i, MAX_D)].sum()
                for lo_i, hi_i in zip(_bkt_lo, _bkt_hi)
            ])

            valid = (T_agg > 0) & (N_agg > 0)
            if valid.sum() < 2:
                A_vals[ts] = nan
                k_vals[ts] = nan
                continue
            x = _thresholds_f[valid]
            y = np.log(N_agg[valid] / T_agg[valid])
        else:
            # No bucketing — one point per integer δ
            T_nz = np.nonzero(T_vec)[0]
            all_d = sorted(set(N_window.keys()) | set(T_nz.tolist()))
            if not all_d:
                A_vals[ts] = nan
                k_vals[ts] = nan
                continue
            n_d = len(all_d)
            N_arr = np.fromiter((N_window.get(d, 0) for d in all_d), dtype=np.float64, count=n_d)
            T_arr = np.fromiter(
                (T_vec[d] if d < MAX_D else 0.0 for d in all_d),
                dtype=np.float64,
                count=n_d,
            )
            valid = (T_arr > 0) & (N_arr > 0)
            if valid.sum() < 2:
                A_vals[ts] = nan
                k_vals[ts] = nan
                continue
            x = np.array(all_d, dtype=np.float64)[valid]
            y = np.log(N_arr[valid] / T_arr[valid])

        intercept, slope = np.polynomial.polynomial.polyfit(x, y, 1)
        A_vals[ts] = float(np.exp(intercept))
        k_vals[ts] = float(-slope)

    side_name = "ask" if side == "a" else "bid"
    return (
        pd.Series(A_vals, name=f"gueant_A_{side_name}"),
        pd.Series(k_vals, name=f"gueant_k_{side_name}"),
    )


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------

class GueantAccessor:
    """
    Accessor for Guéant intensity function parameters (λ(δ) = A · exp(-k · δ)).

    Access via `tl.gueant`:

        A_ask, k_ask = tl.gueant.ask()              # scalars, full timeline
        A_bid, k_bid = tl.gueant.bid()

        A_ask_ts, k_ask_ts = tl.gueant.ask(window)  # pd.Series, rolling
        A_bid_ts, k_bid_ts = tl.gueant.bid(window)

        df = tl.gueant.buckets('a')                  # inspect raw λ̂(δ) buckets
        df = tl.gueant.buckets('b')
    """

    def __init__(self, tl):
        self._tl = tl

    def buckets(self, side, buckets=None):
        """
        Return empirical intensity buckets λ̂(δ) = N(δ) / T(δ).

        Args:
            side:    'a' (ask) or 'b' (bid)
            buckets: Optional list of δ thresholds (same semantics as ask/bid).

        Returns:
            pd.DataFrame with columns [delta, N, T, lambda]
        """
        raw = _compute_buckets(self._tl, side)
        return _aggregate_buckets(raw, buckets) if buckets is not None else raw

    def ask(self, window_size=None, buckets=None):
        """
        Estimate A and k for the ask side (buy aggressor trades).

        Args:
            window_size: If None, returns (A, k) over the full timeline.
                         If given, returns (pd.Series_A, pd.Series_k) over rolling windows
                         anchored at each trade timestamp.
            buckets: Optional list of δ thresholds (e.g. [1, 3, 5, 10]).
                     Each threshold defines a bin upper bound; trades above the last
                     threshold are collected in a silent overflow bin (excluded from fit).
                     If None, one data point per integer δ is used.
        """
        if window_size is None:
            raw = _compute_buckets(self._tl, "a")
            agg = _aggregate_buckets(raw, buckets) if buckets is not None else raw
            return _fit(agg)
        return _compute_rolling_gueant(self._tl, "a", window_size, buckets)

    def bid(self, window_size=None, buckets=None):
        """
        Estimate A and k for the bid side (sell aggressor trades).

        Args:
            window_size: If None, returns (A, k) over the full timeline.
                         If given, returns (pd.Series_A, pd.Series_k) over rolling windows
                         anchored at each trade timestamp.
            buckets: Optional list of δ thresholds (e.g. [1, 3, 5, 10]).
                     Each threshold defines a bin upper bound; trades above the last
                     threshold are collected in a silent overflow bin (excluded from fit).
                     If None, one data point per integer δ is used.
        """
        if window_size is None:
            raw = _compute_buckets(self._tl, "b")
            agg = _aggregate_buckets(raw, buckets) if buckets is not None else raw
            return _fit(agg)
        return _compute_rolling_gueant(self._tl, "b", window_size, buckets)
