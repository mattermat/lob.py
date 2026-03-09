"""
Guéant intensity function estimation.

Model: λ(δ) = A * exp(-k * δ)
Reference: Guéant, Lehalle, Fernandez-Tapia (2013)
# TODO: add link to the paper
"""

import bisect
from collections import defaultdict

import numpy as np
import pandas as pd


def _lob_at(lob_timestamps, lobts, ts):
    """Return the last LOB snapshot at or before ts, or None."""
    idx = bisect.bisect_right(lob_timestamps, ts) - 1
    if idx < 0:
        return None
    return lobts[lob_timestamps[idx]]


def _compute_buckets(tl, side):
    """
    Compute empirical intensity buckets λ̂(δ) = N(δ) / T(δ) for one side.

    For each integer tick distance δ:
      N(δ) = number of trades that occurred at distance δ
      T(δ) = total time the book exposed liquidity at distance δ

    Distance conventions (matching the reference):
      ask side (buy aggressor, side='a'):
        trade:  δ = (trade_price - best_bid) / tick_size
        book:   δ = (ask_level  - best_bid) / tick_size
      bid side (sell aggressor, side='b'):
        trade:  δ = (best_ask - trade_price) / tick_size
        book:   δ = (best_ask - bid_level)   / tick_size

    Args:
        tl:   TL instance
        side: 'a' (ask side) or 'b' (bid side)

    Returns:
        pd.DataFrame with columns [delta, N, T, lambda]
    """
    tick_size = tl.tick_size
    lob_ts = list(tl._lobts.timestamps)

    if not lob_ts:
        return pd.DataFrame(columns=["delta", "N", "T", "lambda"])

    trade_side = "b" if side == "a" else "s"
    trades = [t for t in tl.trades if t.side == trade_side]

    # --- N(δ) ---
    N = defaultdict(int)
    for trade in trades:
        lob = _lob_at(lob_ts, tl._lobts, trade.timestamp)
        if lob is None:
            continue
        best_bid = lob.bid[0]
        best_ask = lob.ask[0]
        if best_bid <= 0 or best_ask <= 0:
            continue
        if side == "a":
            raw = (trade.price - best_bid) / tick_size
        else:
            raw = (best_ask - trade.price) / tick_size
        d = int(round(raw))
        if d >= 0:
            N[d] += 1

    # --- T(δ) ---
    T = defaultdict(float)
    end_ts = max(
        lob_ts[-1],
        max((t.timestamp for t in tl.trades), default=lob_ts[-1]),
    )
    for i, ts in enumerate(lob_ts):
        lob = tl._lobts[ts]
        next_ts = lob_ts[i + 1] if i + 1 < len(lob_ts) else end_ts
        duration = next_ts - ts
        if duration <= 0:
            continue
        best_bid = lob.bid[0]
        best_ask = lob.ask[0]
        if best_bid <= 0 or best_ask <= 0:
            continue
        if side == "a":
            for price in lob._asks:
                d = int(round((price - best_bid) / tick_size))
                if d >= 0:
                    T[d] += duration
        else:
            for price in lob._bids:
                d = int(round((best_ask - price) / tick_size))
                if d >= 0:
                    T[d] += duration

    # --- assemble DataFrame ---
    all_deltas = sorted(set(N) | set(T))
    rows = [
        (d, N.get(d, 0), T.get(d, 0.0),
         N[d] / T[d] if T.get(d, 0.0) > 0 and N.get(d, 0) > 0 else float("nan"))
        for d in all_deltas
    ]
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
                         If given, returns (pd.Series_A, pd.Series_k) over rolling windows.
            buckets: Optional list of δ thresholds (e.g. [1, 3, 5, 10]).
                     Each threshold defines a bin upper bound; trades above the last
                     threshold are collected in a silent overflow bin (excluded from fit).
                     If None, one data point per integer δ is used.
        """
        def _compute(tl):
            raw = _compute_buckets(tl, "a")
            return _aggregate_buckets(raw, buckets) if buckets is not None else raw

        if window_size is None:
            return _fit(_compute(self._tl))
        A_vals, k_vals = {}, {}
        for ts, window in self._tl._rolling_items(window_size):
            A_vals[ts], k_vals[ts] = _fit(_compute(window))
        return (
            pd.Series(A_vals, name="gueant_A_ask"),
            pd.Series(k_vals, name="gueant_k_ask"),
        )

    def bid(self, window_size=None, buckets=None):
        """
        Estimate A and k for the bid side (sell aggressor trades).

        Args:
            window_size: If None, returns (A, k) over the full timeline.
                         If given, returns (pd.Series_A, pd.Series_k) over rolling windows.
            buckets: Optional list of δ thresholds (e.g. [1, 3, 5, 10]).
                     Each threshold defines a bin upper bound; trades above the last
                     threshold are collected in a silent overflow bin (excluded from fit).
                     If None, one data point per integer δ is used.
        """
        def _compute(tl):
            raw = _compute_buckets(tl, "b")
            return _aggregate_buckets(raw, buckets) if buckets is not None else raw

        if window_size is None:
            return _fit(_compute(self._tl))
        A_vals, k_vals = {}, {}
        for ts, window in self._tl._rolling_items(window_size):
            A_vals[ts], k_vals[ts] = _fit(_compute(window))
        return (
            pd.Series(A_vals, name="gueant_A_bid"),
            pd.Series(k_vals, name="gueant_k_bid"),
        )
