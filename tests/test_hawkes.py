"""
Tests for Hawkes process fitting (hawkes.py) and TL.hawkes().
"""

import math

import numpy as np
import pandas as pd

from lobpy.hawkes import fit_hawkes
from lobpy.tl import TL

# ---------------------------------------------------------------------------
# fit_hawkes — unit tests on known synthetic data
# ---------------------------------------------------------------------------


class TestFitHawkes:
    def test_returns_dict_with_required_keys(self):
        rng = np.random.default_rng(42)
        ts = np.cumsum(rng.exponential(1.0, 50))
        result = fit_hawkes(ts)
        assert set(result) == {"mu", "alpha", "beta", "branching_ratio"}

    def test_nan_on_fewer_than_3_events(self):
        for n in (0, 1, 2):
            result = fit_hawkes(np.arange(n, dtype=float))
            assert all(math.isnan(v) for v in result.values())

    def test_branching_ratio_equals_alpha_over_beta(self):
        rng = np.random.default_rng(0)
        ts = np.cumsum(rng.exponential(0.5, 80))
        r = fit_hawkes(ts)
        if not math.isnan(r["alpha"]):
            assert abs(r["branching_ratio"] - r["alpha"] / r["beta"]) < 1e-9

    def test_branching_ratio_below_one(self):
        """Stationary Hawkes must have branching_ratio < 1."""
        rng = np.random.default_rng(7)
        ts = np.cumsum(rng.exponential(0.2, 100))
        r = fit_hawkes(ts)
        if not math.isnan(r["branching_ratio"]):
            assert r["branching_ratio"] < 1.0

    def test_positive_parameters(self):
        rng = np.random.default_rng(3)
        ts = np.cumsum(rng.exponential(0.3, 60))
        r = fit_hawkes(ts)
        for key in ("mu", "alpha", "beta"):
            if not math.isnan(r[key]):
                assert r[key] > 0.0

    def test_recovers_approximate_baseline(self):
        """For a near-Poisson process (very small alpha), mu ≈ true rate."""
        rng = np.random.default_rng(99)
        # Poisson with rate 5: very small clustering
        ts = np.cumsum(rng.exponential(1 / 5.0, 200))
        r = fit_hawkes(ts)
        if not math.isnan(r["mu"]):
            # mu should be in the ballpark of 5 (within 50%)
            assert 2.0 < r["mu"] < 10.0

    def test_timestamp_shift_invariance(self):
        """Adding a constant offset to timestamps should not change parameters."""
        rng = np.random.default_rng(11)
        ts = np.cumsum(rng.exponential(0.4, 60))
        r1 = fit_hawkes(ts)
        r2 = fit_hawkes(ts + 1_000_000)
        for key in ("mu", "alpha", "beta", "branching_ratio"):
            if not (math.isnan(r1[key]) or math.isnan(r2[key])):
                assert abs(r1[key] - r2[key]) / max(abs(r1[key]), 1e-9) < 0.05


# ---------------------------------------------------------------------------
# TL.hawkes — integration tests
# ---------------------------------------------------------------------------


def _make_tl_with_trades(timestamps, side="b"):
    tl = TL(name="test", timestamp_unit="s")
    tl.add_lob_snapshot(0, [(100, 10)], [(101, 10)])
    for ts in timestamps:
        tl.add_trade(int(ts), side, 101.0, 1.0)
    return tl


class TestTLHawkes:
    def test_returns_dict_for_scalar(self):
        rng = np.random.default_rng(5)
        ts = np.cumsum(rng.exponential(0.5, 50)).astype(int)
        tl = _make_tl_with_trades(ts)
        result = tl.hawkes()
        assert isinstance(result, dict)
        assert set(result) == {"mu", "alpha", "beta", "branching_ratio"}

    def test_nan_when_too_few_trades(self):
        tl = TL(timestamp_unit="s")
        tl.add_lob_snapshot(0, [(100, 10)], [(101, 10)])
        tl.add_trade(1, "b", 101.0, 1.0)
        tl.add_trade(2, "b", 101.0, 1.0)
        result = tl.hawkes()
        assert all(math.isnan(v) for v in result.values())

    def test_side_filter_buy(self):
        tl = TL(timestamp_unit="s")
        tl.add_lob_snapshot(0, [(100, 10)], [(101, 10)])
        rng = np.random.default_rng(1)
        for ts in np.cumsum(rng.exponential(0.3, 40)).astype(int):
            tl.add_trade(int(ts), "b", 101.0, 1.0)
        for ts in np.cumsum(rng.exponential(0.3, 40)).astype(int) + 200:
            tl.add_trade(int(ts), "s", 100.0, 1.0)

        r_all = tl.hawkes(side=None)
        r_buy = tl.hawkes(side="b")
        r_sell = tl.hawkes(side="s")
        # All three should return valid (non-nan) dicts
        for r in (r_all, r_buy, r_sell):
            assert not all(math.isnan(v) for v in r.values())

    def test_rolling_returns_dataframe(self):
        rng = np.random.default_rng(9)
        ts = np.cumsum(rng.exponential(0.5, 60)).astype(int)
        tl = _make_tl_with_trades(ts)
        df = tl.hawkes(window_size=20)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["mu", "alpha", "beta", "branching_ratio"]
        assert len(df) == 60

    def test_rolling_index_matches_trade_timestamps(self):
        rng = np.random.default_rng(13)
        ts = np.cumsum(rng.exponential(1.0, 30)).astype(int)
        tl = _make_tl_with_trades(ts)
        df = tl.hawkes(window_size=15)
        assert list(df.index) == [t.timestamp for t in tl.trades]

    def test_rolling_branching_ratio_below_one_where_valid(self):
        rng = np.random.default_rng(17)
        ts = np.cumsum(rng.exponential(0.4, 80)).astype(int)
        tl = _make_tl_with_trades(ts)
        df = tl.hawkes(window_size=30)
        valid = df["branching_ratio"].dropna()
        assert (valid < 1.0).all()

    def test_empty_tl_rolling_returns_empty_df(self):
        tl = TL(timestamp_unit="s")
        df = tl.hawkes(window_size=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_timestamp_unit_ns(self):
        """Parameters should be in events/second regardless of timestamp_unit."""
        rng = np.random.default_rng(21)
        # Integer inter-arrivals (1–4 s) so both TLs hold exactly the same events.
        ts_s = np.cumsum(rng.integers(1, 5, 60))

        tl_s = TL(timestamp_unit="s")
        tl_s.add_lob_snapshot(0, [(100, 10)], [(101, 10)])
        for ts in ts_s:
            tl_s.add_trade(int(ts), "b", 101.0, 1.0)

        tl_ns = TL(timestamp_unit="ns")
        tl_ns.add_lob_snapshot(0, [(100, 10)], [(101, 10)])
        for ts in ts_s * 1_000_000_000:
            tl_ns.add_trade(int(ts), "b", 101.0, 1.0)

        r_s = tl_s.hawkes()
        r_ns = tl_ns.hawkes()

        for key in ("mu", "alpha", "beta"):
            if not (math.isnan(r_s[key]) or math.isnan(r_ns[key])):
                ratio = r_s[key] / r_ns[key]
                assert 0.9 < ratio < 1.1, f"{key}: {r_s[key]} vs {r_ns[key]}"
