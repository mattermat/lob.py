import pandas as pd

from lobpy.tl import TL


class TestTLVpin:
    """Test TL.vpin() method."""

    def test_vpin_empty_tl(self):
        """Test vpin with empty TL returns NaN."""
        tl = TL()
        vpin = tl.vpin()
        assert pd.isna(vpin)

    def test_vpin_single_trade(self):
        """Test vpin with single trade."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        vpin = tl.vpin()
        assert isinstance(vpin, float)
        assert not pd.isna(vpin)

    def test_vpin_returns_scalar(self):
        """Test vpin without window_size returns scalar."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)
        vpin = tl.vpin()
        assert isinstance(vpin, float)
        assert not isinstance(vpin, pd.Series)

    def test_vpin_rolling_returns_series(self):
        """Test vpin with window_size returns pd.Series."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.0)
        vpin_ts = tl.vpin(100)
        assert isinstance(vpin_ts, pd.Series)
        assert vpin_ts.name == "vpin"

    def test_vpin_balanced_trades(self):
        """Test vpin with alternating buy and sell trades."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1300, side="s", price=100.0, volume=1.0)

        vpin = tl.vpin(bucket_size=1.0)
        assert vpin == 1.0

    def test_vpin_imbalanced_trades(self):
        """Test vpin with highly imbalanced buy and sell volumes."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1200, side="s", price=100.0, volume=0.2)
        tl.add_trade(timestamp=1300, side="s", price=100.0, volume=0.2)

        vpin = tl.vpin(bucket_size=1.0)
        assert vpin > 0.0

    def test_vpin_calculation_formula(self):
        """Test vpin calculation matches formula: sum|buy-sell| / (n * bucket_size)."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.5)
        tl.add_trade(timestamp=1300, side="s", price=100.0, volume=1.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        expected_vpin = 0.75
        assert abs(vpin - expected_vpin) < 1e-10

    def test_vpin_trade_splitting_across_buckets(self):
        """Test vpin handles trade splitting across bucket boundaries."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.5)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        expected_vpin = 0.5
        assert abs(vpin - expected_vpin) < 1e-10

    def test_vpin_custom_bucket_size(self):
        """Test vpin with custom bucket_size."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=2.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)

        vpin_small_bucket = tl.vpin(bucket_size=0.5)
        vpin_large_bucket = tl.vpin(bucket_size=2.0)

        assert isinstance(vpin_small_bucket, float)
        assert isinstance(vpin_large_bucket, float)

    def test_vpin_default_bucket_size(self):
        """Test vpin uses default bucket_size (total_volume / 50) when not specified."""
        tl = TL()
        for i in range(10):
            tl.add_trade(
                timestamp=1000 + i * 100, side="b" if i % 2 == 0 else "s", price=100.0, volume=1.0
            )

        vpin_default = tl.vpin()
        total_vol = sum(t.volume for t in tl.trades)
        default_bucket_size = total_vol / 50
        vpin_explicit = tl.vpin(bucket_size=default_bucket_size)

        assert abs(vpin_default - vpin_explicit) < 1e-10

    def test_vpin_rolling_window_size(self):
        """Test vpin rolling windows have correct size."""
        tl = TL()
        for i in range(10):
            tl.add_trade(
                timestamp=1000 + i * 100, side="b" if i % 2 == 0 else "s", price=100.0, volume=1.0
            )

        window_size = 300
        vpin_ts = tl.vpin(window_size=window_size)

        assert len(vpin_ts) == len(tl.timestamps)
        for ts in vpin_ts.index:
            assert ts in tl.timestamps

    def test_vpin_rolling_values(self):
        """Test vpin rolling values are computed correctly for each window."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1300, side="s", price=100.0, volume=1.0)

        window_size = 200
        vpin_ts = tl.vpin(window_size=window_size, bucket_size=1.0)

        assert len(vpin_ts) == 4
        assert all(isinstance(v, float) for v in vpin_ts.values)

    def test_vpin_rolling_empty_window(self):
        """Test vpin rolling with windows containing no trades."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)

        window_size = 100
        vpin_ts = tl.vpin(window_size=window_size)

        assert len(vpin_ts) == 2

    def test_vpin_rolling_with_custom_bucket_size(self):
        """Test vpin rolling with custom bucket_size."""
        tl = TL()
        for i in range(5):
            tl.add_trade(
                timestamp=1000 + i * 100, side="b" if i % 2 == 0 else "s", price=100.0, volume=1.0
            )

        window_size = 250
        bucket_size = 1.0
        vpin_ts = tl.vpin(window_size=window_size, bucket_size=bucket_size)

        assert isinstance(vpin_ts, pd.Series)
        assert len(vpin_ts) == 5

    def test_vpin_all_buy_trades(self):
        """Test vpin with all buy trades (maximum imbalance)."""
        tl = TL()
        for i in range(5):
            tl.add_trade(timestamp=1000 + i * 100, side="b", price=100.0, volume=1.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_all_sell_trades(self):
        """Test vpin with all sell trades (maximum imbalance)."""
        tl = TL()
        for i in range(5):
            tl.add_trade(timestamp=1000 + i * 100, side="s", price=100.0, volume=1.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_large_imbalance_scenario(self):
        """Test vpin with large realistic imbalance scenario."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=10.0)
        tl.add_trade(timestamp=1100, side="b", price=100.0, volume=8.0)
        tl.add_trade(timestamp=1200, side="s", price=100.0, volume=2.0)
        tl.add_trade(timestamp=1300, side="s", price=100.0, volume=3.0)

        bucket_size = 5.0
        vpin = tl.vpin(bucket_size=bucket_size)

        expected_vpin = 0.8
        assert abs(vpin - expected_vpin) < 1e-10

    def test_vpin_trades_sorted_by_timestamp(self):
        """Test vpin processes trades in timestamp order."""
        tl = TL()
        tl.add_trade(timestamp=1100, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_zero_volume_trades(self):
        """Test vpin handles zero volume trades correctly."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.0)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_very_large_bucket_size(self):
        """Test vpin with bucket_size larger than total volume."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)

        bucket_size = 10.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert pd.isna(vpin)

    def test_vpin_very_small_bucket_size(self):
        """Test vpin with very small bucket_size."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)

        bucket_size = 0.1
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_single_large_trade_split(self):
        """Test vpin with single large trade split across buckets."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=5.0)

        bucket_size = 1.0
        vpin = tl.vpin(bucket_size=bucket_size)

        assert vpin == 1.0

    def test_vpin_rolling_varying_bucket_sizes(self):
        """Test that rolling vpin uses consistent bucket_size across windows."""
        tl = TL()
        for i in range(10):
            tl.add_trade(
                timestamp=1000 + i * 100, side="b" if i % 2 == 0 else "s", price=100.0, volume=1.0
            )

        window_size = 300
        bucket_size = 1.0
        vpin_ts = tl.vpin(window_size=window_size, bucket_size=bucket_size)

        assert isinstance(vpin_ts, pd.Series)
        assert len(vpin_ts) == 10
        assert all(v >= 0 for v in vpin_ts.values)
        assert all(v <= 1 for v in vpin_ts.values)

    def test_vpin_volume_buckets_consistency(self):
        """Test that vpin is consistent with volume_buckets output."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1200, side="b", price=100.0, volume=1.5)

        bucket_size = 1.0
        buckets = tl.volume_buckets(bucket_size=bucket_size)
        vpin = tl.vpin(bucket_size=bucket_size)

        n = len(buckets)
        imbalance = (buckets["buy_volume"] - buckets["sell_volume"]).abs().sum()
        manual_vpin = imbalance / (n * bucket_size)

        assert abs(vpin - manual_vpin) < 1e-10

    def test_vpin_different_timestamps_same_volume(self):
        """Test vpin with same total volume distributed differently over time."""
        tl1 = TL()
        tl1.add_trade(timestamp=1000, side="b", price=100.0, volume=2.0)
        tl1.add_trade(timestamp=1100, side="s", price=100.0, volume=2.0)

        tl2 = TL()
        for i in range(4):
            tl2.add_trade(
                timestamp=1000 + i * 100, side="b" if i % 2 == 0 else "s", price=100.0, volume=1.0
            )

        bucket_size = 1.0
        vpin1 = tl1.vpin(bucket_size=bucket_size)
        vpin2 = tl2.vpin(bucket_size=bucket_size)

        assert vpin1 == vpin2

    def test_vpin_with_lob_only(self):
        """Test vpin with only LOB data, no trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.5, 2.0)], asks=[(101.5, 1.5)])

        vpin = tl.vpin()
        assert pd.isna(vpin)

    def test_vpin_rolling_with_partial_window(self):
        """Test rolling vpin where some windows have no trades."""
        tl = TL()
        tl.add_trade(timestamp=2000, side="b", price=100.0, volume=1.0)
        tl.add_trade(timestamp=5000, side="s", price=100.0, volume=1.0)

        window_size = 1000
        vpin_ts = tl.vpin(window_size=window_size, bucket_size=1.0)

        assert len(vpin_ts) == 2
        assert 2000 in vpin_ts.index
        assert 5000 in vpin_ts.index
