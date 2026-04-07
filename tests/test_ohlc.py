import pandas as pd
import pytest

from lobpy.tl import TL


class TestTLOHLC:
    """Test TL OHLC (Open, High, Low, Close) calculation."""

    def test_ohlc_empty_tl(self):
        """Test OHLC on empty TL returns empty DataFrame."""
        tl = TL(timestamp_unit="s")
        df = tl.ohlc("1s")

        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "count"]

    def test_ohlc_invalid_period(self):
        """Test OHLC with invalid period raises ValueError."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)

        with pytest.raises(ValueError, match="Unknown period"):
            tl.ohlc("2s")

    def test_ohlc_single_trade(self):
        """Test OHLC with single trade."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)

        df = tl.ohlc("1s")

        assert len(df) == 1
        candle = df.iloc[0]
        assert candle.name == 1000
        assert candle["open"] == 101.0
        assert candle["high"] == 101.0
        assert candle["low"] == 101.0
        assert candle["close"] == 101.0
        assert candle["volume"] == 0.5
        assert candle["count"] == 1

    def test_ohlc_multiple_trades_same_bucket(self):
        """Test OHLC with multiple trades in same time bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1500, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1800, side="b", price=99.5, volume=0.4)
        tl.add_trade(timestamp=1999, side="s", price=100.5, volume=0.2)

        df = tl.ohlc("1s")

        assert len(df) == 1
        candle = df.iloc[0]
        assert candle.name == 1000
        assert candle["open"] == 100.0
        assert candle["high"] == 101.0
        assert candle["low"] == 99.5
        assert candle["close"] == 100.5
        assert abs(candle["volume"] - 1.4) < 1e-10
        assert candle["count"] == 4

    def test_ohlc_multiple_buckets(self):
        """Test OHLC with trades in different time buckets."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="b", price=102.0, volume=0.4)
        tl.add_trade(timestamp=3000, side="b", price=104.0, volume=0.6)

        df = tl.ohlc("1s")

        assert len(df) == 3
        assert 1000 in df.index
        assert 2000 in df.index
        assert 3000 in df.index

        assert df.loc[1000, "count"] == 1
        assert df.loc[2000, "count"] == 1
        assert df.loc[3000, "count"] == 1

    def test_ohlc_1s_period(self):
        """Test OHLC with 1s period."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="b", price=102.0, volume=0.4)
        tl.add_trade(timestamp=3000, side="b", price=104.0, volume=0.6)

        df = tl.ohlc("1s")

        assert len(df) == 3
        assert 1000 in df.index
        assert 2000 in df.index
        assert 3000 in df.index

    def test_ohlc_15m_period(self):
        """Test OHLC with 15m period."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=0, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=900, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1800, side="b", price=102.0, volume=0.4)

        df = tl.ohlc("15m")

        assert len(df) == 3
        assert 0 in df.index
        assert 900 in df.index
        assert 1800 in df.index

    def test_ohlc_1h_period(self):
        """Test OHLC with 1h period."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=0, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=3600, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=7200, side="b", price=102.0, volume=0.4)

        df = tl.ohlc("1h")

        assert len(df) == 3
        assert 0 in df.index
        assert 3600 in df.index
        assert 7200 in df.index

    def test_ohlc_24h_period(self):
        """Test OHLC with 24h period."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=0, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=86400, side="s", price=101.0, volume=0.3)

        df = tl.ohlc("24h")

        assert len(df) == 2
        assert 0 in df.index
        assert 86400 in df.index

    def test_ohlc_volume_calculation(self):
        """Test OHLC volume is correctly summed."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1001, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1002, side="b", price=99.5, volume=0.4)
        tl.add_trade(timestamp=2000, side="s", price=102.0, volume=0.2)

        df = tl.ohlc("1s")

        assert df.loc[1000, "volume"] == 0.5
        assert df.loc[1001, "volume"] == 0.3
        assert df.loc[1002, "volume"] == 0.4
        assert df.loc[2000, "volume"] == 0.2

    def test_ohlc_count_calculation(self):
        """Test OHLC count is correctly calculated."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1001, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1002, side="b", price=99.5, volume=0.4)
        tl.add_trade(timestamp=2000, side="s", price=102.0, volume=0.2)

        df = tl.ohlc("1s")

        assert df.loc[1000, "count"] == 1
        assert df.loc[1001, "count"] == 1
        assert df.loc[1002, "count"] == 1
        assert df.loc[2000, "count"] == 1

    def test_ohlc_open_is_first_trade(self):
        """Test OHLC open is first trade price in bucket."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1001, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1002, side="b", price=99.5, volume=0.4)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["open"] == 101.0

    def test_ohlc_close_is_last_trade(self):
        """Test OHLC close is last trade price in bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1500, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1800, side="b", price=99.5, volume=0.4)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["close"] == 99.5

    def test_ohlc_high_is_max_price(self):
        """Test OHLC high is maximum price in bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1200, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1400, side="b", price=103.0, volume=0.4)
        tl.add_trade(timestamp=1800, side="s", price=102.0, volume=0.2)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["high"] == 103.0

    def test_ohlc_low_is_min_price(self):
        """Test OHLC low is minimum price in bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1200, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1400, side="b", price=98.0, volume=0.4)
        tl.add_trade(timestamp=1800, side="s", price=99.0, volume=0.2)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["low"] == 98.0

    def test_ohlc_timestamp_unit_ns(self):
        """Test OHLC with nanosecond timestamps."""
        tl = TL(timestamp_unit="ns")
        tl.add_trade(timestamp=1_000_000_000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1_000_000_001, side="s", price=101.0, volume=0.3)

        df = tl.ohlc("1s")

        assert len(df) == 1
        assert df.index[0] == 1_000_000_000

    def test_ohlc_timestamp_unit_ms(self):
        """Test OHLC with millisecond timestamps."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1001, side="s", price=101.0, volume=0.3)

        df = tl.ohlc("1s")

        assert len(df) == 1
        assert df.index[0] == 1000

    def test_ohlc_timestamp_unit_us(self):
        """Test OHLC with microsecond timestamps."""
        tl = TL(timestamp_unit="us")
        tl.add_trade(timestamp=1_000_000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1_000_001, side="s", price=101.0, volume=0.3)

        df = tl.ohlc("1s")

        assert len(df) == 1
        assert df.index[0] == 1_000_000

    def test_ohlc_trades_at_bucket_boundary(self):
        """Test OHLC with trades at exact bucket boundary."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=999, side="b", price=99.0, volume=0.5)
        tl.add_trade(timestamp=1000, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1999, side="b", price=101.0, volume=0.4)
        tl.add_trade(timestamp=2000, side="s", price=102.0, volume=0.2)

        df = tl.ohlc("1s")

        assert 999 in df.index
        assert 1000 in df.index
        assert 1999 in df.index
        assert 2000 in df.index

    def test_ohlc_unordered_trades(self):
        """Test OHLC with trades added in non-chronological order."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=2000, side="b", price=103.0, volume=0.4)
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=3000, side="s", price=104.0, volume=0.6)
        tl.add_trade(timestamp=2001, side="s", price=102.0, volume=0.2)

        df = tl.ohlc("1s")

        assert 1000 in df.index
        assert 2000 in df.index
        assert 2001 in df.index
        assert 3000 in df.index

        assert df.loc[1000, "open"] == 101.0
        assert df.loc[1000, "close"] == 101.0

        assert df.loc[2000, "open"] == 103.0
        assert df.loc[2000, "close"] == 103.0

        assert df.loc[2001, "open"] == 102.0
        assert df.loc[2001, "close"] == 102.0

    def test_ohlc_gaps_in_data(self):
        """Test OHLC with gaps in trade data."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=5000, side="s", price=101.0, volume=0.3)
        tl.add_trade(timestamp=10000, side="b", price=102.0, volume=0.4)

        df = tl.ohlc("1s")

        assert len(df) == 3
        assert 1000 in df.index
        assert 5000 in df.index
        assert 10000 in df.index

    def test_ohlc_all_accepted_periods(self):
        """Test that all accepted periods work."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)

        for period in ["1s", "5s", "1m", "15m", "1h", "24h"]:
            df = tl.ohlc(period)
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 1

    def test_ohlc_index_name(self):
        """Test OHLC DataFrame has correct index name."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)

        df = tl.ohlc("1s")

        assert df.index.name == "timestamp"

    def test_ohlc_dataframe_columns(self):
        """Test OHLC DataFrame has correct columns."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)

        df = tl.ohlc("1s")

        assert list(df.columns) == ["open", "high", "low", "close", "volume", "count"]

    def test_ohlc_large_volume_single_bucket(self):
        """Test OHLC with large volume in single bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=10.5)
        tl.add_trade(timestamp=1500, side="s", price=101.0, volume=20.3)
        tl.add_trade(timestamp=1800, side="b", price=99.5, volume=15.4)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert abs(candle["volume"] - 46.2) < 1e-10

    def test_ohlc_extreme_price_range(self):
        """Test OHLC with extreme price range in bucket."""
        tl = TL(timestamp_unit="ms")
        tl.add_trade(timestamp=1000, side="b", price=1.0, volume=0.5)
        tl.add_trade(timestamp=1500, side="s", price=10000.0, volume=0.3)
        tl.add_trade(timestamp=1800, side="b", price=5000.0, volume=0.4)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["open"] == 1.0
        assert candle["high"] == 10000.0
        assert candle["low"] == 1.0
        assert candle["close"] == 5000.0

    def test_ohlc_same_price_multiple_trades(self):
        """Test OHLC with same price for all trades in bucket."""
        tl = TL(timestamp_unit="s")
        tl.add_trade(timestamp=1000, side="b", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1001, side="s", price=100.0, volume=0.3)
        tl.add_trade(timestamp=1002, side="b", price=100.0, volume=0.4)

        df = tl.ohlc("1s")
        candle = df.iloc[0]

        assert candle["open"] == 100.0
        assert candle["high"] == 100.0
        assert candle["low"] == 100.0
        assert candle["close"] == 100.0
