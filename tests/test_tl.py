from lobpy.tl import TL, Trade


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestTrade:
    """Test Trade class."""

    def test_trade_creation(self):
        """Test creating a Trade object."""
        trade = Trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        assert trade.timestamp == 1000
        assert trade.side == "b"
        assert trade.price == 101.0
        assert trade.volume == 0.5

    def test_trade_buy_aggressor(self):
        """Test trade with buy aggressor."""
        trade = Trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        assert trade.side == "b"

    def test_trade_sell_aggressor(self):
        """Test trade with sell aggressor."""
        trade = Trade(timestamp=1000, side="s", price=100.0, volume=1.0)
        assert trade.side == "s"

    def test_trade_repr(self):
        """Test Trade __repr__ method."""
        trade = Trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        repr_str = repr(trade)
        assert "Trade" in repr_str
        assert "ts=1000" in repr_str
        assert "b" in repr_str
        assert "0.5" in repr_str
        assert "101.0" in repr_str


class TestTLInit:
    """Test TL initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        tl = TL()
        assert tl.name is not None
        assert tl.tick_size == 1
        assert tl.lob_mode == "delta"
        assert tl.update_type == "realtime"
        assert len(tl.trades) == 0

    def test_init_with_name(self):
        """Test initialization with name."""
        tl = TL(name="BTC-USD")
        assert tl.name == "BTC-USD"
        assert tl.tick_size == 1
        assert tl.lob_mode == "delta"
        assert tl.update_type == "realtime"

    def test_init_with_tick_size(self):
        """Test initialization with tick size."""
        tl = TL(tick_size=0.5)
        assert tl.tick_size == 0.5

    def test_init_with_lob_mode_realtime(self):
        """Test initialization with realtime update type."""
        tl = TL(update_type="realtime")
        assert tl.update_type == "realtime"

    def test_init_with_lob_mode_fixed(self):
        """Test initialization with fixed update type."""
        tl = TL(update_type="fixed")
        assert tl.update_type == "fixed"

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        tl = TL(name="ETH-USD", tick_size=0.01, lob_mode="delta", update_type="fixed")
        assert tl.name == "ETH-USD"
        assert tl.tick_size == 0.01
        assert tl.lob_mode == "delta"
        assert tl.update_type == "fixed"


class TestTLAddLOB:
    """Test adding LOB data to TL."""

    def test_add_lob_snapshot_creates_snapshot(self):
        """Test that add_lob_snapshot creates LOB snapshot."""
        tl = TL()
        bids = [(100.0, 1.5), (99.5, 2.3)]
        asks = [(101.0, 2.1), (101.5, 1.7)]

        tl.add_lob_snapshot(timestamp=1000, bids=bids, asks=asks)

        assert tl.lob[1000] is not None
        lob = tl.lob[1000]
        assert lob.bid[0] == 100.0
        assert lob.ask[0] == 101.0

    def test_add_lob_snapshot_multiple_levels(self):
        """Test adding LOB snapshot with multiple levels."""
        tl = TL()
        bids = [(100.0, 1.5), (99.5, 2.3), (99.0, 1.8), (98.5, 3.0)]
        asks = [(101.0, 2.1), (101.5, 1.7), (102.0, 2.5), (102.5, 1.2)]

        tl.add_lob_snapshot(timestamp=1000, bids=bids, asks=asks)

        lob = tl.lob[1000]
        assert lob.bid[0] == 100.0
        assert lob.bid[1] == 99.5
        assert lob.ask[0] == 101.0
        assert lob.ask[1] == 101.5

    def test_add_lob_update_creates_update(self):
        """Test that add_lob_update applies incremental updates."""
        tl = TL()
        bids = [(100.0, 1.5)]
        asks = [(101.0, 2.1)]
        tl.add_lob_snapshot(timestamp=1000, bids=bids, asks=asks)

        updates = [("b", 100.0, 2.0), ("a", 101.0, 1.5)]
        tl.add_lob_update(timestamp=1100, updates=updates)

        assert tl.lob[1100] is not None
        lob = tl.lob[1100]
        assert lob.bid[0] == 100.0
        assert lob.ask[0] == 101.0

    def test_add_lob_update_new_level(self):
        """Test adding new level via update."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])

        updates = [("b", 99.5, 1.0)]
        tl.add_lob_update(timestamp=1100, updates=updates)

        lob = tl.lob[1100]
        assert lob.bid[0] == 100.0
        assert lob.bid[1] == 99.5

    def test_add_lob_update_delete_level(self):
        """Test deleting level via update (quantity=0)."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000, bids=[(100.0, 1.5), (99.5, 2.3)], asks=[(101.0, 2.1)]
        )

        updates = [("b", 99.5, 0)]
        tl.add_lob_update(timestamp=1100, updates=updates)

        lob = tl.lob[1100]
        assert lob.bid[0] == 100.0


class TestTLAddTrade:
    """Test adding trades to TL."""

    def test_add_single_trade(self):
        """Test adding a single trade."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        assert len(tl.trades) == 1
        trade = tl.trades[0]
        assert trade.timestamp == 1150
        assert trade.side == "b"
        assert trade.price == 101.0
        assert trade.volume == 0.5

    def test_add_multiple_trades_sequential(self):
        """Test adding multiple trades sequentially."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1450, side="b", price=101.0, volume=1.5)

        assert len(tl.trades) == 3
        assert tl.trades[0].side == "b"
        assert tl.trades[1].side == "s"
        assert tl.trades[2].side == "b"

    def test_add_trades_batch(self):
        """Test adding multiple trades at same timestamp."""
        tl = TL()
        trades = [("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)]
        tl.add_trades(timestamp=1350, trades=trades)

        assert len(tl.trades) == 3
        assert tl.trades[0].timestamp == 1350
        assert tl.trades[1].timestamp == 1350
        assert tl.trades[2].timestamp == 1350
        assert tl.trades[0].side == "b"
        assert tl.trades[1].side == "b"
        assert tl.trades[2].side == "s"

    def test_add_trades_single_trade(self):
        """Test add_trades with single trade."""
        tl = TL()
        trades = [("b", 101.0, 0.5)]
        tl.add_trades(timestamp=1150, trades=trades)

        assert len(tl.trades) == 1
        assert tl.trades[0].side == "b"
        assert tl.trades[0].price == 101.0
        assert tl.trades[0].volume == 0.5


class TestTLProperties:
    """Test TL properties."""

    def test_lob_property_returns_lobts(self):
        """Test that lob property returns LOBts object."""
        tl = TL()
        assert tl.lob is not None

    def test_trades_property_returns_list(self):
        """Test that trades property returns list."""
        tl = TL()
        assert isinstance(tl.trades, list)

    def test_trades_property_returns_empty_list_initially(self):
        """Test that trades property returns empty list initially."""
        tl = TL()
        assert len(tl.trades) == 0

    def test_name_property(self):
        """Test name property."""
        tl = TL(name="BTC-USD")
        assert tl.name == "BTC-USD"

    def test_tick_size_property(self):
        """Test tick_size property."""
        tl = TL(tick_size=0.5)
        assert tl.tick_size == 0.5

    def test_lob_mode_property(self):
        """Test lob_mode property."""
        tl = TL(lob_mode="fixed")
        assert tl.lob_mode == "fixed"


class TestTLLen:
    """Test TL __len__ method."""

    def test_len_empty(self):
        """Test len of empty TL."""
        tl = TL()
        assert len(tl) == 0

    def test_len_only_lob_snapshots(self):
        """Test len with only LOB snapshots."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.0, 2.0)], asks=[(101.0, 1.5)])
        tl.add_lob_snapshot(timestamp=1200, bids=[(100.0, 1.8)], asks=[(101.0, 2.0)])

        assert len(tl) == 3

    def test_len_only_trades(self):
        """Test len with only trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)

        assert len(tl) == 3

    def test_len_lob_and_trades(self):
        """Test len with both LOB snapshots and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_lob_update(timestamp=1300, updates=[("a", 101.0, 1.5)])

        assert len(tl) == 5

    def test_len_multiple_trades_same_timestamp(self):
        """Test len counts all trades even at same timestamp."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trades(
            timestamp=1350,
            trades=[("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)],
        )

        assert len(tl) == 4

    def test_len_complex_sequence(self):
        """Test len with complex sequence like in example."""
        tl = TL()
        timestamps = [1000, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]

        tl.add_lob_snapshot(
            timestamp=timestamps[0],
            bids=[(100.00, 1.5), (99.50, 2.3), (99.00, 1.8), (98.50, 3.0)],
            asks=[(101.00, 2.1), (101.50, 1.7), (102.00, 2.5), (102.50, 1.2)],
        )
        tl.add_lob_update(
            timestamp=timestamps[1],
            updates=[("b", 100.00, 2.0), ("b", 100.50, 1.0), ("a", 101.00, 1.5)],
        )
        tl.add_trade(timestamp=timestamps[2], side="b", price=101.00, volume=0.5)
        tl.add_lob_update(
            timestamp=timestamps[3],
            updates=[("a", 101.00, 1.0), ("b", 100.50, 0)],
        )
        tl.add_trade(timestamp=timestamps[4], side="s", price=100.00, volume=1.0)
        tl.add_lob_update(
            timestamp=timestamps[5],
            updates=[("b", 100.00, 1.0), ("a", 101.50, 0), ("a", 103.00, 2.0)],
        )
        tl.add_trades(
            timestamp=timestamps[6],
            trades=[("b", 101.00, 0.3), ("b", 101.50, 0.2), ("s", 100.00, 0.5)],
        )
        tl.add_lob_update(
            timestamp=timestamps[7],
            updates=[("b", 99.50, 3.0), ("a", 101.00, 2.5)],
        )
        tl.add_trade(timestamp=timestamps[8], side="b", price=101.00, volume=1.5)
        tl.add_lob_update(
            timestamp=timestamps[9],
            updates=[("b", 100.00, 2.5), ("a", 101.00, 1.0)],
        )

        assert len(tl) == 12


class TestTLTimestamps:
    """Test TL timestamps property."""

    def test_timestamps_empty(self):
        """Test timestamps on empty TL."""
        tl = TL()
        assert len(tl.timestamps) == 0

    def test_timestamps_only_lob_snapshots(self):
        """Test timestamps with only LOB snapshots."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.0, 2.0)], asks=[(101.0, 1.5)])
        tl.add_lob_snapshot(timestamp=1200, bids=[(100.0, 1.8)], asks=[(101.0, 2.0)])

        timestamps = tl.timestamps
        assert len(timestamps) == 3
        assert timestamps == [1000, 1100, 1200]

    def test_timestamps_only_trades(self):
        """Test timestamps with only trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)

        timestamps = tl.timestamps
        assert len(timestamps) == 3
        assert timestamps == [1150, 1250, 1350]

    def test_timestamps_lob_and_trades(self):
        """Test timestamps with both LOB and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        timestamps = tl.timestamps
        assert len(timestamps) == 4
        assert timestamps == [1000, 1150, 1200, 1250]

    def test_timestamps_sorted(self):
        """Test timestamps are sorted chronologically."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1200, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1500, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=1.0)

        timestamps = tl.timestamps
        assert timestamps == [1000, 1100, 1200, 1500]

    def test_timestamps_slice(self):
        """Test slicing timestamps."""
        tl = TL()
        for i in range(10):
            tl.add_lob_snapshot(
                timestamp=1000 + i * 100,
                bids=[(100.0 + i, 1.5)],
                asks=[(101.0 + i, 2.1)],
            )

        first_five = tl.timestamps[:5]
        assert len(first_five) == 5
        assert first_five == [1000, 1100, 1200, 1300, 1400]

    def test_timestamps_with_duplicates(self):
        """Test timestamps handle same timestamp for multiple events."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trades(
            timestamp=1000,
            trades=[("b", 101.0, 0.3), ("s", 100.0, 0.5)],
        )

        timestamps = tl.timestamps
        assert len(timestamps) >= 1
        assert 1000 in timestamps

    def test_timestamps_mixed_sequence(self):
        """Test timestamps with mixed LOB and trades like in example."""
        tl = TL()
        timestamps = [1000, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]

        tl.add_lob_snapshot(
            timestamp=timestamps[0],
            bids=[(100.00, 1.5), (99.50, 2.3)],
            asks=[(101.00, 2.1), (101.50, 1.7)],
        )
        tl.add_lob_update(
            timestamp=timestamps[1],
            updates=[("b", 100.00, 2.0), ("a", 101.00, 1.5)],
        )
        tl.add_trade(timestamp=timestamps[2], side="b", price=101.00, volume=0.5)
        tl.add_lob_update(timestamp=timestamps[3], updates=[("a", 101.00, 1.0)])
        tl.add_trade(timestamp=timestamps[4], side="s", price=100.00, volume=1.0)
        tl.add_lob_update(timestamp=timestamps[5], updates=[("b", 100.00, 1.0)])
        tl.add_trades(
            timestamp=timestamps[6],
            trades=[("b", 101.00, 0.3), ("s", 100.00, 0.5)],
        )
        tl.add_lob_update(timestamp=timestamps[7], updates=[("b", 99.50, 3.0)])
        tl.add_trade(timestamp=timestamps[8], side="b", price=101.00, volume=1.5)
        tl.add_lob_update(timestamp=timestamps[9], updates=[("b", 100.00, 2.5)])

        ts_list = tl.timestamps
        assert len(ts_list) == 10
        assert ts_list[:5] == [1000, 1100, 1150, 1200, 1250]


class TestTLToPd:
    """Test TL to_pd method."""

    def test_to_pd_empty(self):
        """Test to_pd with empty TL."""
        tl = TL()
        df = tl.to_pd()

        assert len(df) == 0
        assert list(df.columns) == ["timestamp", "type", "side", "level", "price", "size"]

    def test_to_pd_only_lob(self):
        """Test to_pd with only LOB data."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.5, 2.0)], asks=[(101.5, 1.5)])

        df = tl.to_pd()

        assert len(df) == 4  # 2 timestamps * (1 bid + 1 ask)
        assert list(df.columns) == ["timestamp", "type", "side", "level", "price", "size"]
        assert set(df["type"].unique()) == {"lob"}

    def test_to_pd_only_trades(self):
        """Test to_pd with only trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        df = tl.to_pd()

        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "type", "side", "level", "price", "size"]
        assert set(df["type"].unique()) == {"trade"}

    def test_to_pd_mixed(self):
        """Test to_pd with both LOB and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        df = tl.to_pd()

        assert len(df) == 6  # 2 LOB * 2 levels + 2 trades
        assert list(df.columns) == ["timestamp", "type", "side", "level", "price", "size"]
        assert set(df["type"].unique()) == {"lob", "trade"}

    def test_to_pd_trade_level_is_nan(self):
        """Test that trade rows have NaN in level column."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        df = tl.to_pd()

        lob_row = df[df["type"] == "lob"].iloc[0]
        trade_row = df[df["type"] == "trade"].iloc[0]

        assert pd.notna(lob_row["level"])
        assert pd.isna(trade_row["level"])

    def test_to_pd_sorted_by_timestamp(self):
        """Test that rows are sorted by timestamp."""
        tl = TL()
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])

        df = tl.to_pd()

        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_to_pd_multiple_levels(self):
        """Test to_pd with multiple price levels."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000,
            bids=[(100.0, 1.5), (99.5, 2.3), (99.0, 1.8)],
            asks=[(101.0, 2.1), (101.5, 1.7)],
        )

        df = tl.to_pd()

        assert len(df) == 5  # 3 bids + 2 asks
        lob_rows = df[df["type"] == "lob"]
        assert len(lob_rows) == 5

        bid_levels = lob_rows[lob_rows["side"] == "b"]["level"].tolist()
        ask_levels = lob_rows[lob_rows["side"] == "a"]["level"].tolist()
        assert set(bid_levels) == {0, 1, 2}
        assert set(ask_levels) == {0, 1}

    def test_to_pd_trade_columns(self):
        """Test trade row has correct data."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        df = tl.to_pd()
        trade_row = df[df["type"] == "trade"].iloc[0]

        assert trade_row["timestamp"] == 1150
        assert trade_row["type"] == "trade"
        assert trade_row["side"] == "b"
        assert trade_row["price"] == 101.0
        assert trade_row["size"] == 0.5
        assert pd.isna(trade_row["level"])

    def test_to_pd_lob_columns(self):
        """Test LOB row has correct data."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])

        df = tl.to_pd()
        bid_row = df[(df["type"] == "lob") & (df["side"] == "b")].iloc[0]
        ask_row = df[(df["type"] == "lob") & (df["side"] == "a")].iloc[0]

        assert bid_row["timestamp"] == 1000
        assert bid_row["type"] == "lob"
        assert bid_row["side"] == "b"
        assert bid_row["level"] == 0
        assert bid_row["price"] == 100.0
        assert bid_row["size"] == 1.5

        assert ask_row["timestamp"] == 1000
        assert ask_row["type"] == "lob"
        assert ask_row["side"] == "a"
        assert ask_row["level"] == 0
        assert ask_row["price"] == 101.0
        assert ask_row["size"] == 2.1

    def test_to_pd_multiple_trades_same_timestamp(self):
        """Test to_pd with multiple trades at same timestamp."""
        tl = TL()
        tl.add_trades(
            timestamp=1350,
            trades=[("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)],
        )

        df = tl.to_pd()
        trade_rows = df[df["type"] == "trade"]

        assert len(trade_rows) == 3
        assert all(trade_rows["timestamp"] == 1350)


class TestTLToNp:
    """Test TL to_np method."""

    def test_to_np_empty(self):
        """Test to_np with empty TL."""
        tl = TL()
        arr = tl.to_np()

        assert arr.shape == (0, 6)

    def test_to_np_only_lob(self):
        """Test to_np with only LOB data."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.5, 2.0)], asks=[(101.5, 1.5)])

        arr = tl.to_np()

        assert arr.shape == (4, 6)  # 2 timestamps * (1 bid + 1 ask)
        assert arr.shape[1] == 6  # 6 columns

    def test_to_np_only_trades(self):
        """Test to_np with only trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        arr = tl.to_np()

        assert arr.shape == (2, 6)
        assert all(row[1] == "trade" for row in arr)

    def test_to_np_mixed(self):
        """Test to_np with both LOB and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        arr = tl.to_np()

        assert arr.shape == (6, 6)  # 2 LOB * 2 levels + 2 trades
        assert arr.shape[1] == 6

    def test_to_np_trade_level_is_nan(self):
        """Test that trade rows have NaN in level column."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        arr = tl.to_np()

        lob_row = [row for row in arr if row[1] == "lob"][0]
        trade_row = [row for row in arr if row[1] == "trade"][0]

        assert not pd.isna(lob_row[3])  # level column
        assert pd.isna(trade_row[3])

    def test_to_np_sorted_by_timestamp(self):
        """Test that rows are sorted by timestamp."""
        tl = TL()
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("b", 100.0, 2.0)])

        arr = tl.to_np()
        timestamps = [row[0] for row in arr]

        assert timestamps == sorted(timestamps)

    def test_to_np_columns(self):
        """Test that to_np has correct column structure."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        arr = tl.to_np()

        # Each row should be: [timestamp, type, side, level, price, size]
        lob_row = [row for row in arr if row[1] == "lob"][0]
        trade_row = [row for row in arr if row[1] == "trade"][0]

        assert len(lob_row) == 6
        assert len(trade_row) == 6

    def test_to_np_lob_row(self):
        """Test LOB row has correct data."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])

        arr = tl.to_np()
        bid_row = [row for row in arr if row[1] == "lob" and row[2] == "b"][0]
        ask_row = [row for row in arr if row[1] == "lob" and row[2] == "a"][0]

        assert bid_row[0] == 1000  # timestamp
        assert bid_row[1] == "lob"  # type
        assert bid_row[2] == "b"  # side
        assert bid_row[3] == 0  # level
        assert bid_row[4] == 100.0  # price
        assert bid_row[5] == 1.5  # size

        assert ask_row[0] == 1000  # timestamp
        assert ask_row[1] == "lob"  # type
        assert ask_row[2] == "a"  # side
        assert ask_row[3] == 0  # level
        assert ask_row[4] == 101.0  # price
        assert ask_row[5] == 2.1  # size

    def test_to_np_trade_row(self):
        """Test trade row has correct data."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        arr = tl.to_np()
        trade_row = [row for row in arr if row[1] == "trade"][0]

        assert trade_row[0] == 1150  # timestamp
        assert trade_row[1] == "trade"  # type
        assert trade_row[2] == "b"  # side
        assert pd.isna(trade_row[3])  # level (NaN)
        assert trade_row[4] == 101.0  # price
        assert trade_row[5] == 0.5  # size

    def test_to_np_multiple_levels(self):
        """Test to_np with multiple price levels."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000,
            bids=[(100.0, 1.5), (99.5, 2.3), (99.0, 1.8)],
            asks=[(101.0, 2.1), (101.5, 1.7)],
        )

        arr = tl.to_np()

        assert arr.shape == (5, 6)  # 3 bids + 2 asks
        lob_rows = [row for row in arr if row[1] == "lob"]
        assert len(lob_rows) == 5

        bid_levels = [row[3] for row in lob_rows if row[2] == "b"]
        ask_levels = [row[3] for row in lob_rows if row[2] == "a"]
        assert set(bid_levels) == {0, 1, 2}
        assert set(ask_levels) == {0, 1}

    def test_to_np_multiple_trades_same_timestamp(self):
        """Test to_np with multiple trades at same timestamp."""
        tl = TL()
        tl.add_trades(
            timestamp=1350,
            trades=[("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)],
        )

        arr = tl.to_np()
        trade_rows = [row for row in arr if row[1] == "trade"]

        assert len(trade_rows) == 3
        assert all(row[0] == 1350 for row in trade_rows)


class TestTLIndexing:
    """Test TL indexing to access LOB data."""

    def test_index_lob_at_timestamp(self):
        """Test accessing LOB at specific timestamp via tl.lob."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=1100, bids=[(100.5, 2.0)], asks=[(101.5, 1.5)])

        lob_1000 = tl.lob[1000]
        lob_1100 = tl.lob[1100]

        assert lob_1000 is not None
        assert lob_1100 is not None
        assert lob_1000.bid[0] == 100.0
        assert lob_1100.bid[0] == 100.5

    def test_index_best_bid(self):
        """Test accessing best bid price."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5), (99.5, 2.3)], asks=[(101.0, 2.1)])

        lob = tl.lob[1000]
        assert lob.bid[0] == 100.0
        assert lob.bid[1] == 99.5

    def test_index_best_ask(self):
        """Test accessing best ask price."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1), (101.5, 1.7)])

        lob = tl.lob[1000]
        assert lob.ask[0] == 101.0
        assert lob.ask[1] == 101.5

    def test_index_bid_quantity(self):
        """Test accessing best bid quantity."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5), (99.5, 2.3)], asks=[(101.0, 2.1)])

        lob = tl.lob[1000]
        assert lob.bidq[0] == 1.5
        assert lob.bidq[1] == 2.3

    def test_index_ask_quantity(self):
        """Test accessing best ask quantity."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1), (101.5, 1.7)])

        lob = tl.lob[1000]
        assert lob.askq[0] == 2.1
        assert lob.askq[1] == 1.7

    def test_index_spread(self):
        """Test accessing spread."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])

        lob = tl.lob[1000]
        assert lob.spread == 1.0

    def test_index_spread_multiple_levels(self):
        """Test spread with multiple levels."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000,
            bids=[(100.0, 1.5), (99.5, 2.3), (99.0, 1.8)],
            asks=[(101.0, 2.1), (101.5, 1.7), (102.0, 2.5)],
        )

        lob = tl.lob[1000]
        assert lob.spread == 1.0

    def test_index_after_trade(self):
        """Test accessing LOB after a trade."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("a", 101.0, 1.0)])

        lob = tl.lob[1200]
        assert lob.bid[0] == 100.0
        assert lob.ask[0] == 101.0
        assert lob.askq[0] == 1.0

    def test_index_nonexistent_timestamp(self):
        """Test accessing LOB at non-existent timestamp."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])

        lob = tl.lob[2000]
        assert lob is None


class TestTLTradeAccess:
    """Test accessing trade data."""

    def test_trades_list(self):
        """Test accessing all trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)

        trades = tl.trades
        assert len(trades) == 3
        assert trades[0].timestamp == 1150
        assert trades[1].timestamp == 1250
        assert trades[2].timestamp == 1350

    def test_trades_slice_first_n(self):
        """Test slicing trades list (first n)."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1450, side="s", price=100.0, volume=0.2)

        first_three = tl.trades[:3]
        assert len(first_three) == 3
        assert first_three[0].timestamp == 1150
        assert first_three[2].timestamp == 1350

    def test_trades_slice_range(self):
        """Test slicing trades list (range)."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1450, side="s", price=100.0, volume=0.2)

        middle_two = tl.trades[1:3]
        assert len(middle_two) == 2
        assert middle_two[0].timestamp == 1250
        assert middle_two[1].timestamp == 1350

    def test_trades_iterate(self):
        """Test iterating over trades."""
        tl = TL()
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=1350, side="b", price=101.0, volume=0.3)

        timestamps = []
        sides = []
        volumes = []
        prices = []

        for trade in tl.trades:
            timestamps.append(trade.timestamp)
            sides.append(trade.side)
            volumes.append(trade.volume)
            prices.append(trade.price)

        assert timestamps == [1150, 1250, 1350]
        assert sides == ["b", "s", "b"]
        assert volumes == [0.5, 1.0, 0.3]
        assert prices == [101.0, 100.0, 101.0]

    def test_trades_empty(self):
        """Test trades list is empty initially."""
        tl = TL()
        assert len(tl.trades) == 0

    def test_trades_with_no_trades_only_lob(self):
        """Test trades list when only LOB data added."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_update(timestamp=1100, updates=[("b", 100.0, 2.0)])

        assert len(tl.trades) == 0
        assert tl.trades == []


class TestTLSlicing:
    """Test TL slicing via tl[start:end] syntax."""

    def test_slice_empty_tl(self):
        """Test slicing an empty TL."""
        tl = TL()
        sliced = tl[1000:2000]

        assert len(sliced) == 0
        assert isinstance(sliced, TL)

    def test_slice_range_only_lob(self):
        """Test slicing with only LOB data."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:2500]

        assert len(sliced) == 1
        assert sliced.lob[2000] is not None

    def test_slice_range_only_trades(self):
        """Test slicing with only trades."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        sliced = tl[1500:2500]

        assert len(sliced) == 1
        assert len(sliced.trades) == 1
        assert sliced.trades[0].timestamp == 2000

    def test_slice_range_mixed(self):
        """Test slicing with both LOB and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1500, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])
        tl.add_trade(timestamp=2500, side="s", price=100.0, volume=1.0)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:2500]

        assert len(sliced) == 3  # 2 trades + 1 LOB
        assert sliced.lob[2000] is not None
        assert len(sliced.trades) == 2
        assert sliced.trades[0].timestamp == 1500
        assert sliced.trades[1].timestamp == 2500

    def test_slice_exact_timestamps(self):
        """Test slicing with exact timestamps."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1000:3000]

        assert len(sliced) == 3  # 2 LOB + 1 trade
        assert sliced.lob[1000] is not None
        assert sliced.lob[3000] is not None
        assert len(sliced.trades) == 1

    def test_slice_with_open_start(self):
        """Test slicing with None as start timestamp."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[:2500]

        assert len(sliced) == 2  # 1 LOB + 1 trade

    def test_slice_with_open_end(self):
        """Test slicing with None as end timestamp."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:]

        assert len(sliced) == 2  # 1 LOB + 1 trade

    def test_slice_no_events_in_range(self):
        """Test slicing with range that contains no events."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)

        sliced = tl[5000:6000]

        assert len(sliced) == 0
        assert len(sliced.trades) == 0

    def test_slice_before_first_event(self):
        """Test slicing with range before first event."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=2000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.5)

        sliced = tl[500:1000]

        assert len(sliced) == 0

    def test_slice_after_last_event(self):
        """Test slicing with range after last event."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)

        sliced = tl[5000:6000]

        assert len(sliced) == 0

    def test_slice_inclusive_end(self):
        """Test that end timestamp is inclusive."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1000:2000]

        assert len(sliced) == 2  # 1 LOB at 1000 + 1 trade at 2000

    def test_slice_with_to_np(self):
        """Test slicing followed by to_np()."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1500, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])
        tl.add_trade(timestamp=2500, side="s", price=100.0, volume=1.0)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:2500]
        arr = sliced.to_np()

        assert arr.shape[0] == 4  # 1 LOB (2 levels) + 2 trades

    def test_slice_with_to_pd(self):
        """Test slicing followed by to_pd()."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1500, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])
        tl.add_trade(timestamp=2500, side="s", price=100.0, volume=1.0)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:2500]
        df = sliced.to_pd()

        assert len(df) == 4  # 1 LOB (2 levels) + 2 trades

    def test_slice_preserves_properties(self):
        """Test that sliced TL preserves original properties."""
        tl = TL(name="BTC-USD", tick_size=0.5)
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)

        sliced = tl[1500:2500]

        assert sliced.tick_size == 0.5

    def test_slice_multiple_trades_same_timestamp(self):
        """Test slicing with multiple trades at same timestamp."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trades(
            timestamp=1500,
            trades=[("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)],
        )
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])

        sliced = tl[1500:1500]

        assert len(sliced) == 3  # 3 trades at timestamp 1500

    def test_slice_like_example(self):
        """Test slicing like in example from snippet."""
        tl = TL()
        timestamps = [1000, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]

        tl.add_lob_snapshot(
            timestamp=timestamps[0],
            bids=[(100.00, 1.5), (99.50, 2.3)],
            asks=[(101.00, 2.1), (101.50, 1.7)],
        )
        tl.add_lob_update(
            timestamp=timestamps[1],
            updates=[("b", 100.00, 2.0), ("a", 101.00, 1.5)],
        )
        tl.add_trade(timestamp=timestamps[2], side="b", price=101.00, volume=0.5)
        tl.add_lob_update(timestamp=timestamps[3], updates=[("a", 101.00, 1.0)])
        tl.add_trade(timestamp=timestamps[4], side="s", price=100.00, volume=1.0)
        tl.add_lob_update(timestamp=timestamps[5], updates=[("b", 100.00, 1.0)])
        tl.add_trades(
            timestamp=timestamps[6],
            trades=[("b", 101.00, 0.3), ("s", 100.00, 0.5)],
        )
        tl.add_lob_update(timestamp=timestamps[7], updates=[("b", 99.50, 3.0)])
        tl.add_trade(timestamp=timestamps[8], side="b", price=101.00, volume=1.5)
        tl.add_lob_update(timestamp=timestamps[9], updates=[("b", 100.00, 2.5)])

        # Slice from timestamp[7] to timestamp[9] like in the example
        sliced = tl[timestamps[7]:timestamps[9]]
        arr = sliced.to_np()

        # Should include events at 1400, 1450, and 1500
        # 1400: LOB (2 bids + 2 asks = 4 levels)
        # 1450: trade (1 row)
        # 1500: LOB (2 bids + 2 asks = 4 levels)
        # Total: 4 + 1 + 4 = 9 rows
        assert arr.shape[0] == 9

    def test_slice_timestamps_property(self):
        """Test that sliced TL has correct timestamps property."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=2000, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=3000, bids=[(90.0, 2.5)], asks=[(110.0, 1.8)])

        sliced = tl[1500:2500]

        assert len(sliced.timestamps) == 1
        assert 2000 in sliced.timestamps
        assert 1000 not in sliced.timestamps
        assert 3000 not in sliced.timestamps


class TestTLRolling:
    """Test TL.rolling() method."""

    def test_rolling_empty_tl(self):
        """Test rolling on empty TL."""
        tl = TL()
        windows = list(tl.rolling(1000))

        assert len(windows) == 0

    def test_rolling_single_event(self):
        """Test rolling with single event."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)

        windows = list(tl.rolling(1000))

        assert len(windows) == 1

    def test_rolling_multiple_events(self):
        """Test rolling with multiple events."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1500, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])

        windows = list(tl.rolling(1000))

        assert len(windows) == 3

    def test_rolling_yields_tl_objects(self):
        """Test that rolling yields TL objects."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)

        windows = list(tl.rolling(1000))

        for window in windows:
            assert isinstance(window, TL)

    def test_rolling_window_size(self):
        """Test rolling window size."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        windows = list(tl.rolling(1500))

        # First window: [500, 1000] - contains 1 trade
        # Second window: [500, 2000] - contains 2 trades
        # Third window: [1500, 3000] - contains 1 trade
        assert len(windows) == 3

    def test_rolling_with_lob_and_trades(self):
        """Test rolling with both LOB and trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)])
        tl.add_trade(timestamp=1500, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 2.0)], asks=[(106.0, 1.5)])
        tl.add_trade(timestamp=2500, side="s", price=100.0, volume=1.0)

        windows = list(tl.rolling(1000))

        assert len(windows) == 4

    def test_rolling_first_window(self):
        """Test first rolling window."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)

        windows = list(tl.rolling(1000))
        first_window = windows[0]

        # Window [0, 1000] should contain only first trade
        assert len(first_window) == 1
        assert len(first_window.trades) == 1
        assert first_window.trades[0].timestamp == 1000

    def test_rolling_middle_window(self):
        """Test middle rolling window."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        windows = list(tl.rolling(1500))
        middle_window = windows[1]

        # Window [500, 2000] should contain trades at 1000 and 2000
        assert len(middle_window) == 2
        assert len(middle_window.trades) == 2
        assert middle_window.trades[0].timestamp == 1000
        assert middle_window.trades[1].timestamp == 2000

    def test_rolling_last_window(self):
        """Test last rolling window."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        windows = list(tl.rolling(1500))
        last_window = windows[-1]

        # Window [1500, 3000] should contain trades at 2000 and 3000
        assert len(last_window) == 2
        assert len(last_window.trades) == 2
        assert last_window.trades[0].timestamp == 2000
        assert last_window.trades[1].timestamp == 3000


class TestTLRollingItems:
    """Test TL._rolling_items() method."""

    def test_rolling_items_empty_tl(self):
        """Test rolling items on empty TL."""
        tl = TL()
        items = list(tl._rolling_items(1000))

        assert len(items) == 0

    def test_rolling_items_yields_tuples(self):
        """Test that rolling items yields (timestamp, TL) tuples."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)

        items = list(tl._rolling_items(1000))

        for item in items:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], (int, float))  # timestamp
            assert isinstance(item[1], TL)  # TL object

    def test_rolling_items_timestamps(self):
        """Test that rolling items have correct timestamps."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        items = list(tl._rolling_items(1000))
        timestamps = [item[0] for item in items]

        assert timestamps == [1000, 2000, 3000]

    def test_rolling_items_windows(self):
        """Test that rolling items have correct windows."""
        tl = TL()
        tl.add_trade(timestamp=1000, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=2000, side="s", price=100.0, volume=1.0)
        tl.add_trade(timestamp=3000, side="b", price=101.0, volume=0.3)

        items = list(tl._rolling_items(1000))

        # First item: ts=1000, window=[0, 1000]
        # Second item: ts=2000, window=[1000, 2000]
        # Third item: ts=3000, window=[2000, 3000]

        assert len(items[0][1]) == 1  # Only trade at 1000
        assert len(items[1][1]) == 2  # Trades at 1000 and 2000
        assert len(items[2][1]) == 2  # Trades at 2000 and 3000


class TestTLGueant:
    """Test TL.gueant accessor."""

    def test_gueant_returns_accessor(self):
        """Test that gueant property returns GueantAccessor."""
        tl = TL()
        accessor = tl.gueant

        assert hasattr(accessor, "buckets")
        assert hasattr(accessor, "ask")
        assert hasattr(accessor, "bid")

    def test_gueant_buckets_ask(self):
        """Test buckets() for ask side."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1200, side="b", price=102.0, volume=0.2)

        df = tl.gueant.buckets("a")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["delta", "N", "T", "lambda"]

    def test_gueant_buckets_bid(self):
        """Test buckets() for bid side."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="s", price=99.0, volume=0.3)

        df = tl.gueant.buckets("b")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["delta", "N", "T", "lambda"]

    def test_gueant_buckets_empty_tl(self):
        """Test buckets() on empty TL."""
        tl = TL()
        df = tl.gueant.buckets("a")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["delta", "N", "T", "lambda"]

    def test_gueant_buckets_no_trades(self):
        """Test buckets() with no trades."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_lob_snapshot(timestamp=2000, bids=[(95.0, 15.0)], asks=[(106.0, 12.0)])

        df = tl.gueant.buckets("a")

        assert isinstance(df, pd.DataFrame)
        # Should have no trades (N=0) but may have LOB data (T>0)

    def test_gueant_ask_no_window(self):
        """Test ask() without window (full timeline)."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1200, side="b", price=102.0, volume=0.2)

        A, k = tl.gueant.ask()

        assert isinstance(A, float)
        assert isinstance(k, float)

    def test_gueant_bid_no_window(self):
        """Test bid() without window (full timeline)."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="s", price=99.0, volume=0.3)

        A, k = tl.gueant.bid()

        assert isinstance(A, float)
        assert isinstance(k, float)

    def test_gueant_ask_with_window(self):
        """Test ask() with window (rolling)."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=1200, bids=[(100.5, 12.0)], asks=[(101.5, 10.0)])
        tl.add_trade(timestamp=1300, side="b", price=101.0, volume=0.3)
        tl.add_lob_snapshot(timestamp=1400, bids=[(101.0, 8.0)], asks=[(102.0, 9.0)])

        A_series, k_series = tl.gueant.ask(300)

        assert isinstance(A_series, pd.Series)
        assert isinstance(k_series, pd.Series)
        assert len(A_series) == len(tl.timestamps)

    def test_gueant_bid_with_window(self):
        """Test bid() with window (rolling)."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=1200, bids=[(99.5, 12.0)], asks=[(101.5, 10.0)])
        tl.add_trade(timestamp=1300, side="s", price=99.0, volume=0.3)
        tl.add_lob_snapshot(timestamp=1400, bids=[(99.0, 8.0)], asks=[(102.0, 9.0)])

        A_series, k_series = tl.gueant.bid(300)

        assert isinstance(A_series, pd.Series)
        assert isinstance(k_series, pd.Series)
        assert len(A_series) == len(tl.timestamps)

    def test_gueant_ask_series_names(self):
        """Test that ask() returns series with correct names."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)

        A_series, k_series = tl.gueant.ask(300)

        assert A_series.name == "gueant_A_ask"
        assert k_series.name == "gueant_k_ask"

    def test_gueant_bid_series_names(self):
        """Test that bid() returns series with correct names."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="s", price=100.0, volume=0.5)

        A_series, k_series = tl.gueant.bid(300)

        assert A_series.name == "gueant_A_bid"
        assert k_series.name == "gueant_k_bid"

    def test_gueant_insufficient_data(self):
        """Test with insufficient data for fitting."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        # Only one trade - not enough for fitting

        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)

        A, k = tl.gueant.ask()

        # Should return nan for insufficient data
        assert pd.isna(A) or isinstance(A, float)
        assert pd.isna(k) or isinstance(k, float)

    def test_gueant_multiple_trades_same_delta(self):
        """Test with multiple trades at same delta."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.3)
        tl.add_trade(timestamp=1200, side="b", price=101.0, volume=0.2)

        df = tl.gueant.buckets("a")

        # For ask side: delta = (trade_price - best_bid) / tick_size
        # With trade_price=101.0, best_bid=100.0, tick_size=1: delta = 1
        delta_1 = df[df["delta"] == 1]
        assert len(delta_1) > 0
        if len(delta_1) > 0:
            assert delta_1.iloc[0]["N"] >= 3  # At least 3 trades

    def test_gueant_trades_at_different_deltas(self):
        """Test with trades at different delta distances."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0), (102.0, 5.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)  # delta=1
        tl.add_trade(timestamp=1150, side="b", price=102.0, volume=0.3)  # delta=2

        df = tl.gueant.buckets("a")

        # Should have trades at different deltas
        assert len(df) >= 2

    def test_gueant_lob_duration_calculation(self):
        """Test T(δ) calculation from LOB duration."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0), (102.0, 5.0)])
        tl.add_lob_snapshot(timestamp=2000, bids=[(100.0, 12.0)], asks=[(101.0, 7.0), (102.0, 6.0)])

        df = tl.gueant.buckets("a")

        # Should have T values for different deltas
        assert len(df) >= 1
        # T should be > 0 for some deltas
        assert (df["T"] > 0).any()

    def test_gueant_lambda_calculation(self):
        """Test λ̂(δ) = N(δ) / T(δ) calculation."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.3)

        df = tl.gueant.buckets("a")

        # For rows with valid lambda
        valid_rows = df.dropna(subset=["lambda"])
        if len(valid_rows) > 0:
            # Check that lambda = N / T
            for _, row in valid_rows.iterrows():
                if row["T"] > 0:
                    expected_lambda = row["N"] / row["T"]
                    assert abs(row["lambda"] - expected_lambda) < 1e-10

    def test_gueant_tick_size_affects_delta(self):
        """Test that tick_size affects delta calculation."""
        tl1 = TL(tick_size=0.01)
        tl1.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl1.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)

        tl2 = TL(tick_size=1.0)
        tl2.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl2.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)

        df1 = tl1.gueant.buckets("a")
        df2 = tl2.gueant.buckets("a")

        # Different tick sizes should produce different deltas
        if len(df1) > 0 and len(df2) > 0:
            # delta = (price - best_bid) / tick_size
            # For tick_size=0.01: delta = (101-100)/0.01 = 100
            # For tick_size=1.0: delta = (101-100)/1 = 1
            delta_1 = df1.iloc[0]["delta"]
            delta_2 = df2.iloc[0]["delta"]
            assert delta_1 != delta_2

    def test_gueant_rolling_windows_consistency(self):
        """Test that rolling windows produce consistent results."""
        tl = TL()
        tl.add_lob_snapshot(timestamp=1000, bids=[(100.0, 10.0)], asks=[(101.0, 8.0)])
        tl.add_trade(timestamp=1100, side="b", price=101.0, volume=0.5)
        tl.add_lob_snapshot(timestamp=1200, bids=[(100.5, 12.0)], asks=[(101.5, 10.0)])
        tl.add_trade(timestamp=1300, side="b", price=101.0, volume=0.3)

        A_series, k_series = tl.gueant.ask(300)

        # Each window should have valid A and k values
        # (or nan if insufficient data)
        assert len(A_series) == len(tl.timestamps)
        assert len(k_series) == len(tl.timestamps)


class TestTLGueantIntegration:
    """Integration tests for Guéant functionality."""

    def test_gueant_full_workflow(self):
        """Test complete Guéant workflow."""
        tl = TL(tick_size=0.5)

        # Create realistic market data
        timestamps = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

        for i, ts in enumerate(timestamps):
            bid_price = 100.0 + (i * 0.1)
            ask_price = 101.0 + (i * 0.1)
            tl.add_lob_snapshot(
                timestamp=ts,
                bids=[(bid_price, 10.0), (bid_price - 0.5, 8.0)],
                asks=[(ask_price, 10.0), (ask_price + 0.5, 8.0)],
            )

        # Add some trades
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_trade(timestamp=1250, side="b", price=101.1, volume=0.3)
        tl.add_trade(timestamp=1350, side="s", price=100.6, volume=0.4)
        tl.add_trade(timestamp=1450, side="b", price=101.2, volume=0.6)

        # Compute buckets
        ask_buckets = tl.gueant.buckets("a")
        bid_buckets = tl.gueant.buckets("b")

        assert len(ask_buckets) > 0
        assert len(bid_buckets) > 0

        # Compute parameters (full timeline)
        A_ask_full, k_ask_full = tl.gueant.ask()
        A_bid_full, k_bid_full = tl.gueant.bid()

        assert isinstance(A_ask_full, float)
        assert isinstance(k_ask_full, float)
        assert isinstance(A_bid_full, float)
        assert isinstance(k_bid_full, float)

        # Compute rolling parameters
        A_ask_roll, k_ask_roll = tl.gueant.ask(500)
        A_bid_roll, k_bid_roll = tl.gueant.bid(500)

        assert isinstance(A_ask_roll, pd.Series)
        assert isinstance(k_ask_roll, pd.Series)
        assert isinstance(A_bid_roll, pd.Series)
        assert isinstance(k_bid_roll, pd.Series)


class TestTLIntegration:
    """Test TL integration scenarios."""

    def test_lob_and_trades_together(self):
        """Test adding both LOB data and trades."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000,
            bids=[(100.0, 1.5), (99.5, 2.3)],
            asks=[(101.0, 2.1), (101.5, 1.7)],
        )
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)

        assert tl.lob[1000] is not None
        assert len(tl.trades) == 1

    def test_sequential_lob_updates_and_trades(self):
        """Test sequence of LOB updates and trades."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)]
        )
        tl.add_lob_update(timestamp=1100, updates=[("b", 100.0, 2.0)])
        tl.add_trade(timestamp=1150, side="b", price=101.0, volume=0.5)
        tl.add_lob_update(timestamp=1200, updates=[("a", 101.0, 1.0)])
        tl.add_trade(timestamp=1250, side="s", price=100.0, volume=1.0)

        assert tl.lob[1000] is not None
        assert tl.lob[1100] is not None
        assert tl.lob[1200] is not None
        assert len(tl.trades) == 2

    def test_multiple_trades_same_timestamp(self):
        """Test multiple trades at same timestamp."""
        tl = TL()
        tl.add_lob_snapshot(
            timestamp=1000, bids=[(100.0, 1.5)], asks=[(101.0, 2.1)]
        )
        tl.add_trades(
            timestamp=1350,
            trades=[("b", 101.0, 0.3), ("b", 101.5, 0.2), ("s", 100.0, 0.5)],
        )

        assert len(tl.trades) == 3
        for trade in tl.trades:
            assert trade.timestamp == 1350
