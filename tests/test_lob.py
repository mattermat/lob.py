import pytest
from lobpy.lob import LOB

# Check if numpy and pandas are available
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY = True
    HAS_PANDAS = True
except ImportError:
    HAS_NUMPY = False
    HAS_PANDAS = False


class TestLOBInit:
    """Test LOB initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        lob = LOB()
        assert lob.name.startswith("lob")
        assert lob.tick_size == 1
        assert lob.timestamp > 0
        assert len(lob._bids) == 0
        assert len(lob._asks) == 0

    def test_init_with_name(self):
        """Test initialization with custom name."""
        lob = LOB(name="test_book")
        assert lob.name == "test_book"


class TestLOBTickSize:
    """Test tick size functionality."""

    def test_set_tick_size(self):
        """Test setting tick size."""
        lob = LOB()
        lob._set_tick_size(0.01)
        assert lob.tick_size == 0.01

        lob._set_tick_size(0.1)
        assert lob.tick_size == 0.1


class TestLOBSetUpdates:
    """Test set_updates functionality."""

    def test_set_updates_with_long_form(self):
        """Test set_updates with long form side parameters ('bid', 'ask')."""
        lob = LOB()
        lob.set_updates([('bid', 100, 10), ('bid', 99, 5), ('ask', 101, 8), ('ask', 102, 4)])

        assert len(lob._bids) == 2
        assert len(lob._asks) == 2
        assert lob._bids[100] == 10
        assert lob._bids[99] == 5
        assert lob._asks[101] == 8
        assert lob._asks[102] == 4

    def test_set_updates_with_short_form(self):
        """Test set_updates with short form side parameters ('b', 'a')."""
        lob = LOB()
        lob.set_updates([('b', 100, 10), ('b', 99, 5), ('a', 101, 8), ('a', 102, 4)])

        assert len(lob._bids) == 2
        assert len(lob._asks) == 2
        assert lob._bids[100] == 10
        assert lob._bids[99] == 5
        assert lob._asks[101] == 8
        assert lob._asks[102] == 4

    def test_set_updates_mixed_forms(self):
        """Test set_updates with mixed short and long form side parameters."""
        lob = LOB()
        lob.set_updates([('bid', 100, 10), ('b', 99, 5), ('ask', 101, 8), ('a', 102, 4)])

        assert len(lob._bids) == 2
        assert len(lob._asks) == 2
        assert lob._bids[100] == 10
        assert lob._bids[99] == 5
        assert lob._asks[101] == 8
        assert lob._asks[102] == 4

    def test_set_updates_delete_levels(self):
        """Test set_updates deleting levels with short form."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)])

        # Delete level with zero size using short form
        lob.set_updates([('b', 100, 0), ('a', 101, 0)])

        assert 100 not in lob._bids
        assert 101 not in lob._asks
        assert 99 in lob._bids
        assert 102 in lob._asks

    def test_set_updates_with_timestamp(self):
        """Test set_updates with timestamp using short form."""
        lob = LOB()
        timestamp = 1234567890
        lob.set_updates([('b', 100, 10), ('a', 101, 8)], timestamp=timestamp)

        assert lob.timestamp == timestamp


class TestLOBSnapshot:
    """Test snapshot functionality."""

    def test_set_snapshot(self):
        """Test setting order book to a snapshot."""
        lob = LOB()
        bids = [(100, 10), (99, 5), (98, 3)]
        asks = [(101, 8), (102, 4)]

        lob.set_snapshot(bids, asks)

        assert len(lob._bids) == 3
        assert len(lob._asks) == 2
        assert lob._bids[100] == 10
        assert lob._asks[101] == 8

    def test_set_snapshot_empty(self):
        """Test setting empty snapshot."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [(101, 8)])
        lob.set_snapshot([], [])

        assert len(lob._bids) == 0
        assert len(lob._asks) == 0

    def test_set_snapshot_with_timestamp(self):
        """Test setting snapshot with timestamp."""
        lob = LOB()
        timestamp = 1234567890
        lob.set_snapshot([(100, 10)], [(101, 8)], timestamp=timestamp)

        assert lob.timestamp == timestamp

    def test_set_snapshot_overwrites(self):
        """Test that snapshot overwrites existing data."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [(101, 8)])
        lob.set_snapshot([(95, 15), (94, 7)], [(106, 12)])

        assert 100 not in lob._bids
        assert 101 not in lob._asks
        assert lob._bids[95] == 15
        assert lob._asks[106] == 12


class TestLOBDeleteLevel:
    """Test delete level functionality."""

    def test_delete_bid_level(self):
        """Test deleting a bid level."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [])
        lob._delete_bid_level(100)

        assert 100 not in lob._bids
        assert 99 in lob._bids
        assert len(lob._bids) == 1

    def test_delete_bid_level_nonexistent(self):
        """Test deleting a non-existent bid level."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        lob._delete_bid_level(99)  # Should not raise exception

        assert len(lob._bids) == 1

    def test_delete_ask_level(self):
        """Test deleting an ask level."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4)])
        lob._delete_ask_level(101)

        assert 101 not in lob._asks
        assert 102 in lob._asks
        assert len(lob._asks) == 1

    def test_delete_ask_level_nonexistent(self):
        """Test deleting a non-existent ask level."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8)])
        lob._delete_ask_level(100)  # Should not raise exception

        assert len(lob._asks) == 1

    def test_delete_level_bid(self):
        """Test delete_level with side='bid'."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        lob._delete_level("bid", 100)

        assert 100 not in lob._bids

    def test_delete_level_ask(self):
        """Test delete_level with side='ask'."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8)])
        lob._delete_level("ask", 101)

        assert 101 not in lob._asks

    def test_delete_level_with_timestamp(self):
        """Test delete_level updates timestamp."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        timestamp = 1234567890
        lob._delete_level("bid", 100, timestamp=timestamp)

        assert lob.timestamp == timestamp


class TestLOBUpdate:
    """Test update functionality."""

    def test_update_bid_existing(self):
        """Test updating an existing bid level."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        lob._update_bid(100, 15)

        assert lob._bids[100] == 15

    def test_update_bid_new(self):
        """Test updating with a new bid level."""
        lob = LOB()
        lob._update_bid(100, 10)

        assert 100 in lob._bids
        assert lob._bids[100] == 10

    def test_update_bid_zero_size(self):
        """Test updating bid with zero size (should delete)."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        lob._update_bid(100, 0)

        assert 100 not in lob._bids

    def test_update_ask_existing(self):
        """Test updating an existing ask level."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8)])
        lob._update_ask(101, 12)

        assert lob._asks[101] == 12

    def test_update_ask_new(self):
        """Test updating with a new ask level."""
        lob = LOB()
        lob._update_ask(101, 8)

        assert 101 in lob._asks
        assert lob._asks[101] == 8

    def test_update_ask_zero_size(self):
        """Test updating ask with zero size (should delete)."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8)])
        lob._update_ask(101, 0)

        assert 101 not in lob._asks

    def test_update_bid_with_timestamp(self):
        """Test update_bid updates timestamp."""
        lob = LOB()
        timestamp = 1234567890
        lob._update_bid(100, 10, timestamp=timestamp)

        assert lob.timestamp == timestamp

    def test_update_ask_with_timestamp(self):
        """Test update_ask updates timestamp."""
        lob = LOB()
        timestamp = 1234567890
        lob._update_ask(101, 8, timestamp=timestamp)

        assert lob.timestamp == timestamp

    def test_update_side_bid(self):
        """Test update method with side='bid'."""
        lob = LOB()
        lob.update("bid", 100, 10)

        assert lob._bids[100] == 10

    def test_update_side_ask(self):
        """Test update method with side='ask'."""
        lob = LOB()
        lob.update("ask", 101, 8)

        assert lob._asks[101] == 8

    def test_update_zero_size_deletes(self):
        """Test that update with zero size deletes level."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [(101, 8)])
        lob.update("bid", 100, 0)
        lob.update("ask", 101, 0)

        assert 100 not in lob._bids
        assert 101 not in lob._asks

    def test_update_with_short_form_bid(self):
        """Test update method with side='b' (short form for bid)."""
        lob = LOB()
        lob.update("b", 100, 10)

        assert lob._bids[100] == 10

    def test_update_with_short_form_ask(self):
        """Test update method with side='a' (short form for ask)."""
        lob = LOB()
        lob.update("a", 101, 8)

        assert lob._asks[101] == 8


class TestLOBProperties:
    """Test LoB properties."""

    def test_ask_property(self):
        """Test ask property returns best ask price."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        # ask should return just the price
        assert lob.ask == 101

    def test_ask_property_empty(self):
        """Test ask property returns zeros when empty."""
        lob = LOB()
        assert lob.ask == 0.0

    def test_bid_property(self):
        """Test bid property returns best bid price."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        # bid should return just the price
        assert lob.bid == 100

    def test_bid_property_empty(self):
        """Test bid property returns zeros when empty."""
        lob = LOB()
        assert lob.bid == 0.0

    def test_spread_property_one_side(self):
        """Test spread when only one side has orders."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        import math
        assert math.isnan(lob.spread)

    def test_spread_property_empty(self):
        """Test spread returns nan when empty."""
        lob = LOB()
        import math
        assert math.isnan(lob.spread)

    def test_spread_property_one_side(self):
        """Test spread when only one side has orders."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])
        import math
        assert math.isnan(lob.spread)

class TestLOBGetDelta:
    """Test get_delta functionality."""

    def test_get_delta_simple(self):
        """Test get_delta with simple changes."""
        lob = LOB(bids=[(100, 10), (99, 5)], asks=[(101, 8)])

        bid_deltas, ask_deltas = lob.get_delta(
            [(100, 15), (98, 3)],  # Changed 100, added 98, removed 99
            [(101, 8), (102, 4)]   # Added 102
        )

        assert len(bid_deltas) == 3
        assert (100, 15) in bid_deltas  # Changed
        assert (98, 3) in bid_deltas    # Added
        assert (99, 0.0) in bid_deltas  # Deleted
        assert len(ask_deltas) == 1
        assert (102, 4) in ask_deltas

    def test_get_delta_no_changes(self):
        """Test get_delta when nothing changes."""
        lob = LOB(bids=[(100, 10)], asks=[(101, 8)])

        bid_deltas, ask_deltas = lob.get_delta(
            [(100, 10)],
            [(101, 8)]
        )

        assert len(bid_deltas) == 0
        assert len(ask_deltas) == 0

    def test_get_delta_updates_state(self):
        """Test that get_delta updates internal state."""
        lob = LOB(bids=[(100, 10)], asks=[(101, 8)])

        lob.get_delta([(95, 15)], [(106, 12)])

        assert len(lob._bids) == 1
        assert 95 in lob._bids
        assert lob._bids[95] == 15
        assert len(lob._asks) == 1
        assert 106 in lob._asks
        assert lob._asks[106] == 12

    def test_get_delta_with_timestamp(self):
        """Test get_delta updates timestamp."""
        lob = LOB()
        timestamp = 1234567890

        lob.get_delta([], [], timestamp=timestamp)

        assert lob.timestamp == timestamp

    def test_get_delta_empty_snapshot(self):
        """Test get_delta with empty new snapshot."""
        lob = LOB(bids=[(100, 10)], asks=[(101, 8)])

        bid_deltas, ask_deltas = lob.get_delta([], [])

        assert len(bid_deltas) == 1
        assert (100, 0.0) in bid_deltas
        assert len(ask_deltas) == 1
        assert (101, 0.0) in ask_deltas

    def test_get_delta_all_new(self):
        """Test get_delta when all levels are new."""
        lob = LOB()

        bid_deltas, ask_deltas = lob.get_delta(
            [(100, 10), (99, 5)],
            [(101, 8), (102, 4)]
        )

        assert len(bid_deltas) == 2
        assert len(ask_deltas) == 2

    def test_get_delta_complex_scenario(self):
        """Test get_delta with complex scenario from docstring example."""
        lob = LOB(bids=[(100, 10), (99, 5)], asks=[])

        bid_deltas, ask_deltas = lob.get_delta(
            [(100, 15), (98, 3)],
            []
        )

        assert len(bid_deltas) == 3
        assert (100, 15) in bid_deltas  # Changed
        assert (99, 0.0) in bid_deltas  # Deleted
        assert (98, 3) in bid_deltas    # Added


class TestLOBRepr:
    """Test LoB string representation."""

    def test_repr_default_name(self):
        """Test __repr__ with default name."""
        lob = LOB()
        repr_str = repr(lob)
        assert repr_str.startswith("<Book[lob")
        assert repr_str.endswith("]>")

    def test_repr_custom_name(self):
        """Test __repr__ with custom name."""
        lob = LOB(name="my_book")
        assert repr(lob) == "<Book[my_book]>"


class TestLOBBestPriceOrdering:
    """Test that best prices are correctly ordered."""

    def test_best_bid_is_highest(self):
        """Test that best bid is the highest price."""
        lob = LOB()
        lob.set_snapshot([(98, 5), (100, 10), (99, 7)], [])
        assert lob.bid == 100

    def test_best_ask_is_lowest(self):
        """Test that best ask is the lowest price."""
        lob = LOB()
        lob.set_snapshot([], [(103, 2), (101, 8), (102, 4)])

        assert lob.ask == 101
        # ask[0] should equal ask (price only)
        assert lob.ask[0] == 101
        # ask[1] should be second best ask (price only)
        assert lob.ask[1] == 102
        # ask[2] should be third best ask (price only)
        assert lob._asks.peekitem(2) == (103, 2)

    def test_bid_ask_sizes_with_floats(self):
        """Test bidq and askq properties for best bid/ask sizes with floats."""
        lob = LOB()
        lob.set_snapshot([(100, 10.5)], [(101, 8.75)])

        # bidq should be best bid size (float)
        assert lob.bidq == 10.5
        # askq should be best ask size (float)
        assert lob.askq == 8.75


class TestLOBSpreadProperties:
    """Test LOB spread-related properties."""

    def test_spread_with_different_tick_sizes(self):
        """Test spread property with different tick sizes."""
        lob = LOB()
        lob._set_tick_size(0.5)
        lob.set_snapshot([(100, 10)], [(101, 8)])

        # Spread = 1.0, Tick size = 0.5
        assert lob.spread == 1.0
        # Tick size is just for reference, spread is always in price units

    def test_midprice_with_floats(self):
        """Test midprice property - mid-price."""
        lob = LOB()
        lob.set_snapshot([(100.5, 10)], [(101.75, 8)])

        # Mid-price = (bid + ask) / 2 = (100.5 + 101.75) / 2 = 101.125
        # Note: This property doesn't exist yet in LOB class
        # Just documenting what should be tested once implemented

    def test_vw_midprice_with_floats(self):
        """Test vw_midprice property - volume-weighted mid-price."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [(101, 8)])

        # VW mid-price = (bid_price * bid_size + ask_price * ask_size) / (bid_size + ask_size)
        # = (100 * 10 + 101 * 8) / (10 + 8)
        # = (1000 + 808) / 18
        # = 1808 / 18
        # = 100.444...
        # Note: This property doesn't exist yet in LOB class
        # Just documenting what should be tested once implemented


class TestLOBAggq:
    """Test LOB.aggq() method for aggregating order book quantities."""

    def test_aggq_ask_nlevels(self):
        """Test aggq for asks with nlevel parameter."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2), (104, 6)])

        # Sum top 3 ask levels: 8 + 4 + 2 = 14
        assert lob.aggq('ask', nlevel=3) == 14

    def test_aggq_bid_nlevels(self):
        """Test aggq for bids with nlevel parameter."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3), (97, 2)], [])

        # Sum top 3 bid levels: 10 + 5 + 3 = 18
        assert lob.aggq('bid', nlevel=3) == 18

    def test_aggq_bid_with_floats(self):
        """Test aggq for bids with float prices and sizes."""
        lob = LOB()
        lob.set_snapshot([(100.5, 10.25), (99.75, 5.5), (98.5, 3.75)], [])

        # Sum all bid levels: 10.25 + 5.5 + 3.75 = 19.5
        assert lob.aggq('bid', nlevel=3) == 19.5

    def test_aggq_ask_empty(self):
        """Test aggq for asks when no asks."""
        lob = LOB()
        lob.set_snapshot([(100, 10)], [])

         # assert lob.aggq('ask', nlevel=10) == 0.0
        assert lob.aggq('ask', nlevel=10) == 0.0
        assert lob.aggq('bid', nlevel=10) == 10.0

    def test_aggq_bid_empty(self):
        """Test aggq for bids when no bids."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8)])


    def test_aggq_ask_one_level(self):
        """Test aggq for asks with nlevel=1."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        # Sum top 1 ask level: 8
        assert lob.aggq('ask', nlevel=1) == 8

    def test_aggq_bid_one_level(self):
        """Test aggq for bids with nlevel=1."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        # Sum top 1 bid level: 10
        assert lob.aggq('bid', nlevel=1) == 10

    def test_aggq_ask_nlevel_zero(self):
        """Test aggq for asks with nlevel=0."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        assert lob.aggq('ask', nlevel=0) == 0.0

    def test_aggq_bid_nlevel_zero(self):
        """Test aggq for bids with nlevel=0."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        assert lob.aggq('bid', nlevel=0) == 0.0

    def test_aggq_ask_nlevel_exceeds_levels(self):
        """Test aggq for asks when nlevel > available levels."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4)])

        # nlevel=5 but only 2 levels exist, sum all: 8 + 4 = 12
        assert lob.aggq('ask', nlevel=5) == 12

    def test_aggq_bid_nlevel_exceeds_levels(self):
        """Test aggq for bids when nlevel > available levels."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [])

        assert lob.aggq('bid', nlevel=10) == 15

    def test_aggq_ask_large_ticks(self):
        """Test aggq for asks with ticks parameter that includes all levels."""
        lob = LOB()
        lob._set_tick_size(0.5)
        lob.set_snapshot([], [(101, 8), (101.5, 4), (102, 2)])

        # Sum all asks (within large tick count): 8 + 4 + 2 = 14
        assert lob.aggq('ask', ticks=100) == 14

    def test_aggq_bid_ticks_boundary(self):
        """Test aggq for bids with ticks at boundary."""
        lob = LOB()
        lob._set_tick_size(1.0)
        lob.set_snapshot([(100, 10), (99, 5), (98, 3), (97, 2)], [])

        # Within 2 ticks from top (100, 99): 10 + 5 = 15
        # 98 is at 100 - 2 = 98, which is exactly 2 ticks away, so included
        assert lob.aggq('bid', ticks=2) == 10 + 5 + 3 == 18

    def test_aggq_price_boundary(self):
        """Test aggq for asks with price at boundary."""
        lob = LOB()
        lob.set_snapshot([], [(105, 8), (103, 4), (102, 2), (101, 6)])

        # At price <= 103.5: 6 + 2 + 4 = 12
        assert lob.aggq('ask', price=103.5) == 12
        # At price <= 103 is the same
        assert lob.aggq('ask', price=103) == 12

    def test_aggq_with_short_form(self):
        """Test aggq with short form side parameters ('b', 'a')."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3), (97, 2)], [(101, 8), (102, 4), (103, 2), (104, 6)])

        # Test with short form 'b' for bids
        assert lob.aggq('b', nlevel=3) == 18
        # Test with short form 'a' for asks
        assert lob.aggq('a', nlevel=3) == 14

    def test_aggq_short_form_with_ticks(self):
        """Test aggq with short form and ticks parameter."""
        lob = LOB()
        lob._set_tick_size(1.0)
        lob.set_snapshot([(100, 10), (99, 5), (98, 3), (97, 2)], [(101, 8), (102, 4), (103, 2), (104, 6)])

        # Test with short form 'b' and ticks
        assert lob.aggq('b', ticks=2) == 18
        # Test with short form 'a' and ticks
        assert lob.aggq('a', ticks=2) == 14

    def test_aggq_short_form_with_price(self):
        """Test aggq with short form and price parameter."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3), (97, 2)], [(105, 8), (103, 4), (102, 2), (101, 6)])

        # Test with short form 'b' and price
        assert lob.aggq('b', price=98) == 10 + 5 + 3
        # Test with short form 'a' and price
        assert lob.aggq('a', price=103) == 6 + 2 + 4


class TestLOBToNumpy:

    def test_to_np_bids_side(self):
        """Test to_np with side='b' for bids."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        result = lob.to_np(side='b')

        assert result.shape == (3, 2)
        # Check ordering (best bid first)
        assert result[0, 0] == 100  # Best bid price
        assert result[0, 1] == 10   # Best bid size
        assert result[1, 0] == 99
        assert result[1, 1] == 5
        assert result[2, 0] == 98
        assert result[2, 1] == 3

        """Test to_np with side='a' for asks."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        result = lob.to_np(side='a')

        assert result.shape == (3, 2)
        # Check ordering (best ask first)
        assert result[0, 0] == 101  # Best ask price
        assert result[0, 1] == 8    # Best ask size
        assert result[1, 0] == 102
        assert result[1, 1] == 4
        assert result[2, 0] == 103
        assert result[2, 1] == 2

        """Test to_np with side=None for both sides."""
        lob = LOB()
        lob.set_snapshot(
            [(100, 10), (99, 5)],
            [(101, 8), (102, 4)]
        )

        result = lob.to_np(side=None)

        assert result.shape == (4, 3)
        # Bids first (best to worst)
        assert result[0, 0] == 'b'   # Side column
        assert result[0, 1] == 100   # Price
        assert result[0, 2] == 10    # Size
        assert result[1, 0] == 'b'
        assert result[1, 1] == 99
        assert result[1, 2] == 5
        # Then asks (best to worst)
        assert result[2, 0] == 'a'   # Side column
        assert result[2, 1] == 101   # Price
        assert result[2, 2] == 8    # Size
        assert result[3, 0] == 'a'
        assert result[3, 1] == 102
        assert result[3, 2] == 4

        """Test to_np with empty order book."""
        lob = LOB()

        result_bids = lob.to_np(side='b')
        result_asks = lob.to_np(side='a')
        result_both = lob.to_np(side=None)

        assert result_bids.shape == (0, 2)
        assert result_asks.shape == (0, 2)
        assert result_both.shape == (0, 3)

        """Test to_np with nlevels parameter."""
        lob = LOB()
        lob.set_snapshot(
            [(100, 10), (99, 5), (98, 3), (97, 2)],
            [(101, 8), (102, 4), (103, 2)]
        )

        result_bids = lob.to_np(side='b', nlevels=2)
        result_asks = lob.to_np(side='a', nlevels=2)
        result_both = lob.to_np(side=None, nlevels=3)

        assert result_bids.shape == (2, 2)
        assert result_bids[0, 0] == 100
        assert result_bids[1, 0] == 99

        assert result_asks.shape == (2, 2)
        assert result_asks[0, 0] == 101
        assert result_asks[1, 0] == 102

        assert result_both.shape == (3, 3)
        # 2 bids + 1 ask = 3 levels
        assert result_both[0, 0] == 'b'
        assert result_both[1, 0] == 'b'
        assert result_both[2, 0] == 'a'

        """Test to_np with float prices and sizes."""
        lob = LOB()
        lob.set_snapshot([(100.5, 10.25), (99.75, 5.5)], [(101.25, 8.75)])

        result = lob.to_np(side=None)

        assert result[0, 1] == 100.5
        assert result[0, 2] == 10.25
        assert result[1, 1] == 99.75
        assert result[1, 2] == 5.5
        assert result[2, 1] == 101.25
        assert result[2, 2] == 8.75


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestLOBToPandas:

        """Test to_pd with side='b' for bids."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        df = lob.to_pd(side='b')

        assert df.shape == (3, 2)
        assert list(df.columns) == ['price', 'size']
        # Check ordering
        assert df.iloc[0]['price'] == 100
        assert df.iloc[0]['size'] == 10
        assert df.iloc[1]['price'] == 99
        assert df.iloc[2]['price'] == 98

        """Test to_pd with side='a' for asks."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        df = lob.to_pd(side='a')

        assert df.shape == (3, 2)
        assert list(df.columns) == ['price', 'size']
        assert df.iloc[0]['price'] == 101
        assert df.iloc[0]['size'] == 8
        assert df.iloc[1]['price'] == 102
        assert df.iloc[2]['price'] == 103

        """Test to_pd with side=None for both sides."""
        lob = LOB()
        lob.set_snapshot(
            [(100, 10), (99, 5)],
            [(101, 8), (102, 4)]
        )

        df = lob.to_pd(side=None)

        assert df.shape == (4, 3)
        assert list(df.columns) == ['price', 'size', 'side']
        # Bids first
        assert df.iloc[0]['side'] == 'b'
        assert df.iloc[0]['price'] == 100
        assert df.iloc[0]['size'] == 10
        assert df.iloc[1]['side'] == 'b'
        assert df.iloc[1]['price'] == 99
        # Then asks
        assert df.iloc[2]['side'] == 'a'
        assert df.iloc[2]['price'] == 101
        assert df.iloc[3]['side'] == 'a'
        assert df.iloc[3]['price'] == 102

        """Test to_pd with empty order book."""
        lob = LOB()

        df_bids = lob.to_pd(side='b')
        df_asks = lob.to_pd(side='a')
        df_both = lob.to_pd(side=None)

        assert len(df_bids) == 0
        assert len(df_asks) == 0
        assert len(df_both) == 0
        assert list(df_bids.columns) == ['price', 'size']
        assert list(df_asks.columns) == ['price', 'size']
        assert list(df_both.columns) == ['price', 'size', 'side']

        """Test to_pd with nlevels parameter."""
        lob = LOB()
        lob.set_snapshot(
            [(100, 10), (99, 5), (98, 3)],
            [(101, 8), (102, 4)]
        )

        df = lob.to_pd(side=None, nlevels=2)

        assert len(df) == 2
        # Only best bid
        assert df.iloc[0]['side'] == 'b'
        assert df.iloc[0]['price'] == 100
        # And best ask
        assert df.iloc[1]['side'] == 'a'
        assert df.iloc[1]['price'] == 101

        """Test to_pd with float prices and sizes."""
        lob = LOB()
        lob.set_snapshot([(100.5, 10.25)], [(101.75, 8.5)])

        df = lob.to_pd(side=None)

        assert df.iloc[0]['price'] == 100.5
        assert df.iloc[0]['size'] == 10.25
        assert df.iloc[0]['side'] == 'b'
        assert df.iloc[1]['price'] == 101.75
        assert df.iloc[1]['size'] == 8.5
        assert df.iloc[1]['side'] == 'a'


class TestLOBExportCSV:

    def test_to_csv_bids(self, tmp_path):
        """Test to_csv with side='b' for bids."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        path = tmp_path / "bids.csv"
        lob.to_csv(path, side='b')

        assert path.exists()
        df = lob.to_pd(side='b')
        df.to_csv(path, index=False)
        assert path.exists()

    def test_to_csv_asks(self, tmp_path):
        """Test to_csv with side='a' for asks."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4)])

        path = tmp_path / "asks.csv"
        lob.to_csv(path, side='a')

        assert path.exists()

    def test_to_csv_both_sides(self, tmp_path):
        """Test to_csv with side=None for both sides."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)])

        path = tmp_path / "both.csv"
        lob.to_csv(path, side=None)

        assert path.exists()

    def test_to_csv_with_nlevels(self, tmp_path):
        """Test to_csv with nlevels parameter."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)])

        path = tmp_path / "nlevels.csv"
        lob.to_csv(path, side='b', nlevels=2)

        assert path.exists()

    def test_to_csv_empty_book(self, tmp_path):
        """Test to_csv with empty order book."""
        lob = LOB()

        path = tmp_path / "empty.csv"
        lob.to_csv(path, side=None)

        assert path.exists()


class TestLOBExportXLSX:

    def test_to_csv_bids(self, tmp_path):
        """Test to_xlsx with side='b' for bids."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        path = tmp_path / "bids.xlsx"
        lob.to_xlsx(path, side='b')

        assert path.exists()

    def test_to_csv_asks(self, tmp_path):
        """Test to_xlsx with side='a' for asks."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        path = tmp_path / "asks.xlsx"
        lob.to_xlsx(path, side='a')

        assert path.exists()

    def test_to_csv_both_sides(self, tmp_path):
        """Test to_xlsx with side=None for both sides."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)])

        path = tmp_path / "both.xlsx"
        lob.to_xlsx(path, side=None)

        assert path.exists()

    def test_to_csv_with_nlevels(self, tmp_path):
        """Test to_xlsx with nlevels parameter."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)])

        path = tmp_path / "nlevels.xlsx"
        lob.to_xlsx(path, side='b', nlevels=2)

        assert path.exists()

    def test_to_csv_empty_book(self, tmp_path):
        """Test to_xlsx with empty order book."""
        lob = LOB()

        path = tmp_path / "empty.xlsx"
        lob.to_xlsx(path, side=None)

        assert path.exists()


class TestLOBExportParquet:

    def test_to_csv_bids(self, tmp_path):
        """Test to_parquet with side='b' for bids."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [])

        path = tmp_path / "bids.parquet"
        lob.to_parquet(path, side='b')

        assert path.exists()

    def test_to_csv_asks(self, tmp_path):
        """Test to_parquet with side='a' for asks."""
        lob = LOB()
        lob.set_snapshot([], [(101, 8), (102, 4), (103, 2)])

        path = tmp_path / "asks.parquet"
        lob.to_parquet(path, side='a')

        assert path.exists()

    def test_to_csv_both_sides(self, tmp_path):
        """Test to_parquet with side=None for both sides."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)])

        path = tmp_path / "both.parquet"
        lob.to_parquet(path, side=None)

        assert path.exists()

    def test_to_csv_with_nlevels(self, tmp_path):
        """Test to_parquet with nlevels parameter."""
        lob = LOB()
        lob.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)])

        path = tmp_path / "nlevels.parquet"
        lob.to_parquet(path, side='b', nlevels=2)

        assert path.exists()

    def test_to_csv_empty_book(self, tmp_path):
        """Test to_parquet with empty order book."""
        lob = LOB()

        path = tmp_path / "empty.parquet"
        lob.to_parquet(path, side=None)

        assert path.exists()

        """Test to_parquet with float prices and sizes."""
        lob = LOB()
        lob.set_snapshot([(100.5, 10.25), (99.75, 5.5)], [(101.25, 8.75)])

        path = tmp_path / "floats.parquet"
        lob.to_parquet(path, side=None)

        assert path.exists()
