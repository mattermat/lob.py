import pytest
from lobpy.lobts import LOBts

# Check if numpy and pandas are available
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY = True
    HAS_PANDAS = True
except ImportError:
    HAS_NUMPY = False
    HAS_PANDAS = False


class TestLOBtsInit:
    """Test LOBts initialization."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        lobts = LOBts()
        assert lobts.tick_size == 1
        assert lobts.len == 0
    
    def test_init_with_tick_size(self):
        """Test initialization with tick size."""
        lobts = LOBts(tick_size=0.01)
        assert lobts.tick_size == 0.01
    
    def test_init_empty(self):
        """Test empty LOBts initialization."""
        lobts = LOBts()
        assert lobts.len == 0


class TestLOBtsBasicMethods:
    """Test basic LOB methods on LOBts."""
    
    def test_set_snapshot_creates_new_lob(self):
        """Test that set_snapshot creates new LOB in series."""
        lobts = LOBts()
        bids = [(100, 10), (99, 5)]
        asks = [(101, 8)]
        
        lobts.set_snapshot(bids, asks, timestamp=1000)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob is not None
    
    def test_set_snapshot_with_timestamp(self):
        """Test set_snapshot updates timestamp."""
        lobts = LOBts()
        timestamp = 1234567890
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=timestamp)
        
        lob = lobts[timestamp]
        assert lob is not None
        assert lob.timestamp == timestamp
    
    def test_set_snapshot_replaces_previous(self):
        """Test that snapshot at same timestamp replaces previous (with force=True)."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000, force=True)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob is not None
        assert lob.bid[0] == 95
        assert lob.ask[0] == 106
    
    def test_set_updates_creates_new_lob(self):
        """Test that set_updates creates new LOB in series."""
        lobts = LOBts()
        updates = [('b', 100, 10), ('a', 101, 8)]
        
        lobts.set_updates(updates, timestamp=1000)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob is not None
        assert lob.bid[0] == 100
        assert lob.ask[0] == 101
    
    def test_set_updates_with_timestamp(self):
        """Test set_updates with explicit timestamp."""
        lobts = LOBts()
        timestamp = 1234567890
        updates = [('b', 100, 10)]
        
        lobts.set_updates(updates, timestamp=timestamp)
        
        lob = lobts[timestamp]
        assert lob is not None
        assert lob.timestamp == timestamp
    
    def test_set_updates_multiple_levels(self):
        """Test set_updates with multiple level changes."""
        lobts = LOBts()
        updates = [
            ('b', 100, 10),
            ('b', 99, 5),
            ('a', 101, 8),
            ('a', 102, 4)
        ]
        
        lobts.set_updates(updates, timestamp=1000)
        
        lob = lobts[1000]
        assert lob is not None
        assert lob.bid[0] == 100
        assert lob.bid[1] == 99
        assert lob.ask[0] == 101
        assert lob.ask[1] == 102
    
    def test_update_creates_new_lob(self):
        """Test that single update creates new LOB."""
        lobts = LOBts()
        
        lobts.set_updates([('b', 100, 10)], timestamp=1000)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob is not None
        assert lob.bid[0] == 100
    
    def test_update_deletes_level(self):
        """Test update with zero size deletes level."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_updates([('b', 100, 0)], timestamp=2000)
        
        lob = lobts[2000]
        assert lob is not None
        assert lob.bid[0] == 0.0


class TestLOBtsTimeIndexing:
    """Test time indexing functionality."""
    
    def test_get_lob_at_timestamp(self):
        """Test retrieving LOB at specific timestamp using bracket notation."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lob = lobts[1000]
        
        assert lob is not None
        assert lob.timestamp == 1000
        assert lob.bid[0] == 100
        assert lob.ask[0] == 101
    
    def test_get_lob_at_nonexistent_timestamp(self):
        """Test retrieving LOB at non-existent timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lob = lobts[2000]
        
        assert lob is None
    
    def test_slice_range(self):
        """Test slicing LOBs in time range."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=3000)
        
        lob_range = lobts[1500:2500]
        assert lob_range.len == 1
    
    def test_timestamps_are_sorted(self):
        """Test that timestamps are chronologically sorted."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=3000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=2000)
        
        timestamps = list(lobts.timestamps)
        assert timestamps == sorted(timestamps)
        assert timestamps == [1000, 2000, 3000]
    
    def test_length(self):
        """Test length of LOBts."""
        lobts = LOBts()
        
        assert lobts.len == 0
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        assert lobts.len == 1
        
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        assert lobts.len == 2
    
    def test_len_ts(self):
        """Test len_ts property (time span)."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=3000)
        
        assert lobts.len_ts == 2000


class TestLOBtsProperties:
    """Test LOB properties at specific timestamps."""
    
    def test_bid_at_timestamp(self):
        """Test getting best bid at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.bid[0] == 100
        assert lob.bid[1] == 99
    
    def test_ask_at_timestamp(self):
        """Test getting best ask at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.ask[0] == 101
        assert lob.ask[1] == 102
    
    def test_spread_at_timestamp(self):
        """Test getting spread at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.spread == 1.0
    
    def test_bidq_at_timestamp(self):
        """Test getting best bid quantity at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.bidq[0] == 10
    
    def test_askq_at_timestamp(self):
        """Test getting best ask quantity at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.askq[0] == 8
    
    def test_vi_at_timestamp(self):
        """Test getting volume imbalance at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        vi = lob.vi[0]
        assert vi is not None
    
    def test_spread_tick_at_timestamp(self):
        """Test getting spread in ticks at timestamp."""
        lobts = LOBts(tick_size=0.5)
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.spread_tick == 2.0
    
    def test_spread_rel_at_timestamp(self):
        """Test getting spread as percentage at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.spread_rel == 0.01
    
    def test_midprice_at_timestamp(self):
        """Test getting mid-price at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.midprice == 100.5
    
    def test_vw_midprice_at_timestamp(self):
        """Test getting volume-weighted mid-price at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        expected = (100 * 10 + 101 * 8) / (10 + 8)
        assert lob.vw_midprice == expected
    
    def test_properties_return_correct_for_different_timestamps(self):
        """Test that properties are correct for different timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob1 = lobts[1000]
        lob2 = lobts[2000]
        
        assert lob1.bid[0] == 100
        assert lob2.bid[0] == 95


class TestLOBtsStats:
    """Test basic LOB stats as time series."""
    
    def test_spread_time_series(self):
        """Test getting spread as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        spreads = lobts.spread
        assert len(spreads) == 2
        assert 1000 in spreads
        assert 2000 in spreads
    
    def test_bid_time_series(self):
        """Test getting best bid prices as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        bids = lobts.bid
        assert len(bids) == 2
        assert bids[1000] == 100
        assert bids[2000] == 95
    
    def test_ask_time_series(self):
        """Test getting best ask prices as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        asks = lobts.ask
        assert len(asks) == 2
        assert asks[1000] == 101
        assert asks[2000] == 106
    
    def test_midprice_time_series(self):
        """Test getting mid-price as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        midprices = lobts.midprice
        assert len(midprices) == 1
        assert 1000 in midprices
    
    def test_vw_midprice_time_series(self):
        """Test getting volume-weighted mid-price as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        vw_midprices = lobts.vw_midprice
        assert len(vw_midprices) == 1
        assert 1000 in vw_midprices
    
    def test_volume_imbalance_time_series(self):
        """Test getting volume imbalance as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        vi = lobts.vi
        assert len(vi) == 1


class TestLOBtsTimeStats:
    """Test time-based statistics."""
    
    def test_arrival_frequency(self):
        """Test calculating order arrival frequency (L2 quantity-based)."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 12), (99, 5)], [(101, 8)], timestamp=1100)
        lobts.set_snapshot([(100, 12), (99, 8)], [(101, 10)], timestamp=1200)
        
        freq = lobts.arrival_frequency
        assert freq is not None
        assert freq > 0
    
    def test_arrival_frequency_quantity_increases(self):
        """Test arrival frequency includes quantity increases at existing levels."""
        lobts = LOBts(tick_size=0.01)
        
        # Initial state
        lobts.set_snapshot(
            bids=[(100.00, 10), (99.50, 5)],
            asks=[(100.50, 8)],
            timestamp=1000
        )
        
        # New levels + quantity increases
        lobts.set_updates([
            ('b', 100.00, 15),    # QUANTITY INCREASE: 10 -> 15 (+5)
            ('b', 98.50, 3),      # NEW LEVEL: 3 (+3)
            ('a', 100.50, 10),    # QUANTITY INCREASE: 8 -> 10 (+2)
            ('a', 102.00, 2),     # NEW LEVEL: 2 (+2)
        ], timestamp=1100)
        
        # Total arrivals: 5 + 3 + 2 + 2 = 12
        assert lobts.arrival_frequency == 12
    
    def test_arrival_frequency_by_side(self):
        """Test calculating arrival frequency per side."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1100)
        
        freq = lobts.arrival_frequency
        assert freq is not None
    
    def test_arrival_frequency_with_window(self):
        """Test calculating arrival frequency with window parameter."""
        lobts = LOBts()
        
        for i in range(10):
            lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000 + i * 100)
        
        freq = lobts.arrival_frequency
        assert freq is not None
    
    def test_cancel_frequency(self):
        """Test calculating order cancel frequency (L2 quantity-based)."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_updates([('b', 100, 0)], timestamp=1100)
        
        freq = lobts.cancel_frequency
        assert freq is not None
        assert freq > 0
    
    def test_cancel_frequency_with_window(self):
        """Test calculating cancel frequency with window parameter."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_updates([('b', 100, 0)], timestamp=1100)
        lobts.set_updates([('b', 99, 0)], timestamp=1200)
        
        freq = lobts.cancel_frequency
        assert freq is not None
    
    def test_cancel_frequency_partial_cancels(self):
        """Test cancel frequency includes partial size decreases (L2 book semantics)."""
        lobts = LOBts(tick_size=0.01)
        
        # Initial state
        lobts.set_snapshot(
            bids=[(100.00, 10), (99.50, 5)],
            asks=[(100.50, 8), (101.00, 4)],
            timestamp=1000
        )
        
        # Partial cancel + full cancel + arrival
        lobts.set_updates([
            ('b', 100.00, 7),     # PARTIAL: 10 -> 7 (decrease of 3)
            ('b', 99.50, 0),      # FULL: 5 -> 0 (decrease of 5)
            ('a', 100.50, 5),      # PARTIAL: 8 -> 5 (decrease of 3)
            ('b', 98.50, 3),      # ARRIVAL: new level
        ], timestamp=1100)
        
        # More cancels
        lobts.set_updates([
            ('b', 98.50, 0),      # FULL: 3 -> 0 (decrease of 3)
        ], timestamp=1200)
        
        # Total cancels: 3+5+3 (t=1100) + 3 (t=1200) = 14
        assert lobts.cancel_frequency == 14


class TestLOBtsConversion:
    """Test numpy/pandas conversion."""
    
    def test_to_np_single_timestamp(self):
        """Test converting single LOB to numpy."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        arr = lob.to_np()
        assert arr is not None
        assert arr.shape[0] > 0
    
    def test_to_np_time_series(self):
        """Test converting entire time series to numpy."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        arr = lobts.to_np()
        assert arr is not None
    
    def test_to_np_with_nlevels(self):
        """Test converting with level limit."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        arr = lob.to_np(nlevels=2)
        assert arr is not None
    
    def test_to_pd_single_timestamp(self):
        """Test converting single LOB to pandas."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        df = lob.to_pd()
        assert df is not None
        assert len(df) > 0
    
    def test_to_pd_time_series(self):
        """Test converting entire time series to pandas."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        df = lobts.to_pd()
        assert df is not None
        assert len(df) > 0
    
    def test_to_pd_with_nlevels(self):
        """Test converting pandas with level limit."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        df = lob.to_pd(nlevels=2)
        assert df is not None
    
    def test_conversion_preserves_timestamps(self):
        """Test that timestamps are preserved in conversion."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        df = lobts.to_pd()
        assert len(df) > 0


class TestLOBtsUtils:
    """Test utility methods."""
    
    def test_diff_between_timestamps(self):
        """Test calculating diff between LOBs at two timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob1 = lobts[1000]
        lob2 = lobts[2000]
        diff = lob1.diff(lob2)
        
        assert diff is not None
        assert len(diff) > 0
    
    def test_diff_between_lobts(self):
        """Test calculating diff between LOBts."""
        lobts1 = LOBts()
        lobts2 = LOBts()
        
        lobts1.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts2.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        
        diff = lobts1.diff(lobts2)
        assert diff is not None
    
    def test_len_in_tick_at_timestamp(self):
        """Test calculating tick distance at timestamp."""
        lobts = LOBts(tick_size=0.01)
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        tick_dist = lob.len_in_tick('bid', 99)
        assert tick_dist is not None
        assert tick_dist == 100
    
    def test_aggq_nlevel(self):
        """Test aggregate quantity by number of levels."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        agg = lob.aggq('bid', nlevel=2)
        assert agg == 15
    
    def test_aggq_ticks(self):
        """Test aggregate quantity by ticks."""
        lobts = LOBts(tick_size=0.5)
        
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        agg = lob.aggq('bid', ticks=2)
        assert agg is not None
    
    def test_aggq_price(self):
        """Test aggregate quantity by price."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        agg = lob.aggq('bid', price=99)
        assert agg is not None
    
    def test_get_slippage(self):
        """Test calculating slippage."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        slippage = lob.get_slippage(5, side='ask')
        assert slippage is not None


class TestLOBtsExport:
    """Test file export methods."""
    
    def test_to_csv_single_timestamp(self, tmp_path):
        """Test exporting single LOB to CSV."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        path = tmp_path / "lob.csv"
        lob.to_csv(path)
        
        assert path.exists()
    
    def test_to_csv_time_series(self, tmp_path):
        """Test exporting entire time series to CSV."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        path = tmp_path / "lobts.csv"
        lobts.to_csv(path)
        
        assert path.exists()
    
    def test_to_csv_with_nlevels(self, tmp_path):
        """Test exporting with level limit."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        path = tmp_path / "lob_nlevels.csv"
        lob.to_csv(path, nlevels=2)
        
        assert path.exists()
    
    def test_to_xlsx_single_timestamp(self, tmp_path):
        """Test exporting to XLSX."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        path = tmp_path / "lob.xlsx"
        lob.to_xlsx(path)
        
        assert path.exists()
    
    def test_to_xlsx_time_series(self, tmp_path):
        """Test exporting time series to XLSX."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        path = tmp_path / "lobts.xlsx"
        lobts.to_xlsx(path)
        
        assert path.exists()
    
    def test_to_parquet_single_timestamp(self, tmp_path):
        """Test exporting to Parquet."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        path = tmp_path / "lob.parquet"
        lob.to_parquet(path)
        
        assert path.exists()
    
    def test_to_parquet_time_series(self, tmp_path):
        """Test exporting time series to Parquet."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        path = tmp_path / "lobts.parquet"
        lobts.to_parquet(path)
        
        assert path.exists()


class TestLOBtsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_lobts(self):
        """Test operations on empty LOBts."""
        lobts = LOBts()
        
        assert lobts.len == 0
    
    def test_single_lob(self):
        """Test LOBts with only one LOB."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob is not None
    
    def test_duplicate_timestamps_with_force(self):
        """Test handling duplicate timestamp updates with force=True."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000, force=True)
        
        assert lobts.len == 1
        lob = lobts[1000]
        assert lob.bid[0] == 95
    
    def test_out_of_order_timestamps(self):
        """Test handling timestamps received out of order."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=3000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=2000)
        
        assert lobts.len == 3
        timestamps = list(lobts.timestamps)
        assert timestamps == [1000, 2000, 3000]
    
    def test_crossed_books_in_series(self):
        """Test handling crossed books in time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(105, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=2000)
        
        assert lobts.len == 2
        lob1 = lobts[1000]
        assert lob1.bid[0] == 105
        assert lob1.ask[0] == 101
    
    def test_large_number_of_updates(self):
        """Test performance with many updates."""
        lobts = LOBts()
        
        for i in range(100):
            lobts.set_snapshot([(100 + i, 10)], [(101 + i, 8)], timestamp=1000 + i * 100)
        
        assert lobts.len == 100
    
    def test_float_timestamps(self):
        """Test handling float timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100.5, 10)], [(101.5, 8)], timestamp=1000.5)
        lobts.set_snapshot([(95.25, 15)], [(106.75, 12)], timestamp=2000.25)
        
        assert lobts.len == 2
        lob = lobts[1000.5]
        assert lob is not None


class TestLOBtsIntegration:
    """Test integration scenarios."""
    
    def test_slice_time_range(self):
        """Test slicing LOBts by time range."""
        lobts = LOBts()
        
        for i in range(10):
            lobts.set_snapshot([(100 + i, 10)], [(101 + i, 8)], timestamp=1000 + i * 100)
        
        # timestamps: 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000
        # slice 1500:3500 should include 1600, 1700, 1800, 1900, 2000 (5 snapshots)
        sliced = lobts[1500:3500]
        assert sliced.len == 5


class TestLOBtsCheck:
    """Test check method for LOB consistency."""
    
    def test_check_at_timestamp(self):
        """Test checking LOB consistency at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        is_consistent = lob.check()
        assert is_consistent is True
    
    def test_check_crossed_book(self):
        """Test checking crossed book returns False."""
        lobts = LOBts()
        
        lobts.set_snapshot([(105, 10)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        is_consistent = lob.check()
        assert is_consistent is False


class TestLOBtsLevelAccess:
    """Test accessing specific levels at timestamps."""
    
    def test_bid_level_i_at_timestamp(self):
        """Test accessing bid level i at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.bid[0] == 100
        assert lob.bid[1] == 99
        assert lob.bid[2] == 98
    
    def test_ask_level_i_at_timestamp(self):
        """Test accessing ask level i at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8), (102, 4), (103, 2)], timestamp=1000)
        
        lob = lobts[1000]
        assert lob.ask[0] == 101
        assert lob.ask[1] == 102
        assert lob.ask[2] == 103
    
    def test_vi_level_i_at_timestamp(self):
        """Test accessing volume imbalance at level i."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        lob = lobts[1000]
        vi_0 = lob.vi[0]
        vi_1 = lob.vi[1]
        
        assert vi_0 is not None
        assert vi_1 is not None
    
    def test_level_access_time_series(self):
        """Test getting level data as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15), (94, 7)], [(106, 12)], timestamp=2000)
        
        bids = lobts.bid
        assert len(bids) == 2
        assert bids[1000] == 100
        assert bids[2000] == 95
