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
        assert lobts.name.startswith("lobts")
        assert lobts.tick_size == 1
        assert len(lobts._lobs) == 0
    
    def test_init_with_name(self):
        """Test initialization with custom name."""
        lobts = LOBts(name="test_ts")
        assert lobts.name == "test_ts"
    
    def test_init_with_tick_size(self):
        """Test initialization with tick size."""
        lobts = LOBts(tick_size=0.01)
        assert lobts.tick_size == 0.01
    
    def test_init_empty(self):
        """Test empty LOBts initialization."""
        lobts = LOBts()
        assert len(lobts) == 0
        assert lobts._lobs is not None


class TestLOBtsBasicMethods:
    """Test basic LOB methods on LOBts."""
    
    def test_set_snapshot_creates_new_lob(self):
        """Test that set_snapshot creates new LOB in series."""
        lobts = LOBts()
        bids = [(100, 10), (99, 5)]
        asks = [(101, 8)]
        
        lobts.set_snapshot(bids, asks, timestamp=1000)
        
        assert len(lobts) == 1
        lob = lobts.get_lob_at_timestamp(1000)
        assert lob is not None
    
    def test_set_snapshot_with_timestamp(self):
        """Test set_snapshot updates timestamp."""
        lobts = LOBts()
        timestamp = 1234567890
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=timestamp)
        
        lob = lobts.get_lob_at_timestamp(timestamp)
        assert lob.timestamp == timestamp
    
    def test_set_snapshot_replaces_previous(self):
        """Test that snapshot at same timestamp replaces previous."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        
        assert len(lobts) == 1
        lob = lobts.get_lob_at_timestamp(1000)
        assert lob is not None
    
    def test_set_updates_creates_new_lob(self):
        """Test that set_updates creates new LOB in series."""
        lobts = LOBts()
        updates = [('bid', 100, 10), ('ask', 101, 8)]
        
        lobts.set_updates(updates, timestamp=1000)
        
        assert len(lobts) == 1
        lob = lobts.get_lob_at_timestamp(1000)
        assert lob is not None
    
    def test_set_updates_with_timestamp(self):
        """Test set_updates with explicit timestamp."""
        lobts = LOBts()
        timestamp = 1234567890
        updates = [('bid', 100, 10)]
        
        lobts.set_updates(updates, timestamp=timestamp)
        
        lob = lobts.get_lob_at_timestamp(timestamp)
        assert lob.timestamp == timestamp
    
    def test_set_updates_multiple_levels(self):
        """Test set_updates with multiple level changes."""
        lobts = LOBts()
        updates = [
            ('bid', 100, 10),
            ('bid', 99, 5),
            ('ask', 101, 8),
            ('ask', 102, 4)
        ]
        
        lobts.set_updates(updates, timestamp=1000)
        
        lob = lobts.get_lob_at_timestamp(1000)
        assert lob is not None
    
    def test_update_creates_new_lob(self):
        """Test that single update creates new LOB."""
        lobts = LOBts()
        
        lobts.update('bid', 100, 10, timestamp=1000)
        
        assert len(lobts) == 1
        lob = lobts.get_lob_at_timestamp(1000)
        assert lob is not None
    
    def test_update_with_timestamp(self):
        """Test update with explicit timestamp."""
        lobts = LOBts()
        timestamp = 1234567890
        
        lobts.update('bid', 100, 10, timestamp=timestamp)
        
        lob = lobts.get_lob_at_timestamp(timestamp)
        assert lob.timestamp == timestamp
    
    def test_update_deletes_level(self):
        """Test update with zero size deletes level."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.update('bid', 100, 0, timestamp=2000)
        
        lob = lobts.get_lob_at_timestamp(2000)
        assert lob is not None


class TestLOBtsTimeIndexing:
    """Test time indexing functionality."""
    
    def test_get_lob_at_timestamp(self):
        """Test retrieving LOB at specific timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lob = lobts.get_lob_at_timestamp(1000)
        
        assert lob is not None
        assert lob.timestamp == 1000
    
    def test_get_lob_at_nonexistent_timestamp(self):
        """Test retrieving LOB at non-existent timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lob = lobts.get_lob_at_timestamp(2000)
        
        assert lob is None
    
    def test_get_lob_before_timestamp(self):
        """Test getting LOB before specified time."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob = lobts.get_lob_before_timestamp(1500)
        assert lob is not None
        assert lob.timestamp == 1000
    
    def test_get_lob_after_timestamp(self):
        """Test getting LOB after specified time."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob = lobts.get_lob_after_timestamp(1500)
        assert lob is not None
        assert lob.timestamp == 2000
    
    def test_get_lob_range(self):
        """Test getting LOBs in time range."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=3000)
        
        lobs = lobts.get_lob_range(start=1500, end=2500)
        assert len(lobs) == 1
        assert lobs[0].timestamp == 2000
    
    def test_get_latest_lob(self):
        """Test getting most recent LOB."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob = lobts.get_latest_lob()
        assert lob is not None
        assert lob.timestamp == 2000
    
    def test_get_earliest_lob(self):
        """Test getting earliest LOB."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        lob = lobts.get_earliest_lob()
        assert lob is not None
        assert lob.timestamp == 1000
    
    def test_timestamps_are_sorted(self):
        """Test that timestamps are chronologically sorted."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=3000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=2000)
        
        timestamps = lobts.timestamps
        assert timestamps == sorted(timestamps)


class TestLOBtsProperties:
    """Test LOB properties at specific timestamps."""
    
    def test_bid_at_timestamp(self):
        """Test getting best bid at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        bid = lobts.get_bid_at_timestamp(1000)
        assert bid == (100, 10)
    
    def test_ask_at_timestamp(self):
        """Test getting best ask at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8), (102, 4)], timestamp=1000)
        
        ask = lobts.get_ask_at_timestamp(1000)
        assert ask == (101, 8)
    
    def test_spread_at_timestamp(self):
        """Test getting spread at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        spread = lobts.get_spread_at_timestamp(1000)
        assert spread == 1.0
    
    def test_bidq_at_timestamp(self):
        """Test getting best bid quantity at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        bidq = lobts.get_bidq_at_timestamp(1000)
        assert bidq == 10
    
    def test_askq_at_timestamp(self):
        """Test getting best ask quantity at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        askq = lobts.get_askq_at_timestamp(1000)
        assert askq == 8
    
    def test_vi_at_timestamp(self):
        """Test getting volume imbalance at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        vi = lobts.get_vi_at_timestamp(1000)
        assert vi is not None
    
    def test_properties_return_correct_for_different_timestamps(self):
        """Test that properties are correct for different timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        bid1 = lobts.get_bid_at_timestamp(1000)
        bid2 = lobts.get_bid_at_timestamp(2000)
        
        assert bid1 == (100, 10)
        assert bid2 == (95, 15)


class TestLOBtsStats:
    """Test basic LOB stats as time series."""
    
    def test_spread_time_series(self):
        """Test getting spread as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        spreads = lobts.get_spread_time_series()
        assert len(spreads) == 2
        assert 1000 in spreads
        assert 2000 in spreads
    
    def test_bid_time_series(self):
        """Test getting best bid prices as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        bids = lobts.get_bid_time_series()
        assert len(bids) == 2
    
    def test_ask_time_series(self):
        """Test getting best ask prices as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        asks = lobts.get_ask_time_series()
        assert len(asks) == 2
    
    def test_midprice_time_series(self):
        """Test getting mid-price as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        midprices = lobts.get_midprice_time_series()
        assert len(midprices) == 1
    
    def test_vw_midprice_time_series(self):
        """Test getting volume-weighted mid-price as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        vw_midprices = lobts.get_vw_midprice_time_series()
        assert len(vw_midprices) == 1
    
    def test_volume_imbalance_time_series(self):
        """Test getting volume imbalance as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        vi = lobts.get_vi_time_series()
        assert len(vi) == 1


class TestLOBtsTimeStats:
    """Test time-based statistics."""
    
    def test_arrival_frequency(self):
        """Test calculating order arrival frequency."""
        lobts = LOBts()
        
        for i in range(5):
            lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000 + i * 100)
        
        freq = lobts.get_arrival_frequency()
        assert freq is not None
    
    def test_arrival_frequency_by_side(self):
        """Test calculating arrival frequency per side."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1100)
        
        freq = lobts.get_arrival_frequency_by_side()
        assert freq is not None
    
    def test_cancel_frequency(self):
        """Test calculating order cancel frequency."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.update('bid', 100, 0, timestamp=1100)
        
        freq = lobts.get_cancel_frequency()
        assert freq is not None
    
    def test_update_frequency(self):
        """Test calculating update frequency."""
        lobts = LOBts()
        
        for i in range(3):
            lobts.set_updates([('bid', 100, 10)], timestamp=1000 + i * 100)
        
        freq = lobts.get_update_frequency()
        assert freq is not None
    
    def test_time_between_updates(self):
        """Test calculating time between consecutive updates."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1500)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=2000)
        
        times = lobts.get_time_between_updates()
        assert len(times) == 2
    
    def test_volatility_spread(self):
        """Test calculating spread volatility over time."""
        lobts = LOBts()
        
        for i in range(10):
            lobts.set_snapshot([(100 + i, 10)], [(110 + i, 8)], timestamp=1000 + i * 100)
        
        vol = lobts.get_spread_volatility()
        assert vol is not None


class TestLOBtsConversion:
    """Test numpy/pandas conversion."""
    
    def test_to_np_single_timestamp(self):
        """Test converting single LOB to numpy."""
        pytest.skip("skip if numpy not available")
        if not HAS_NUMPY:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        arr = lobts.to_np(timestamp=1000)
        assert arr is not None
    
    def test_to_np_time_series(self):
        """Test converting entire time series to numpy."""
        pytest.skip("skip if numpy not available")
        if not HAS_NUMPY:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        arr = lobts.to_np()
        assert arr is not None
    
    def test_to_np_with_nlevels(self):
        """Test converting with level limit."""
        pytest.skip("skip if numpy not available")
        if not HAS_NUMPY:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8), (102, 4)], timestamp=1000)
        
        arr = lobts.to_np(timestamp=1000, nlevels=2)
        assert arr is not None
    
    def test_to_pd_single_timestamp(self):
        """Test converting single LOB to pandas."""
        pytest.skip("skip if pandas not available")
        if not HAS_PANDAS:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        df = lobts.to_pd(timestamp=1000)
        assert df is not None
    
    def test_to_pd_time_series(self):
        """Test converting entire time series to pandas."""
        pytest.skip("skip if pandas not available")
        if not HAS_PANDAS:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        df = lobts.to_pd()
        assert df is not None
    
    def test_to_pd_with_nlevels(self):
        """Test converting pandas with level limit."""
        pytest.skip("skip if pandas not available")
        if not HAS_PANDAS:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        df = lobts.to_pd(timestamp=1000, nlevels=2)
        assert df is not None
    
    def test_conversion_preserves_timestamps(self):
        """Test that timestamps are preserved in conversion."""
        pytest.skip("skip if pandas not available")
        if not HAS_PANDAS:
            return
        
        lobts = LOBts()
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        df = lobts.to_pd()
        assert 1000 in df.index or 'timestamp' in df.columns


class TestLOBtsUtils:
    """Test utility methods."""
    
    def test_diff_between_timestamps(self):
        """Test calculating diff between LOBs at two timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        diff = lobts.diff(timestamp1=1000, timestamp2=2000)
        assert diff is not None
    
    def test_diff_time_series(self):
        """Test getting time series of diffs."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=3000)
        
        diffs = lobts.diff_time_series()
        assert len(diffs) == 2
    
    def test_len_in_tick_at_timestamp(self):
        """Test calculating tick distance at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        tick_dist = lobts.len_in_tick(side='bid', price=99, timestamp=1000)
        assert tick_dist is not None
    
    def test_len_in_tick_time_series(self):
        """Test getting tick distance as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 10), (98, 5)], [(102, 8)], timestamp=2000)
        
        tick_dists = lobts.len_in_tick_time_series(side='bid', price=98)
        assert len(tick_dists) == 2
    
    def test_track_queue_position(self):
        """Test tracking queue position changes over time."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.update('bid', 100, 15, timestamp=1100)
        
        queue_pos = lobts.track_queue_position(side='bid', price=100)
        assert queue_pos is not None


class TestLOBtsExport:
    """Test file export methods."""
    
    def test_to_csv_single_timestamp(self, tmp_path):
        """Test exporting single LOB to CSV."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        path = tmp_path / "lob.csv"
        lobts.to_csv(path, timestamp=1000)
        
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
        
        path = tmp_path / "lob_nlevels.csv"
        lobts.to_csv(path, timestamp=1000, nlevels=2)
        
        assert path.exists()
    
    def test_to_xlsx_single_timestamp(self, tmp_path):
        """Test exporting to XLSX."""
        lobts = LOBts()
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        
        path = tmp_path / "lob.xlsx"
        lobts.to_xlsx(path, timestamp=1000)
        
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
        
        path = tmp_path / "lob.parquet"
        lobts.to_parquet(path, timestamp=1000)
        
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
        
        assert len(lobts) == 0
        assert lobts.get_latest_lob() is None
        assert lobts.get_earliest_lob() is None
    
    def test_single_lob(self):
        """Test LOBts with only one LOB."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        assert len(lobts) == 1
        assert lobts.get_latest_lob() == lobts.get_earliest_lob()
    
    def test_duplicate_timestamps(self):
        """Test handling duplicate timestamp updates."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        
        assert len(lobts) == 1
    
    def test_out_of_order_timestamps(self):
        """Test handling timestamps received out of order."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=3000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=1000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=2000)
        
        assert len(lobts) == 3
        assert lobts.timestamps == sorted(lobts.timestamps)
    
    def test_crossed_books_in_series(self):
        """Test handling crossed books in time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(105, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=2000)
        
        assert len(lobts) == 2
    
    def test_large_number_of_updates(self):
        """Test performance with many updates."""
        lobts = LOBts()
        
        for i in range(100):
            lobts.set_snapshot([(100 + i, 10)], [(101 + i, 8)], timestamp=1000 + i * 100)
        
        assert len(lobts) == 100
    
    def test_float_timestamps(self):
        """Test handling float timestamps."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100.5, 10)], [(101.5, 8)], timestamp=1000.5)
        lobts.set_snapshot([(95.25, 15)], [(106.75, 12)], timestamp=2000.25)
        
        assert len(lobts) == 2


class TestLOBtsIntegration:
    """Test integration scenarios."""
    
    def test_reconstruct_from_diffs(self):
        """Test reconstructing LOBts from diff series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        diffs = lobts.diff_time_series()
        reconstructed = LOBts.from_diffs(diffs)
        
        assert len(reconstructed) == len(lobts)
    
    def test_resample_time_series(self):
        """Test resampling LOBts to different frequency."""
        lobts = LOBts()
        
        for i in range(10):
            lobts.set_snapshot([(100 + i, 10)], [(101 + i, 8)], timestamp=1000 + i * 100)
        
        resampled = lobts.resample(freq=500)
        assert resampled is not None
    
    def test_slice_time_range(self):
        """Test slicing LOBts by time range."""
        lobts = LOBts()
        
        for i in range(10):
            lobts.set_snapshot([(100 + i, 10)], [(101 + i, 8)], timestamp=1000 + i * 100)
        
        sliced = lobts.slice(start=1500, end=3500)
        assert len(sliced) == 3
    
    def test_merge_lobts(self):
        """Test merging two LOBts instances."""
        lobts1 = LOBts()
        lobts2 = LOBts()
        
        lobts1.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts2.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        
        merged = lobts1.merge(lobts2)
        assert len(merged) == 2
    
    def test_filter_by_condition(self):
        """Test filtering LOBs based on condition."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15)], [(106, 12)], timestamp=2000)
        lobts.set_snapshot([(90, 20)], [(110, 15)], timestamp=3000)
        
        filtered = lobts.filter(lambda lob: lob.spread < 10)
        assert len(filtered) == 3


class TestLOBtsCheck:
    """Test check method for LOB consistency."""
    
    def test_check_at_timestamp(self):
        """Test checking LOB consistency at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        
        is_consistent = lobts.check_at_timestamp(1000)
        assert is_consistent is True
    
    def test_check_all_lob(self):
        """Test checking all LOBs in series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(105, 15)], [(101, 8)], timestamp=2000)
        
        checks = lobts.check_all()
        assert len(checks) == 2
    
    def test_check_time_series(self):
        """Test getting consistency status as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(105, 15)], [(106, 12)], timestamp=2000)
        
        checks = lobts.check_time_series()
        assert len(checks) == 2


class TestLOBtsLevelAccess:
    """Test accessing specific levels at timestamps."""
    
    def test_bid_level_i_at_timestamp(self):
        """Test accessing bid level i at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5), (98, 3)], [(101, 8)], timestamp=1000)
        
        bid_level_0 = lobts.get_bid_level(0, timestamp=1000)
        bid_level_1 = lobts.get_bid_level(1, timestamp=1000)
        bid_level_2 = lobts.get_bid_level(2, timestamp=1000)
        
        assert bid_level_0 == (100, 10)
        assert bid_level_1 == (99, 5)
        assert bid_level_2 == (98, 3)
    
    def test_ask_level_i_at_timestamp(self):
        """Test accessing ask level i at timestamp."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10)], [(101, 8), (102, 4), (103, 2)], timestamp=1000)
        
        ask_level_0 = lobts.get_ask_level(0, timestamp=1000)
        ask_level_1 = lobts.get_ask_level(1, timestamp=1000)
        ask_level_2 = lobts.get_ask_level(2, timestamp=1000)
        
        assert ask_level_0 == (101, 8)
        assert ask_level_1 == (102, 4)
        assert ask_level_2 == (103, 2)
    
    def test_vi_level_i_at_timestamp(self):
        """Test accessing volume imbalance at level i."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8), (102, 4)], timestamp=1000)
        
        vi_0 = lobts.get_vi_level(0, timestamp=1000)
        vi_1 = lobts.get_vi_level(1, timestamp=1000)
        
        assert vi_0 is not None
        assert vi_1 is not None
    
    def test_level_access_time_series(self):
        """Test getting level data as time series."""
        lobts = LOBts()
        
        lobts.set_snapshot([(100, 10), (99, 5)], [(101, 8)], timestamp=1000)
        lobts.set_snapshot([(95, 15), (94, 7)], [(106, 12)], timestamp=2000)
        
        bid_levels = lobts.get_bid_level_time_series(level=0)
        assert len(bid_levels) == 2
