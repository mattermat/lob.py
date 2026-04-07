import numpy as np

from lobpy.lob import LOB
from lobpy.lobts import LOBts


class TestLazyConstruction:
    """Test 1: LOBts(mode='lazy') creates the right empty internal structures."""

    def test_lazy_mode_attribute(self):
        lobts = LOBts(mode="lazy")
        assert lobts.mode == "lazy"

    def test_delta_log_is_empty_structured_array(self):
        lobts = LOBts(mode="lazy")
        assert hasattr(lobts, "_delta_log")
        assert isinstance(lobts._delta_log, np.ndarray)
        assert lobts._delta_log.dtype.names == ("ts", "side", "price", "qty")
        assert len(lobts._delta_log) == 0

    def test_ckpt_ts_is_empty_int64_array(self):
        lobts = LOBts(mode="lazy")
        assert hasattr(lobts, "_ckpt_ts")
        assert isinstance(lobts._ckpt_ts, np.ndarray)
        assert lobts._ckpt_ts.dtype == np.int64
        assert len(lobts._ckpt_ts) == 0

    def test_ckpts_is_empty_dict(self):
        lobts = LOBts(mode="lazy")
        assert hasattr(lobts, "_ckpts")
        assert isinstance(lobts._ckpts, dict)
        assert len(lobts._ckpts) == 0

    def test_cache_is_empty_ordered_dict(self):
        from collections import OrderedDict

        lobts = LOBts(mode="lazy")
        assert hasattr(lobts, "_cache")
        assert isinstance(lobts._cache, OrderedDict)
        assert len(lobts._cache) == 0

    def test_lobs_not_populated(self):
        lobts = LOBts(mode="lazy")
        assert not hasattr(lobts, "_lobs")


class TestLazySetSnapshot:
    """Test 2: set_snapshot stores a checkpoint correctly."""

    def test_single_snapshot_stores_ckpt(self):
        lobts = LOBts(mode="lazy")
        bids = [(100.0, 10.0), (99.0, 5.0)]
        asks = [(101.0, 8.0), (102.0, 4.0)]
        lobts.set_snapshot(bids, asks, timestamp=1000)

        assert 1000 in lobts._ckpts
        assert len(lobts._ckpt_ts) == 1
        assert lobts._ckpt_ts[0] == 1000

    def test_snapshot_stores_bids_as_numpy_desc(self):
        lobts = LOBts(mode="lazy")
        bids = [(99.0, 5.0), (100.0, 10.0)]
        asks = [(102.0, 4.0), (101.0, 8.0)]
        lobts.set_snapshot(bids, asks, timestamp=1000)

        bids_arr, asks_arr = lobts._ckpts[1000]
        assert isinstance(bids_arr, np.ndarray)
        assert bids_arr.shape == (2, 2)
        assert bids_arr.dtype == np.float64
        assert bids_arr[0, 0] == 100.0
        assert bids_arr[0, 1] == 10.0
        assert bids_arr[1, 0] == 99.0
        assert bids_arr[1, 1] == 5.0

    def test_snapshot_stores_asks_as_numpy_asc(self):
        lobts = LOBts(mode="lazy")
        bids = [(100.0, 10.0)]
        asks = [(103.0, 2.0), (101.0, 8.0)]
        lobts.set_snapshot(bids, asks, timestamp=1000)

        _, asks_arr = lobts._ckpts[1000]
        assert asks_arr[0, 0] == 101.0
        assert asks_arr[1, 0] == 103.0

    def test_multiple_snapshots(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)

        assert len(lobts._ckpts) == 2
        assert len(lobts._ckpt_ts) == 2
        assert set(lobts._ckpts.keys()) == {1000, 2000}

    def test_ckpt_ts_sorted(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=3000)
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=2000)

        assert list(lobts._ckpt_ts) == [1000, 2000, 3000]


class TestLazySetUpdates:
    """Test 3: set_updates appends to _delta_log with correct dtype/values."""

    def test_single_update_appends_row(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 10.0)], timestamp=1500)

        assert len(lobts._delta_log) == 1
        row = lobts._delta_log[0]
        assert row["ts"] == 1500
        assert row["side"] == 0
        assert row["price"] == 100.0
        assert row["qty"] == 10.0

    def test_ask_update_side_encoding(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("a", 101.0, 8.0)], timestamp=1500)

        row = lobts._delta_log[0]
        assert row["side"] == 1

    def test_multiple_updates_single_call(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates(
            [("b", 100.0, 10.0), ("a", 101.0, 8.0), ("b", 99.0, 5.0)],
            timestamp=1500,
        )

        assert len(lobts._delta_log) == 3
        assert lobts._delta_log[0]["ts"] == 1500
        assert lobts._delta_log[1]["ts"] == 1500
        assert lobts._delta_log[2]["ts"] == 1500

    def test_updates_across_timestamps(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 10.0)], timestamp=1000)
        lobts.set_updates([("a", 101.0, 8.0)], timestamp=2000)

        assert len(lobts._delta_log) == 2
        assert lobts._delta_log[0]["ts"] == 1000
        assert lobts._delta_log[1]["ts"] == 2000

    def test_delete_update_qty_zero(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 0.0)], timestamp=1500)

        row = lobts._delta_log[0]
        assert row["qty"] == 0.0

    def test_long_form_side_accepted(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("bid", 100.0, 10.0)], timestamp=1000)
        lobts.set_updates([("ask", 101.0, 8.0)], timestamp=1001)

        assert lobts._delta_log[0]["side"] == 0
        assert lobts._delta_log[1]["side"] == 1


class TestLazyGetItemExactCheckpoint:
    """Test 4: __getitem__ at exact checkpoint timestamp returns matching LOB."""

    def test_exact_checkpoint_returns_lob(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot(
            [(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0), (102.0, 4.0)], timestamp=1000
        )

        lob = lobts[1000]
        assert lob is not None
        assert isinstance(lob, LOB)
        assert lob.bid[0] == 100.0
        assert lob.bid[1] == 99.0
        assert lob.ask[0] == 101.0
        assert lob.ask[1] == 102.0

    def test_exact_checkpoint_quantities(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0)], timestamp=1000)

        lob = lobts[1000]
        assert lob.bidq[0] == 10.0
        assert lob.bidq[1] == 5.0
        assert lob.askq[0] == 8.0

    def test_exact_checkpoint_with_at(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0)], timestamp=1000)

        lob = lobts[1000]
        assert lob.at("b", 100.0) == 10.0
        assert lob.at("b", 99.0) == 5.0
        assert lob.at("a", 101.0) == 8.0
        assert lob.at("b", 98.0) == 0.0


class TestLazyGetItemBetweenCheckpoints:
    """Test 5: __getitem__ between checkpoints applies deltas correctly."""

    def test_price_update_between_checkpoints(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=1500)

        lob = lobts[1500]
        assert lob is not None
        assert lob.bid[0] == 100.0
        assert lob.bidq[0] == 15.0
        assert lob.ask[0] == 101.0

    def test_new_level_between_checkpoints(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 99.0, 5.0)], timestamp=1500)

        lob = lobts[1500]
        assert lob.bid[0] == 100.0
        assert lob.bid[1] == 99.0
        assert lob.bidq[1] == 5.0

    def test_delete_level_between_checkpoints(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot(
            [(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0), (102.0, 4.0)], timestamp=1000
        )
        lobts.set_updates([("b", 99.0, 0.0)], timestamp=1500)

        lob = lobts[1500]
        assert lob.bid[0] == 100.0
        assert lob.at("b", 99.0) == 0.0

    def test_multiple_deltas_applied(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates(
            [("b", 100.0, 20.0), ("a", 101.0, 4.0), ("a", 103.0, 2.0)], timestamp=1500
        )

        lob = lobts[1500]
        assert lob.bidq[0] == 20.0
        assert lob.askq[0] == 4.0
        assert lob.ask[1] == 103.0


class TestLazyGetItemMultipleCheckpoints:
    """Test 6: with 3+ checkpoints, lookup picks the nearest preceding one."""

    def test_picks_nearest_preceding_checkpoint(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)
        lobts.set_snapshot([(90.0, 20.0)], [(110.0, 15.0)], timestamp=3000)

        lob = lobts[2500]
        assert lob is not None
        assert lob.bid[0] == 95.0
        assert lob.ask[0] == 106.0

    def test_deltas_after_correct_checkpoint(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)
        lobts.set_updates([("b", 95.0, 25.0)], timestamp=2500)

        lob = lobts[2500]
        assert lob.bid[0] == 95.0
        assert lob.bidq[0] == 25.0

    def test_exact_second_checkpoint(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)
        lobts.set_updates([("b", 95.0, 25.0)], timestamp=1500)

        lob = lobts[2000]
        assert lob.bid[0] == 95.0
        assert lob.bidq[0] == 15.0


class TestLazyReconstructionVsEager:
    """Test 7: compare lazy output against eagerly-built LOB."""

    def _build_eager(self):
        lobts = LOBts(mode="delta")
        lobts.set_snapshot(
            [(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0), (102.0, 4.0)], timestamp=1000
        )
        lobts.set_updates([("b", 100.0, 15.0), ("a", 101.0, 0.0)], timestamp=1500)
        lobts.set_snapshot([(95.0, 20.0)], [(105.0, 10.0)], timestamp=2000)
        lobts.set_updates([("b", 95.0, 25.0), ("a", 106.0, 3.0)], timestamp=2500)
        return lobts

    def _build_lazy(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot(
            [(100.0, 10.0), (99.0, 5.0)], [(101.0, 8.0), (102.0, 4.0)], timestamp=1000
        )
        lobts.set_updates([("b", 100.0, 15.0), ("a", 101.0, 0.0)], timestamp=1500)
        lobts.set_snapshot([(95.0, 20.0)], [(105.0, 10.0)], timestamp=2000)
        lobts.set_updates([("b", 95.0, 25.0), ("a", 106.0, 3.0)], timestamp=2500)
        return lobts

    def assert_lobs_equal(self, lob_a, lob_b):
        assert np.array_equal(lob_a._bids, lob_b._bids)
        assert np.array_equal(lob_a._asks, lob_b._asks)

    def test_t1000_matches(self):
        eager, lazy = self._build_eager(), self._build_lazy()
        self.assert_lobs_equal(eager[1000], lazy[1000])

    def test_t1500_matches(self):
        eager, lazy = self._build_eager(), self._build_lazy()
        self.assert_lobs_equal(eager[1500], lazy[1500])

    def test_t2000_matches(self):
        eager, lazy = self._build_eager(), self._build_lazy()
        self.assert_lobs_equal(eager[2000], lazy[2000])

    def test_t2500_matches(self):
        eager, lazy = self._build_eager(), self._build_lazy()
        self.assert_lobs_equal(eager[2500], lazy[2500])


class TestLazyCacheHit:
    """Test 8: accessing the same timestamp twice returns the same cached LOB."""

    def test_same_object_returned(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)

        lob1 = lobts[1000]
        lob2 = lobts[1000]
        assert lob1 is lob2

    def test_reconstructed_timestamp_cached(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=1500)

        lob1 = lobts[1500]
        lob2 = lobts[1500]
        assert lob1 is lob2


class TestLazyCacheEviction:
    """Test 9: inserting more than 32 timestamps causes the oldest to be evicted."""

    def test_cache_max_size_32(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=0)

        for i in range(1, 34):
            lobts.set_updates([("b", 100.0, float(i))], timestamp=i * 100)

        for i in range(33):
            _ = lobts[i * 100]

        assert len(lobts._cache) == 32

    def test_oldest_evicted(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=0)

        for i in range(1, 34):
            lobts.set_updates([("b", 100.0, float(i))], timestamp=i * 100)

        for i in range(33):
            _ = lobts[i * 100]

        assert 0 not in lobts._cache
        assert 100 in lobts._cache
        assert 3200 in lobts._cache

    def test_evicted_item_reconstructed(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=0)

        for i in range(1, 34):
            lobts.set_updates([("b", 100.0, float(i))], timestamp=i * 100)

        lob_first = lobts[0]

        for i in range(1, 34):
            _ = lobts[i * 100]

        lob_again = lobts[0]
        assert lob_again is not lob_first
        assert np.array_equal(lob_again._bids, lob_first._bids)


class TestLazyTimestamps:
    """Test 10: timestamps returns all expected timestamps, sorted."""

    def test_empty(self):
        lobts = LOBts(mode="lazy")
        assert list(lobts.timestamps) == []

    def test_checkpoints_only(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)

        assert list(lobts.timestamps) == [1000, 2000]

    def test_deltas_only(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 10.0)], timestamp=1000)
        lobts.set_updates([("a", 101.0, 8.0)], timestamp=2000)

        ts = list(lobts.timestamps)
        assert 1000 in ts
        assert 2000 in ts

    def test_mixed_checkpoints_and_deltas(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=1500)
        lobts.set_updates([("a", 101.0, 4.0)], timestamp=2000)

        ts = list(lobts.timestamps)
        assert ts == sorted(ts)
        assert 1000 in ts
        assert 1500 in ts
        assert 2000 in ts

    def test_timestamps_sorted_out_of_order_insertion(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=3000)
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=2000)

        ts = list(lobts.timestamps)
        assert ts == sorted(ts)


class TestLazyLen:
    """Test 11: __len__ returns correct count."""

    def test_empty(self):
        lobts = LOBts(mode="lazy")
        assert len(lobts) == 0

    def test_single_checkpoint(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        assert len(lobts) == 1

    def test_checkpoints_and_deltas(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=1500)
        lobts.set_updates([("a", 101.0, 4.0)], timestamp=2000)
        assert len(lobts) == 3

    def test_duplicate_timestamp_counted_once(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 10.0)], timestamp=1000)
        lobts.set_updates([("a", 101.0, 8.0)], timestamp=1000)
        assert len(lobts) == 1


class TestLazySlicing:
    """Test 12: slicing [start:stop] restricts range correctly."""

    def _build(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=1500)
        lobts.set_updates([("a", 101.0, 4.0)], timestamp=2000)
        lobts.set_snapshot([(95.0, 20.0)], [(105.0, 10.0)], timestamp=3000)
        lobts.set_updates([("b", 95.0, 25.0)], timestamp=3500)
        return lobts

    def test_slice_contains_only_in_range_timestamps(self):
        lobts = self._build()
        sliced = lobts[1500:3000]
        ts = list(sliced.timestamps)
        for t in ts:
            assert 1500 <= t <= 3000

    def test_slice_excludes_out_of_range(self):
        lobts = self._build()
        sliced = lobts[1500:3000]
        ts = list(sliced.timestamps)
        assert 1000 not in ts
        assert 3500 not in ts

    def test_slice_is_still_lazy(self):
        lobts = self._build()
        sliced = lobts[1500:3000]
        assert sliced.mode == "lazy"

    def test_slice_deltas_excluded(self):
        lobts = self._build()
        sliced = lobts[2001:2999]
        ts = list(sliced.timestamps)
        assert 1000 not in ts
        assert 1500 not in ts
        assert 2000 not in ts

    def test_slice_preserves_reconstruction(self):
        lobts = self._build()
        sliced = lobts[1500:3000]
        lob = sliced[3000]
        assert lob is not None
        assert lob.bid[0] == 95.0

    def test_open_start_slice(self):
        lobts = self._build()
        sliced = lobts[:2000]
        ts = list(sliced.timestamps)
        for t in ts:
            assert t <= 2000

    def test_open_end_slice(self):
        lobts = self._build()
        sliced = lobts[2000:]
        ts = list(sliced.timestamps)
        for t in ts:
            assert t >= 2000


class TestLazyContains:
    """Test 13: __contains__ works for present/absent timestamps."""

    def test_checkpoint_present(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        assert 1000 in lobts

    def test_delta_present(self):
        lobts = LOBts(mode="lazy")
        lobts.set_updates([("b", 100.0, 10.0)], timestamp=1500)
        assert 1500 in lobts

    def test_absent(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        assert 9999 not in lobts

    def test_empty(self):
        lobts = LOBts(mode="lazy")
        assert 1000 not in lobts


class TestLazyNoCrossContamination:
    """Test 14: modifying a returned LOB does not corrupt checkpoints."""

    def test_modify_returned_lob_does_not_affect_checkpoint(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)

        lob = lobts[1000]
        lob._bids[0, 1] = 999.0

        ckpt_bids, ckpt_asks = lobts._ckpts[1000]
        assert ckpt_bids[0, 1] == 10.0

    def test_fresh_reconstruction_after_mutation(self):
        lobts = LOBts(mode="lazy")
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)

        lob_a = lobts[1000]
        lob_a._bids[0, 1] = 999.0

        lobts._cache.clear()
        lob_b = lobts[1000]
        assert lob_b.bidq[0] == 10.0

    def test_lobs_not_used(self):
        lobts = LOBts(mode="lazy")
        assert not hasattr(lobts, "_lobs")


class TestExistingDeltaModeUnaffected:
    """Test 15: mode='delta' (default) is completely unaffected."""

    def test_delta_mode_still_default(self):
        lobts = LOBts()
        assert lobts.mode == "delta"

    def test_delta_set_snapshot(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        assert lobts[1000] is not None
        assert lobts[1000].bid[0] == 100.0

    def test_delta_set_updates(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=2000)
        assert lobts[2000].bidq[0] == 15.0

    def test_delta_len(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_updates([("b", 100.0, 15.0)], timestamp=2000)
        assert len(lobts) == 2

    def test_delta_contains(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        assert 1000 in lobts
        assert 2000 not in lobts

    def test_delta_timestamps(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=2000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=1000)
        assert list(lobts.timestamps) == [1000, 2000]

    def test_delta_slicing(self):
        lobts = LOBts()
        lobts.set_snapshot([(100.0, 10.0)], [(101.0, 8.0)], timestamp=1000)
        lobts.set_snapshot([(95.0, 15.0)], [(106.0, 12.0)], timestamp=2000)
        lobts.set_snapshot([(90.0, 20.0)], [(110.0, 15.0)], timestamp=3000)

        sliced = lobts[1500:2500]
        assert sliced.len == 1
        assert sliced[2000] is not None

    def test_delta_no_lazy_attributes(self):
        lobts = LOBts()
        assert not hasattr(lobts, "_delta_log")
        assert not hasattr(lobts, "_ckpt_ts")
        assert not hasattr(lobts, "_ckpts")
        assert not hasattr(lobts, "_cache")
