"""
Profiling version of large_input.py.

Strategy
--------
- All non-gueant computations are profiled on the full dataset with cProfile.
- Gueant is profiled on a small time slice (first GUEANT_SLICE_S seconds),
  because tl.gueant.ask/bid on the full dataset never returns in reasonable
  time and a profile dump at the call-site is therefore not feasible.

The slice size is intentionally kept small enough to finish quickly while
still exercising every code path inside _compute_rolling_gueant.

Run from the repo root:
    python examples/profile_large_input.py
"""

import cProfile
import io
import pstats
import time

from lobpy import TL

PATH = "test_data/blofin_BTC-USDT_20260330_194305.parquet"
WINDOW = 30_000_000_000  # 30 s in nanoseconds
GUEANT_SLICE_S = 120  # seconds of data used for gueant profiling


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def profile(label, fn, *, top_n=25):
    """
    Run fn() under cProfile.  Print stats sorted by cumulative time AND by
    total (self) time so both callers and hotspots are visible.
    Returns the function result.
    """
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    result = fn()
    pr.disable()
    elapsed = time.perf_counter() - t0

    header = f"  {label}  [{elapsed:.3f}s]"
    bar = "=" * max(70, len(header) + 4)
    print(f"\n{bar}")
    print(header)
    print(bar)

    for sort_key in ("cumulative", "tottime"):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(top_n)
        print(f"\n--- sorted by {sort_key} ---")
        # strip the leading file-path noise pstats adds
        lines = buf.getvalue().splitlines()
        print("\n".join(lines))

    return result


# ---------------------------------------------------------------------------
# Load
# 59 seconds -> 14 seconds.
# ---------------------------------------------------------------------------

print("Loading data …")
t_load = time.perf_counter()
tl = TL(name="BTC-USDT", tick_size=0.1)
tl.from_parquet(PATH, mode="lazy")
print(f"  loaded in {time.perf_counter() - t_load:.2f}s")

ts_all = tl._lobts.timestamps
ts_start = ts_all[0]
ts_end = ts_all[-1]
duration_s = (ts_end - ts_start) / 1e9

print(f"\nTimeline span : {duration_s:.1f}s  ({ts_start} → {ts_end})")
print(f"LOB snapshots : {len(ts_all):,}")
print(f"Trades        : {len(tl.trades):,}")


# ---------------------------------------------------------------------------
# Full-dataset: non-gueant computations
# ---------------------------------------------------------------------------

print("\n\n" + "#" * 70)
print("# FULL DATASET  –  non-gueant computations")
print("#" * 70)

profile("ohlc('5s')", lambda: tl.ohlc("5s"))
profile("bid_ts() / ask_ts()", lambda: (tl.lob.bid_ts(), tl.lob.ask_ts()))
profile("spread_ts()", lambda: tl.lob.spread_ts())
profile(f"realized_vol(window={WINDOW/1e9:.0f}s)", lambda: tl.realized_vol(window_size=WINDOW))
profile(f"vpin(window={WINDOW/1e9:.0f}s)", lambda: tl.vpin(window_size=WINDOW))

# ======================================================================
# ohlc('5s')                [ 0.087s]
# bid_ts() / ask_ts()       [50.047s] => [34.498s]
# spread_ts()               [28.424s] => [16.653s]
# realized_vol(window=30s)  [ 2.422s] => [2.032s]
# vpin(window=30s)          [13.646s] => [13.968s]
# ======================================================================

# ---------------------------------------------------------------------------
# Gueant: profile on a small time slice
# ---------------------------------------------------------------------------

GUEANT_SLICE_NS = int(GUEANT_SLICE_S * 1e9)
slice_end = ts_start + GUEANT_SLICE_NS

print("\n\n" + "#" * 70)
print(f"# GUEANT  –  first {GUEANT_SLICE_S}s slice  (full dataset would not return)")
print("#" * 70)

tl_small = tl[ts_start:slice_end]
n_lob_small = len(tl_small.lob.timestamps)
n_trade_small = len(tl_small.trades)
all_ts_small = len(set(tl_small._lobts.timestamps) | {t.timestamp for t in tl_small._trades})

print("\nSlice stats:")
print(f"  LOB snapshots : {n_lob_small:,}")
print(f"  Trades        : {n_trade_small:,}")
print(f"  Unique ts     : {all_ts_small:,}  ← main loop iterations in _compute_rolling_gueant")

custom_buckets = [1, 3, 5, 10]

profile(
    f"gueant.ask(window={WINDOW/1e9:.0f}s, buckets={custom_buckets})  [{GUEANT_SLICE_S}s slice]",
    lambda: tl_small.gueant.ask(WINDOW, buckets=custom_buckets),
)
profile(
    f"gueant.bid(window={WINDOW/1e9:.0f}s, buckets={custom_buckets})  [{GUEANT_SLICE_S}s slice]",
    lambda: tl_small.gueant.bid(WINDOW, buckets=custom_buckets),
)
