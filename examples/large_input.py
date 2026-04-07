"""
Large file loading benchmark — eager vs lazy mode.

Tests TL.from_parquet on data/blofin_BTC-USDT_20260330_194305.parquet,
comparing memory usage and load time between 'eager' and 'lazy' modes,
then exercises the lazy TL with the same analytics as real_timeline.py.

Run from the repo root:
    python examples/large_input.py
"""

import time

from lobpy import TL

PATH = "test_data/blofin_BTC-USDT_20260330_194305.parquet"
WINDOW = 30_000_000_000  # 30 s in nanoseconds

tl = TL(name="BTC-USDT", tick_size=0.1)
tl.from_parquet(PATH, mode="lazy")

print(f"- lob snapshots : {len(tl.lob.timestamps)}")
print(f"- trades        : {len(tl.trades)}")

t0 = time.perf_counter()
ohlc = tl.ohlc("5s")
print(f"- ohlc (5s)         : {len(ohlc)} candles  [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
bid_ts = tl.lob.bid_ts()
ask_ts = tl.lob.ask_ts()
print(f"- bid/ask ts        : {len(bid_ts)} points  [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
spread = tl.lob.spread_ts()
print(f"- spread ts         : mean={spread.mean():.4f}  [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
rvol = tl.realized_vol(window_size=WINDOW)
print(f"- realized vol (30s): {len(rvol)} points  [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
vpin = tl.vpin(window_size=WINDOW)
print(f"- vpin (30s)        : {len(vpin)} points  [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
custom_buckets = [1, 3, 5, 10]
Aa, ka = tl.gueant.ask(WINDOW, buckets=custom_buckets)
Ab, kb = tl.gueant.bid(WINDOW, buckets=custom_buckets)
print(f"- gueant (30s)      : {len(ka)} points  [{time.perf_counter()-t0:.2f}s]")
