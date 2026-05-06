"""
timeline_real_data.py — comprehensive TL analytics with progress logging.
"""

import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lobpy import TL

warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)


# ── progress logging utilities ────────────────────────────────────────────────
class Stopwatch:
    def __init__(self):
        self._records = OrderedDict()

    def tick(self, label):
        self._t0 = time.perf_counter()
        self._label = label

    def tock(self, extra=None):
        elapsed = time.perf_counter() - self._t0
        self._records[self._label] = (elapsed, extra)
        suffix = f"  ({extra})" if extra else ""
        print(f"[{elapsed:8.3f}s] {self._label}{suffix}", flush=True)
        return elapsed

    def summary(self):
        print("\n" + "=" * 60, flush=True)
        print("Timing summary (sorted by duration)", flush=True)
        print("=" * 60, flush=True)
        total = sum(v[0] for v in self._records.values())
        for label, (elapsed, extra) in sorted(self._records.items(), key=lambda x: -x[1][0]):
            pct = elapsed / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            extra_str = f"  [{extra}]" if extra else ""
            print(f"  {elapsed:8.3f}s ({pct:5.1f}%) {bar} {label}{extra_str}", flush=True)
        print(f"  {'─'*8}  {'─'*5}", flush=True)
        print(f"  {total:8.3f}s (100.0%) TOTAL", flush=True)
        print("=" * 60, flush=True)


sw = Stopwatch()

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
sw.tick("data: init TL + from_parquet")
tl = TL(name="BTC-USDT", tick_size=0.1)
tl.from_parquet("test_data/blofin_BTC-USDT_20260330_194305.parquet", mode="lazy")
sw.tock()

sw.tick("data: timestamps")
total_duration = tl.timestamps[-1] - tl.timestamps[0]
n_ts = len(tl.timestamps)
window = total_duration // 10
sw.tock(f"{n_ts:,} ts, {total_duration/1e9:.1f}s")

custom_buckets = [1, 3, 5, 10]
holding_time_ns = int(1e9)

# ═══════════════════════════════════════════════════════════════════════════════
# LOB time-series stats
# ═══════════════════════════════════════════════════════════════════════════════

sw.tick("lob.spread")
spread = tl.lob.spread
sw.tock()

sw.tick("lob.bid")
bid = tl.lob.bid
sw.tock()

sw.tick("lob.ask")
ask = tl.lob.ask
sw.tock()

sw.tick("lob.midprice")
midprice = tl.lob.midprice
sw.tock()

sw.tick("lob.bidq")
bidq = tl.lob.bidq
sw.tock()

sw.tick("lob.askq")
askq = tl.lob.askq
sw.tock()

sw.tick("lob.vw_midprice")
vw_midprice = tl.lob.vw_midprice
sw.tock()

sw.tick("lob.vi")
vi = tl.lob.vi
sw.tock()

sw.tick("lob.order_flow_imbalance")
ofi = tl.lob.order_flow_imbalance
sw.tock()

sw.tick("stats: build lob_ts DataFrame")
lob_ts = pd.DataFrame(
    {
        "spread": spread,
        "bid": bid,
        "ask": ask,
        "midprice": midprice,
        "vw_midprice": vw_midprice,
        "bidq": bidq,
        "askq": askq,
        "vi": vi,
        "ofi": ofi,
    }
)
sw.tock(f"shape={lob_ts.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling analytics (global window)
# ═══════════════════════════════════════════════════════════════════════════════

sw.tick("rolling: hawkes")
hawkes_roll = tl.hawkes(window_size=window).rename(columns=lambda c: f"hawkes_{c}")
sw.tock()

sw.tick("rolling: gueant ask")
A_ask_roll, k_ask_roll = tl.gueant.ask(window_size=window, buckets=custom_buckets)
sw.tock()

sw.tick("rolling: gueant bid")
A_bid_roll, k_bid_roll = tl.gueant.bid(window_size=window, buckets=custom_buckets)
sw.tock()

sw.tick("rolling: realized_vol")
rv_roll = tl.realized_vol(window_size=window).rename("realized_vol")
sw.tock()

sw.tick("rolling: vpin")
vp_roll = tl.vpin(window_size=window)
sw.tock()

sw.tick("rolling: kyle_lambda")
kl_roll = tl.kyle_lambda(window_size=window).rename("kyle_lambda")
sw.tock()

# ═══════════════════════════════════════════════════════════════════════════════
# Main DataFrame assembly
# ═══════════════════════════════════════════════════════════════════════════════

sw.tick("df: concat")
df = pd.concat(
    [
        lob_ts,
        rv_roll,
        vp_roll,
        kl_roll,
        hawkes_roll,
        A_ask_roll.rename("gueant_A_ask"),
        k_ask_roll.rename("gueant_k_ask"),
        A_bid_roll.rename("gueant_A_bid"),
        k_bid_roll.rename("gueant_k_bid"),
    ],
    axis=1,
)
df.index.name = "timestamp"
sw.tock(f"shape={df.shape}")

sw.tick("df: ffill LOB columns")
lob_cols = list(lob_ts.columns)
df[lob_cols] = df[lob_cols].ffill()
sw.tock()

# ═══════════════════════════════════════════════════════════════════════════════
# OHLC & other stats
# ═══════════════════════════════════════════════════════════════════════════════

sw.tick("stats: OHLC 1s")
ohlc = tl.ohlc("1s")
sw.tock(f"shape={ohlc.shape}")

sw.tick("stats: volume_buckets")
vol_buckets = tl.volume_buckets()
sw.tock()

sw.tick("stats: fill_rate ask")
fill_rate_ask = tl.fill_rate(holding_time=holding_time_ns, side="a")
sw.tock()

sw.tick("stats: fill_rate bid")
fill_rate_bid = tl.fill_rate(holding_time=holding_time_ns, side="b")
sw.tock()

sw.tick("stats: gueant buckets ask")
gueant_buckets_ask = tl.gueant.buckets("a", buckets=custom_buckets)
sw.tock()

sw.tick("stats: gueant buckets bid")
gueant_buckets_bid = tl.gueant.buckets("b", buckets=custom_buckets)
sw.tock()

# ── Global scalars ────────────────────────────────────────────────────────────

sw.tick("scalars: gueant ask (global)")
A_ask_s, k_ask_s = tl.gueant.ask(buckets=custom_buckets)
sw.tock()

sw.tick("scalars: gueant bid (global)")
A_bid_s, k_bid_s = tl.gueant.bid(buckets=custom_buckets)
sw.tock()

sw.tick("scalars: hawkes (global)")
hawkes_global = tl.hawkes()
sw.tock()

sw.tick("scalars: all misc")
scalars = pd.Series(
    {
        "trade_frequency": tl.trade_frequency,
        "ask_trade_frequency": tl.ask_trade_frequency,
        "bid_trade_frequency": tl.bid_trade_frequency,
        "trade_volume_imbalance": tl.trade_volume_imbalance,
        "order_arrival_volume": tl.lob.order_arrival_volume,
        "order_cancel_volume": tl.lob.order_cancel_volume,
        "order_arrival_frequency": tl.lob.order_arrival_frequency,
        "order_cancel_frequency": tl.lob.order_cancel_frequency,
        "bid_arrival_frequency": tl.lob.bid_order_arrival_frequency,
        "ask_arrival_frequency": tl.lob.ask_order_arrival_frequency,
        "bid_cancel_frequency": tl.lob.bid_order_cancel_frequency,
        "ask_cancel_frequency": tl.lob.ask_order_cancel_frequency,
        "realized_vol": tl.realized_vol(),
        "vpin": tl.vpin(),
        "kyle_lambda": tl.kyle_lambda(),
        **{f"hawkes_{k}": v for k, v in hawkes_global.items()},
        "gueant_A_ask": A_ask_s,
        "gueant_k_ask": k_ask_s,
        "gueant_A_bid": A_bid_s,
        "gueant_k_bid": k_bid_s,
    },
    name="value",
)
sw.tock()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

window_5s = 5_000_000_000
window_10s = 10_000_000_000
window_60s = 60_000_000_000

sw.tick("summary: merge_asof OHLC")
summary = pd.merge_asof(
    df.sort_index().reset_index(),
    ohlc.add_prefix("ohlc_").reset_index(),
    on="timestamp",
    direction="backward",
).set_index("timestamp")
sw.tock()

sw.tick("summary: realized_vol 10s + 60s")
summary = summary.join(
    tl.realized_vol(window_size=window_10s).rename("realized_vol_10s"), how="left"
)
summary = summary.join(
    tl.realized_vol(window_size=window_60s).rename("realized_vol_60s"), how="left"
)
sw.tock()

# ═══════════════════════════════════════════════════════════════════════════════
# Rolling 5s windows
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Rolling 5s windows ---", flush=True)
sw.tick("rolling 5s: all-compute")
records, ts_out = [], []
roll_iter = list(tl.rolling(window_5s))
for tl_w in tqdm(roll_iter, desc="rolling 5s", unit="window", mininterval=0.5):
    ts_out.append(tl_w.timestamps[-1])
    fr_a = tl_w.fill_rate(holding_time=window_10s, side="a", buckets=custom_buckets)
    fr_b = tl_w.fill_rate(holding_time=window_10s, side="b", buckets=custom_buckets)
    fr_a_map = fr_a.set_index("delta")["fill_rate"].to_dict()
    fr_b_map = fr_b.set_index("delta")["fill_rate"].to_dict()
    records.append(
        {
            "order_arrival_rate_5s": tl_w.lob.order_arrival_frequency,
            "order_cancel_rate_5s": tl_w.lob.order_cancel_frequency,
            "bid_order_arrival_rate_5s": tl_w.lob.bid_order_arrival_frequency,
            "ask_order_arrival_rate_5s": tl_w.lob.ask_order_arrival_frequency,
            "bid_order_cancel_rate_5s": tl_w.lob.bid_order_cancel_frequency,
            "ask_order_cancel_rate_5s": tl_w.lob.ask_order_cancel_frequency,
            "trade_freq_5s": tl_w.trade_frequency,
            "ask_trade_freq_5s": tl_w.ask_trade_frequency,
            "bid_trade_freq_5s": tl_w.bid_trade_frequency,
            **{f"fill_rate_ask_d{d}": fr_a_map.get(float(d), np.nan) for d in custom_buckets},
            **{f"fill_rate_bid_d{d}": fr_b_map.get(float(d), np.nan) for d in custom_buckets},
        }
    )
sw.tock(f"{len(records)} windows")

sw.tick("summary: rolling_5s join")
rolling_5s = pd.DataFrame(records, index=pd.Index(ts_out, name="timestamp"))
summary = summary.join(rolling_5s, how="left")
sw.tock()

sw.tick("rolling: gueant ask 5s")
A_ask_5s, k_ask_5s = tl.gueant.ask(window_size=window_5s, buckets=custom_buckets)  # noqa: N806
sw.tock()

sw.tick("rolling: gueant bid 5s")
A_bid_5s, k_bid_5s = tl.gueant.bid(window_size=window_5s, buckets=custom_buckets)
sw.tock()

sw.tick("summary: gueant 5s join")
summary = summary.join(
    pd.DataFrame(
        {
            "gueant_A_ask_5s": A_ask_5s,
            "gueant_k_ask_5s": k_ask_5s,
            "gueant_A_bid_5s": A_bid_5s,
            "gueant_k_bid_5s": k_bid_5s,
        }
    ),
    how="left",
)
sw.tock()

# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== main df ({df.shape}) ===", flush=True)
print(df, flush=True)

print(f"\n=== ohlc 1s ({ohlc.shape}) ===", flush=True)
print(ohlc, flush=True)

print("\n=== volume buckets ===", flush=True)
print(vol_buckets, flush=True)

print("\n=== fill rate — ask ===", flush=True)
print(fill_rate_ask, flush=True)

print("\n=== fill rate — bid ===", flush=True)
print(fill_rate_bid, flush=True)

print(f"\n=== guéant buckets — ask {custom_buckets} ===", flush=True)
print(gueant_buckets_ask, flush=True)

print(f"\n=== guéant buckets — bid {custom_buckets} ===", flush=True)
print(gueant_buckets_bid, flush=True)

print(f"\n=== summary ({summary.shape}) ===", flush=True)
print(summary.head(), flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
sw.summary()
