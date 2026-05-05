"""
features_from_parquet.py — extract trading features from raw LOB+trades parquet.

Usage:
    python examples/features_from_parquet.py <input.parquet> [options]

Output:
    A parquet file containing one row per LOB timestamp with features:
      - Best bid/ask prices & quantities
      - Spread, midprice, vw_midprice
      - Volume imbalance (vi), order flow imbalance (ofi)
      - OHLC 1s bars (merged asof)
      - Long-window rolling: realized_vol, vpin, kyle_lambda, gueant A/k
      - Short-window rolling: order rates, trade frequencies, fill rates, gueant A/k

Options:
    --output, -o PATH     Output file path [default: <input>_features.parquet]
    --format, -f FMT      Output format: parquet or csv [default: parquet]
    --hawkes              Enable rolling Hawkes fits (very slow)
    --short-window, -s S  Short rolling window in seconds [default: 5]
    --long-window, -l S   Long rolling window in seconds [default: 60]
    --tick-size TS        Minimum price increment [default: 0.1]
    --buckets B1,B2,...   Gueant/fill-rate delta buckets [default: 1,3,5,10]
"""

import argparse
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lobpy import TL

warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)

_SEC_TO_NS = 1_000_000_000


# ── progress logging ──────────────────────────────────────────────────────────
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
        print("Timing summary", flush=True)
        print("=" * 60, flush=True)
        total = sum(v[0] for v in self._records.values())
        for label, (elapsed, extra) in sorted(
            self._records.items(), key=lambda x: -x[1][0]
        ):
            pct = elapsed / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            extra_str = f"  [{extra}]" if extra else ""
            print(f"  {elapsed:8.3f}s ({pct:5.1f}%) {bar} {label}{extra_str}", flush=True)
        print(f"  {'─' * 8}  {'─' * 5}", flush=True)
        print(f"  {total:8.3f}s (100.0%) TOTAL", flush=True)
        print("=" * 60, flush=True)


sw = Stopwatch()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract trading features from raw LOB+trades parquet"
    )
    p.add_argument("input", type=str, help="Path to input parquet file")
    p.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    p.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format [default: parquet]",
    )
    p.add_argument(
        "--hawkes", action="store_true",
        help="Enable rolling Hawkes fits (very slow)",
    )
    p.add_argument(
        "--short-window", "-s",
        type=float,
        default=5.0,
        help="Short rolling window in seconds [default: 5]",
    )
    p.add_argument(
        "--long-window", "-l",
        type=float,
        default=60.0,
        help="Long rolling window in seconds [default: 60]",
    )
    p.add_argument(
        "--tick-size",
        type=float,
        default=0.1,
        help="Minimum price increment [default: 0.1]",
    )
    p.add_argument(
        "--buckets",
        type=str,
        default="1,3,5,10",
        help="Comma-separated gueant/fill-rate delta buckets [default: 1,3,5,10]",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_stem(
        input_path.stem + "_features"
    )
    if args.format == "csv":
        output_path = output_path.with_suffix(".csv")
    else:
        output_path = output_path.with_suffix(".parquet")

    short_ns = int(args.short_window * _SEC_TO_NS)
    long_ns  = int(args.long_window * _SEC_TO_NS)
    short_label = f"{args.short_window:.0f}s".replace(".0s", "s")
    long_label  = f"{args.long_window:.0f}s".replace(".0s", "s")

    print(f"Input:         {input_path}", flush=True)
    print(f"Output:        {output_path}", flush=True)
    print(f"Short window:  {short_label}  ({short_ns} ns)", flush=True)
    print(f"Long window:   {long_label}  ({long_ns} ns)", flush=True)
    print(f"Hawkes:        {'on' if args.hawkes else 'off'}", flush=True)
    print(flush=True)

    custom_buckets = [float(x.strip()) for x in args.buckets.split(",")]

    # ── load ──────────────────────────────────────────────────────────────
    sw.tick("load: from_parquet")
    tl = TL(name=input_path.stem, tick_size=args.tick_size)
    tl.from_parquet(str(input_path), mode="lazy")
    sw.tock()

    sw.tick("load: timestamps")
    total_duration = tl.timestamps[-1] - tl.timestamps[0]
    n_ts = len(tl.timestamps)
    sw.tock(f"{n_ts:,} raw ts, {total_duration / 1e9:.1f}s total")

    # ── LOB snapshot features ─────────────────────────────────────────────
    print("\n--- LOB snapshot features ---", flush=True)
    lob_series = {}
    for attr, label in [
        ("spread",      "lob.spread"),
        ("bid",         "lob.bid"),
        ("ask",         "lob.ask"),
        ("midprice",    "lob.midprice"),
        ("bidq",        "lob.bidq"),
        ("askq",        "lob.askq"),
        ("vw_midprice", "lob.vw_midprice"),
        ("vi",          "lob.vi"),
    ]:
        sw.tick(label)
        lob_series[attr] = getattr(tl.lob, attr)
        sw.tock()

    sw.tick("lob.order_flow_imbalance")
    lob_series["ofi"] = tl.lob.order_flow_imbalance
    sw.tock()

    sw.tick("build lob df")
    df_lob = pd.DataFrame(lob_series)
    lob_cols = list(df_lob.columns)
    sw.tock(f"shape={df_lob.shape}")

    # ── long-window rolling features ──────────────────────────────────────
    print(f"\n--- Long-window ({long_label}) rolling features ---", flush=True)

    if args.hawkes:
        sw.tick("rolling: hawkes")
        hawkes_roll = tl.hawkes(window_size=long_ns).rename(
            columns=lambda c: f"hawkes_{c}"
        )
        sw.tock()
    else:
        hawkes_roll = pd.DataFrame()

    sw.tick("rolling: gueant ask")
    A_ask_long, k_ask_long = tl.gueant.ask(window_size=long_ns, buckets=custom_buckets) # noqa: N806
    sw.tock()

    sw.tick("rolling: gueant bid")
    A_bid_long, k_bid_long = tl.gueant.bid(window_size=long_ns, buckets=custom_buckets) # noqa: N806
    sw.tock()

    sw.tick("rolling: realized_vol")
    rv_long = tl.realized_vol(window_size=long_ns).rename("realized_vol")
    sw.tock()

    sw.tick("rolling: vpin")
    vp_long = tl.vpin(window_size=long_ns)
    sw.tock()

    sw.tick("rolling: kyle_lambda")
    kl_long = tl.kyle_lambda(window_size=long_ns).rename("kyle_lambda")
    sw.tock()

    # ── assemble ──────────────────────────────────────────────────────────
    print("\n--- Assembly ---", flush=True)
    sw.tick("df: concat")
    pieces = [
        df_lob,
        rv_long,
        vp_long,
        kl_long,
        A_ask_long.rename("gueant_A_ask"),
        k_ask_long.rename("gueant_k_ask"),
        A_bid_long.rename("gueant_A_bid"),
        k_bid_long.rename("gueant_k_bid"),
    ]
    if args.hawkes and not hawkes_roll.empty:
        pieces.append(hawkes_roll)
    df = pd.concat(pieces, axis=1)
    df.index.name = "timestamp"
    sw.tock(f"shape={df.shape}")

    sw.tick("df: ffill LOB cols")
    df[lob_cols] = df[lob_cols].ffill()
    sw.tock()

    # ── OHLC ──────────────────────────────────────────────────────────────
    sw.tick("ohlc: 1s bars")
    ohlc = tl.ohlc("1s")
    sw.tock(f"shape={ohlc.shape}")

    sw.tick("features: merge_asof ohlc")
    df = (
        pd.merge_asof(
            df.sort_index().reset_index(),
            ohlc.add_prefix("ohlc_").reset_index(),
            on="timestamp",
            direction="backward",
        )
        .set_index("timestamp")
        .ffill()
    )
    sw.tock()

    # ── short-window rolling features ─────────────────────────────────────
    print(f"\n--- Short-window ({short_label}) rolling features ---", flush=True)
    short_suffix = f"_{short_label}"

    # per-window loop: order rates, trade frequencies, fill rates
    sw.tick(f"rolling {short_label}: per-window loop")
    records, ts_out = [], []
    for tl_w in tqdm(
        tl.rolling(short_ns),
        desc=f"rolling {short_label}",
        unit="window",
        mininterval=0.5
    ):
        ts_out.append(tl_w.timestamps[-1])
        records.append(
            {
                f"order_arrival_rate{short_suffix}":     tl_w.lob.order_arrival_frequency,
                f"order_cancel_rate{short_suffix}":      tl_w.lob.order_cancel_frequency,
                f"bid_order_arrival_rate{short_suffix}": tl_w.lob.bid_order_arrival_frequency,
                f"ask_order_arrival_rate{short_suffix}": tl_w.lob.ask_order_arrival_frequency,
                f"bid_order_cancel_rate{short_suffix}":  tl_w.lob.bid_order_cancel_frequency,
                f"ask_order_cancel_rate{short_suffix}":  tl_w.lob.ask_order_cancel_frequency,
                f"trade_freq{short_suffix}":             tl_w.trade_frequency,
                f"ask_trade_freq{short_suffix}":         tl_w.ask_trade_frequency,
                f"bid_trade_freq{short_suffix}":         tl_w.bid_trade_frequency,
            }
        )
    sw.tock(f"{len(records)} windows")

    sw.tick(f"rolling {short_label}: join")
    rolling_short = pd.DataFrame(records, index=pd.Index(ts_out, name="timestamp"))
    df = df.join(rolling_short, how="left")
    sw.tock()

    # gueant at short window
    sw.tick(f"rolling: gueant ask {short_label}")
    A_ask_short, k_ask_short = tl.gueant.ask(window_size=short_ns, buckets=custom_buckets) # noqa: N806
    sw.tock()

    sw.tick(f"rolling: gueant bid {short_label}")
    A_bid_short, k_bid_short = tl.gueant.bid(window_size=short_ns, buckets=custom_buckets) # noqa: N806
    sw.tock()

    sw.tick(f"features: gueant {short_label} join")
    df = df.join(
        pd.DataFrame(
            {
                f"gueant_A_ask{short_suffix}": A_ask_short,
                f"gueant_k_ask{short_suffix}": k_ask_short,
                f"gueant_A_bid{short_suffix}": A_bid_short,
                f"gueant_k_bid{short_suffix}": k_bid_short,
            }
        ),
        how="left",
    )
    sw.tock()

    # ── save ──────────────────────────────────────────────────────────────
    sw.tick(f"save: {args.format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        df.to_parquet(output_path, index=True)
    else:
        df.to_csv(output_path, index=True)
    file_size_mb = output_path.stat().st_size / 1e6
    sw.tock(f"{file_size_mb:.1f} MB")

    # ── report ────────────────────────────────────────────────────────────
    print(f"\nOutput: {output_path}", flush=True)
    print(f"Shape:  {df.shape}", flush=True)
    print(f"Cols:   {len(df.columns)}", flush=True)
    print(df.dtypes.value_counts().to_string(), flush=True)
    sw.summary()


if __name__ == "__main__":
    main()
