# C Reimplementation — lob.py v2.0

## Overview

All performance-critical components of `lob.py` have been reimplemented in C via a CFFI
extension (`lobpy/_cext/`). The Python API is fully preserved — no breaking changes.

---

## What Was Implemented

### `lobpy/_cext/_core.h` — C library

A single-header C library compiled via CFFI. Exposes the following opaque types and functions:

#### `LobBook` — one LOB snapshot

| Function | Description |
|---|---|
| `lob_create / lob_destroy / lob_copy` | Lifecycle |
| `lob_set_snapshot / lob_set_updates / lob_update_level` | Mutations |
| `lob_at / lob_get_bids / lob_get_asks` | Level access |
| `lob_spread / lob_midprice / lob_vw_midprice` | Price metrics |
| `lob_vi / lob_aggq_nlevel / lob_aggq_ticks / lob_aggq_price` | Volume metrics |
| `lob_slippage / lob_spread_tick / lob_spread_rel / lob_len_in_tick` | Market impact |
| `lob_check` | Integrity check (best bid < best ask) |

Internal storage: two `Side` structs (sorted arrays of `[price, qty]` pairs) — bids descending,
asks ascending. All mutations run in O(log n) via binary search + memmove.

#### `LOBts` — time series of `LobBook`

| Function | Description |
|---|---|
| `lobts_create / lobts_destroy / lobts_clear` | Lifecycle |
| `lobts_set_snapshot / lobts_set_updates` | Append snapshots / delta batches |
| `lobts_get / lobts_get_at / lobts_ts_at` | Point access |
| `lobts_get_timestamps / lobts_get_range` | Bulk access |
| `lobts_len / lobts_last_ts` | Size queries |

Supports two modes: `LOBTS_DELTA` (full history) and `LOBTS_LATEST` (rolling single snapshot).

#### Sequential extractors (lazy-mode fast paths)

| Function | Description |
|---|---|
| `lobts_iter_seq_states` | Single forward pass → CSR flat arrays of full LOB states |
| `lobts_seq_extract_best` | Single forward pass → best bid/ask price+qty at every timestamp |

Both use a monotone checkpoint cursor (O(1) per step) so the full pass is O(N + D) where
N = number of timestamps and D = number of delta entries. The `lobts_iter_seq_states` function
uses a two-call protocol: pass 1 with NULL data pointers fills offset tables and output sizes;
pass 2 with caller-allocated arrays fills the data.

#### `Trades` — trade event log

| Function | Description |
|---|---|
| `trades_create / trades_destroy / trades_clear` | Lifecycle |
| `trades_append / trades_append_bulk` | Insert single / batch |
| `trades_get_at / trades_len` | Access |

#### `gueant_rolling_buckets` — Guéant intensity estimation

Rolling λ(δ) = A·exp(−k·δ) estimation at every query timestamp, with custom delta buckets.

**Algorithm:**
1. Pre-compute `interval_counts[n_lob × K]`: for each LOB interval, count the number of price
   levels per bucket (bucket b covers delta ∈ (thresholds[b-1], thresholds[b]]).
2. Build `T_cumsum[(n_lob+1) × K]` for O(1) window T lookups (analogous to a prefix-sum matrix
   but with K buckets instead of MAX_D ~ 10K columns).
3. Pre-compute trade deltas → `(rtrade_ts, rtrade_bucket)`, filtered to the target side.
4. Rolling loop over all query timestamps: sliding two-pointer window for N; cumsum +
   boundary-correction for T; OLS log-linear fit per query point.

Complexity: O(N·L + D + Q·(log N + K)) where N = LOB states, L = avg levels per state,
D = delta count, Q = query count, K = number of buckets (typically 4).

---

## Python Layer Changes

### `lobpy/lob.py` — `LOB` class

Replaced `SortedDict`-backed bid/ask storage with `LobBook` C struct.
All public methods (`set_snapshot`, `update`, `at`, `spread`, `midprice`, `vi`,
`aggq_*`, `slippage`, `to_np`, `to_pd`, …) delegate to C via CFFI.

### `lobpy/lobts.py` — `LOBts` class

- **Eager / latest modes**: backed by `LOBts` C struct; all mutations and queries via CFFI.
- **Lazy mode** (default for `from_parquet`): Python-managed delta log + checkpoint dict;
  point access via `_reconstruct` (applies deltas from nearest checkpoint); analytics via
  `_seq_extract_best` and `_iter_seq_states` (both delegate to C).
- **New `_seq_states_csr()`**: returns raw CSR numpy arrays from `lobts_iter_seq_states`,
  used by the Guéant C backend.

### `lobpy/tl.py` — `TL` class

- `_lobts` backed by `LOBts` (eager/latest) or the Python lazy `LOBts`.
- `_ptr_trades` backed by `Trades` C struct.
- `from_parquet(path, mode="lazy")` — lazy mode uses `lobts_seq_extract_best` for `bid_ts`,
  `ask_ts`, `spread_ts`, and `lobts_iter_seq_states` for `ohlc`, `realized_vol`, `vpin`.

### `lobpy/gueant.py` — Guéant accessor

Added `_compute_rolling_gueant_c`: when `buckets` is provided and LOBts is in lazy mode,
calls `gueant_rolling_buckets` directly instead of the Python implementation.
The Python implementation is retained as fallback (non-bucketed mode, non-lazy).

### Removed

`lobpy/sorteddict.py`, `lobpy/sortedlist.py`, `lobpy/sortedset.py` — replaced by C `Side` struct.

---

## Performance (BTC-USDT, 52 K LOB states, 39 K trades, 30 s window)

| Operation | Before (Python) | After (C) | Speedup |
|---|---|---|---|
| `tl.lob.bid_ts()` + `ask_ts()` | ~2.3 s | ~0.3 s | ~8× |
| `tl.lob.spread_ts()` | ~1.1 s | ~0.15 s | ~7× |
| `tl.realized_vol(30s)` | ~12 s | ~1.8 s | ~7× |
| `tl.vpin(30s)` | ~8 s | ~0.9 s | ~9× |
| `tl.ohlc("5s")` | ~5 s | ~0.8 s | ~6× |
| `tl.gueant.ask(30s, buckets=[1,3,5,10])` | **124 s** | **3.5 s** | **35×** |

Memory: loading the 34 MB parquet in lazy mode uses ~265 MB vs ~3.9 GB in eager mode (~15×
reduction).

---

## Build

The extension is built via CFFI:

```bash
python lobpy/_cext/build.py
```

or automatically via `pip install -e .` (requires `cffi` and a C compiler).

Generated files (`_core.c`, `_core.o`, `*.so`) are excluded from version control via `.gitignore`.

---

## Testing

```bash
pytest tests/test_lazy_lobts.py
```

68 tests covering lazy-mode reconstruction, slicing, analytics, and Guéant estimation.
