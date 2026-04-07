# Performance Profiling Report

**Date**: 2026-04-07  
**Input**: `test_data/blofin_BTC-USDT_20260330_194305.parquet` (34 MB, 10,120,473 rows)  
**Script**: `examples/large_input.py` (lazy mode, 30 s rolling windows)

---

## Executive Summary

| Metric | Value |
|---|---|
| Total runtime | 278.5 s |
| Peak memory | 9.46 GB |
| Total allocations | 51.77 GB (churn) |
| LOB snapshots | 52,687 |
| Trades | 39,433 |

The profiling reveals that **two codepaths account for ~88% of CPU time**: the Guéant rolling computation (69%) and the LOB forward-pass state iteration (18.5%). Both are pure-Python loops over tens of thousands of iterations with per-step object allocation (dict copies, DataFrames, polyfit). A targeted rewrite of these hot paths in a compiled language (Rust/Cython) would reduce total runtime by an estimated 60-80%.

---

## CPU Breakdown by Operation

| Operation | Time (s) | % CPU | Peak Memory |
|---|---|---|---|
| Guéant rolling (`_compute_rolling_gueant`) | 193.2 | **69.4%** | **4.3 GB** |
| LOB time series (`bid_ts` / `ask_ts` / `spread_ts`) | 51.5 | **18.5%** | 2.3 GB |
| Data loading (`_from_parquet_lazy` + `build_checkpoints`) | 29.6 | **10.6%** | 1.1 GB |
| VPIN rolling | 13.3 | 4.8% | — |
| Realized vol rolling | 2.1 | 0.7% | — |
| OHLC (5s) | 0.12 | <0.1% | — |

### Wall-clock output from `large_input.py`

```
- lob snapshots : 52687
- trades        : 39433
- ohlc (5s)         : 6143 candles  [0.12s]
- bid/ask ts        : 52687 points  [35.99s]
- spread ts         : mean=-0.0178  [15.70s]
- realized vol (30s): 39433 points  [2.05s]
- vpin (30s)        : 39433 points  [13.31s]
- gueant (30s)      : 92120 points  [194.12s]
```

---

## Top Function-Level Hotspots (cProfile)

| Function | Own Time (s) | Calls | Bottleneck |
|---|---|---|---|
| `_compute_rolling_gueant` (`gueant.py:227`) | **86.0** | 2 | Per-timestamp DataFrame + polyfit in a Python loop (92K iterations) |
| `_iter_seq_states` (`lobts.py:497`) | **38.7** | 105,376 | Python dict copies + delta replay at every timestamp |
| `_seq_extract_best` (`lobts.py:377`) | **34.0** | 3 | Same dict-based forward pass pattern |
| `_delta_contribs` (`gueant.py:20`) | **22.5** | 105,372 | Dict iteration per LOB state |
| `polyfit` (`polynomial.py:1285`) | **11.2** | **89,558** | Per-timestep least-squares fit (Vandermonde + lstsq) |
| `numpy.searchsorted` | **11.1** | 408,136 | Python-level dispatch overhead |
| `_fill_buckets` (VPIN) | **11.1** | 39,433 | Per-trade Python loop |

### Full cProfile (top 30, cumulative)

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.010    0.010  278.554  278.554 large_input.py:1(<module>)
        2   85.990   42.995  193.187   96.593 gueant.py:227(_compute_rolling_gueant)
        1    0.551    0.551  100.416  100.416 gueant.py:416(ask)
        1    0.378    0.378   93.701   93.701 gueant.py:435(bid)
        2    0.323    0.162   86.645   43.323 gueant.py:62(_precompute)
   105376   38.721    0.000   51.762    0.000 lobts.py:497(_iter_seq_states)
        3   33.971   11.324   51.545   17.182 lobts.py:377(_seq_extract_best)
   105372   22.539    0.000   33.609    0.000 gueant.py:20(_delta_contribs)
        1    0.002    0.002   18.437   18.437 lobts.py:734(bid_ts)
        1    0.003    0.003   17.553   17.553 lobts.py:766(ask_ts)
        1    0.079    0.079   16.369   16.369 tl.py:344(from_parquet)
        1    0.323    0.323   16.290   16.290 tl.py:400(_from_parquet_lazy)
        1    0.001    0.001   15.701   15.701 lobts.py:663(spread_ts)
        1   12.685   12.685   13.322   13.322 lobts.py:579(build_checkpoints)
        1    0.000    0.000   13.306   13.306 tl.py:312(vpin)
        1    0.221    0.221   13.305   13.305 vpin.py:93(vpin)
    39433    0.062    0.000   12.111    0.000 vpin.py:64(volume_buckets)
    89558    0.154    0.000   11.188    0.000 polynomial.py:1285(polyfit)
    39433    1.999    0.000   11.132    0.000 vpin.py:10(_fill_buckets)
   408136   11.066    0.000   11.066    0.000 {method 'searchsorted' of 'numpy.ndarray' objects}
    89558    2.583    0.000   11.033    0.000 polyutils.py:582(_fit)
 1558030    9.420    0.000    9.497    0.000 {built-in method builtins.max}
  6264538    9.343    0.000    9.343    0.000 {built-in method builtins.min}
 42342056    5.835    0.000    5.835    0.000 {built-in method builtins.round}
    89558    2.777    0.000    4.561    0.000 _linalg.py:2394(lstsq)
```

---

## Memory Analysis (memray)

| Metric | Value |
|---|---|
| Total memory allocated | 51.77 GB |
| Peak memory usage | 9.46 GB |
| Total allocation events | 3,645,167 |

### Top Allocating Locations (by size)

| Location | Total Allocated | Own Memory |
|---|---|---|
| `numpy._core.fromnumeric._wrapfunc` | 17.58 GB | — |
| `_compute_rolling_gueant` (`gueant.py:314`) | 11.07 GB | 4.33 GB |
| `_compute_rolling_gueant` (`gueant.py:256`) | 6.33 GB | — |
| `_iter_seq_states` (`lobts.py:577`) | 3.91 GB | 2.28 GB |
| `_delta_contribs` (`gueant.py:43`) | 1.91 GB | 1.45 GB |

### Top Allocating Locations (by count)

| Location | Allocations |
|---|---|
| `numpy._core._methods._sum` | 1,285,740 |
| `numpy.linalg._linalg.lstsq` | 358,238 |
| `_delta_contribs` (`gueant.py:43`) | 316,116 |
| `_delta_contribs` (`gueant.py:38`) | 315,950 |
| `_iter_seq_states` (`lobts.py:577`) | 210,768 |

### Memory Distribution by Call Stack (memray summary)

| Call Stack | Total Memory | % | Own Memory |
|---|---|---|---|
| `ask` -> `_compute_rolling_gueant` | 8.07 GB | 85.3% | 4.33 GB |
| `ask` -> `_compute_rolling_gueant` -> `_precompute` -> `_iter_seq_states` | 2.28 GB | 24.0% | 2.28 GB |
| `ask` -> `_compute_rolling_gueant` -> `_precompute` -> `_delta_contribs` | 1.45 GB | 15.3% | 1.45 GB |
| `_from_parquet_lazy` | 1.08 GB | 11.4% | 600 MB |

### Allocation Size Histogram

```
min: 1.000B
< 9.000B   :    9,807
< 84.000B  :  719,219
< 778.000B : 1,461,506  ← bulk of allocations
< 7.158kB  :  822,076
< 65.850kB :  537,639
< 605.723kB:   92,965
< 5.572MB  :    1,683
< 51.251MB :       25
< 471.424MB:      244
<=4.336GB  :        3   ← cumsum matrix in _compute_rolling_gueant
max: 4.336GB
```

---

## Flamegraphs

- **py-spy CPU flamegraph**: `optimization/lob_pyspy.svg` (25,760 samples, 100 Hz)
- **memray memory flamegraph**: `optimization/lob_memray.html`

Open `lob_pyspy.svg` in a browser to explore the CPU flame graph. Open `lob_memray.html` to explore allocation hotspots interactively.

---

## Optimization Priorities

### Priority 1 — Guéant Rolling Computation (69% CPU, 45% memory)

**Files**: `lobpy/gueant.py:227-324`

The rolling Guéant estimator runs a Python loop over 92,120 timestamps. At each step it:
1. Builds a `T_vec` via cumsum subtraction (numpy array allocation)
2. Calls `_assemble_raw` (creates a DataFrame)
3. Calls `_aggregate_buckets` (creates another DataFrame)
4. Calls `_fit` which runs `polyfit` -> `Vandermonde matrix` -> `lstsq` (**89,558 polyfit calls**)
5. The cumsum matrix `T_cumsum` can reach a single 4.3 GB allocation

**Why it's slow**: 92K iterations of per-step object creation + polynomial fitting in Python. The `polyfit`/`lstsq` call chain alone accounts for ~11s.

**Recommendations**:
- Replace per-step `polyfit` with an **incremental/online least-squares update** — O(K) per step instead of O(K^2) for Vandermonde construction + O(K^2 · M) for lstsq, where M is the number of δ buckets.
- Avoid creating DataFrames inside the loop — work with raw numpy arrays.
- Rewrite the entire inner loop in **Rust (PyO3)** or **Cython** to eliminate Python overhead and control memory layout.
- Consider chunking the cumsum matrix or using memory-mapped arrays to reduce the 4.3 GB peak.

**Expected impact**: 60-70% reduction in total runtime.

### Priority 2 — LOB State Forward Pass (18.5% CPU, 24% memory)

**Files**: `lobpy/lobts.py:478-558` (`_iter_seq_states`), `lobpy/lobts.py:377-476` (`_seq_extract_best`)

Both functions replay the delta log through Python dicts at every LOB timestamp:
- `_iter_seq_states` is called 105,376 times (called twice by `_precompute`, once per side)
- Each call does `dict(bid_dict)` / `dict(ask_dict)` copies — 210,768 dict copies allocating 2.28 GB
- Per-timestamp delta replay via `while delta_idx < len(deltas)` loop

**Why it's slow**: Python dict operations are ~50-100x slower than compiled hashmap lookups. The dict copies at each yield point create massive memory churn.

**Recommendations**:
- Rewrite the forward-pass engine in **Rust/Cython** using flat sorted arrays or B-tree maps.
- Avoid copying state at each yield — instead, expose an iterator that maintains mutable state and only snapshots on demand.
- Use contiguous numpy arrays (price, qty pairs) instead of Python dicts for the LOB state.

**Expected impact**: Eliminate 2.28 GB of memory churn; reduce `_iter_seq_states` from 38.7s to ~1-2s.

### Priority 3 — Data Loading (10.6% CPU, 1.1 GB memory)

**Files**: `lobpy/tl.py:400-452` (`_from_parquet_lazy`), `lobpy/lobts.py:560-634` (`build_checkpoints`)

- `_from_parquet_lazy`: Row-by-row `zip()` loop over 10M rows, classifying into snapshots/deltas/trades via Python `if/elif`.
- `build_checkpoints`: Forward pass through the delta log applying each delta to a Python dict.

**Recommendations**:
- Replace the `zip()` loop with **vectorized pyarrow/numpy operations**: filter columns by `event_type`, batch-convert sides to uint8, construct the delta log array directly without Python row-by-row iteration.
- No compiled language needed — proper use of existing libraries would suffice.

**Expected impact**: Reduce `_from_parquet_lazy` from 16.3s to ~2-3s.

### Lower Priority

| Operation | Time | Notes |
|---|---|---|
| VPIN rolling (`vpin.py:93-127`) | 13.3s (4.8%) | Per-window volume bucket rebuild. An incremental sliding approach or compiling `_fill_buckets` would help. |
| Realized vol rolling (`realized_volatility.py`) | 2.1s (0.7%) | Already fast enough. |
| OHLC (`ohlc.py`) | 0.12s (<0.1%) | No action needed. |

---

## Recommended Rewrite Strategy

### Option A: Rust (PyO3)

**Pros**: Maximum performance, zero-copy interop with numpy arrays, fine-grained memory control.  
**Cons**: Larger initial investment, build complexity (need `maturin` or `setuptools-rust`).

Best for Priority 1 (Guéant) and Priority 2 (LOB forward pass). The hot inner loops map naturally to Rust iterators and the cumsum matrix can use contiguous `Vec<f64>` without Python object overhead.

### Option B: Cython

**Pros**: Incremental — can optimize one function at a time without changing the package structure. Direct numpy buffer access via typed memoryviews.  
**Cons**: Less performant than Rust for complex control flow; still depends on CPython GIL.

Suitable for Priority 2 and 3. For Priority 1, Cython would help but may not achieve the same speedup as Rust due to the polyfit overhead (would still need to call numpy or implement least-squares in Cython).

### Option C: Python-level optimizations first

Before committing to a compiled rewrite, consider:

1. **Guéant**: Replace `polyfit` with incremental QR decomposition or a streaming exponential fit. Remove DataFrame creation from the inner loop (use raw numpy). This alone could cut Guéant time by 50%.
2. **LOB forward pass**: Replace Python dicts with sorted numpy arrays for bid/ask state. Avoid copying at each yield.
3. **Data loading**: Vectorize with pyarrow column operations.

This is the lowest-risk path and would validate that the algorithmic improvements are correct before investing in a compiled extension.

---

## Reproducing the Profiles

```bash
source .env/bin/activate

# cProfile (function-level CPU)
python -m cProfile -s cumulative examples/large_input.py 2>&1 | head -60

# py-spy (sampling CPU flamegraph)
py-spy record -o optimization/lob_pyspy.svg --rate 100 -- python examples/large_input.py

# memray (memory allocation profiler)
python -m memray run -o optimization/lob_memray.bin examples/large_input.py
python -m memray flamegraph optimization/lob_memray.bin --output optimization/lob_memray.html
python -m memray stats optimization/lob_memray.bin
python -m memray summary optimization/lob_memray.bin
```
