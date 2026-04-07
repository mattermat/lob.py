# Full C Engine Rewrite Plan

**Date**: 2026-04-07
**Scope**: Re-implement all of `lobpy` in C (LOB, LOBts, TL, analytics), shipped as a required native extension with pre-built wheels.
**Standalone `LOB`**: Included in the C rewrite (all LOB operations backed by C structs).

---

## Architecture

```
Python (public API — thin wrappers)
  ↕ cffi boundary (numpy arrays + scalar params)
C Engine (all data structures + computation)
```

Every `TL`, `LOB`, `LOBts` Python object holds a `cffi` pointer to a C struct. All operations happen in C. Python only handles:

- Constructor arguments
- Return value formatting (C arrays → numpy/pandas)
- File I/O dispatch (parquet reading stays Python/pyarrow, then hands raw arrays to C)

---

## Current Performance Baseline

From `optimization/performance_improvement.md`:

| Operation | Python (current) |
|---|---|
| Data loading (`from_parquet`) | 16.3s |
| `bid_ts` / `ask_ts` | 54s combined |
| `spread_ts` | 15.7s |
| `realized_vol` (30s window) | 2.1s |
| `vpin` (30s window) | 13.3s |
| `gueant` (30s window) | 194s |
| **Total** | **278.5s** |
| **Peak memory** | **9.46 GB** |

### Estimated Performance After C Rewrite

| Operation | C (estimated) | Speedup |
|---|---|---|
| Data loading | ~1s | 16x |
| `bid_ts` / `ask_ts` | ~0.6s combined | 90x |
| `spread_ts` | ~0.3s | 52x |
| `realized_vol` | ~0.1s | 21x |
| `vpin` | ~0.3s | 44x |
| `gueant` | ~3-5s | 40-65x |
| **Total** | **~5-7s** | **40-56x** |
| **Peak memory** | **~200-500 MB** | **19-47x** |

---

## C Data Structures

```c
// --- Side Book (sorted flat arrays) ---

typedef struct {
    double *prices;   // bids: descending, asks: ascending
    double *qtys;     // parallel to prices
    int len;
    int cap;
} SideBook;

// --- LOB (standalone order book) ---

typedef struct {
    SideBook bids;
    SideBook asks;
    double tick_size;
    int64_t timestamp;
    char name[64];
} LobBook;

// SideBook operations
void sidebook_init(SideBook *sb, int initial_cap);
void sidebook_free(SideBook *sb);
void sidebook_insert(SideBook *sb, double price, double qty);  // sorted insert
void sidebook_update(SideBook *sb, double price, double qty);  // update or insert
void sidebook_delete(SideBook *sb, double price);
double sidebook_at(const SideBook *sb, double price);           // qty at price, 0 if absent
int sidebook_level_ticks(const SideBook *sb, double price, double tick_size);

// LobBook operations
LobBook *lob_create(const char *name, double tick_size);
void lob_destroy(LobBook *lob);
void lob_set_snapshot(LobBook *lob, int64_t ts,
                      const double *bid_p, const double *bid_q, int n_bids,
                      const double *ask_p, const double *ask_q, int n_asks);
void lob_set_updates(LobBook *lob, int64_t ts,
                     const uint8_t *sides, const double *prices, const double *qtys, int n);
double lob_spread(const LobBook *lob);
double lob_midprice(const LobBook *lob);
double lob_vw_midprice(const LobBook *lob);
double lob_spread_tick(const LobBook *lob);
double lob_spread_rel(const LobBook *lob);
double lob_vi(const LobBook *lob, int nlevels);
double lob_aggq(const LobBook *lob, int side, int nlevel, int ticks, double price);
double lob_slippage(const LobBook *lob, double volume, int side);
double lob_at(const LobBook *lob, int side, double price);

// --- Delta Log ---

typedef struct {
    int64_t *ts;
    uint8_t *side;    // 0=bid, 1=ask
    double *price;
    double *qty;
    int len;
    int cap;
} DeltaLog;

// --- Checkpoint Store ---

typedef struct {
    int64_t *ts;           // sorted checkpoint timestamps
    double **bid_prices;   // per-checkpoint bid price array
    double **bid_qtys;     // per-checkpoint bid qty array
    int *bid_lens;
    double **ask_prices;   // per-checkpoint ask price array
    double **ask_qtys;     // per-checkpoint ask qty array
    int *ask_lens;
    int len;
    int cap;
} CheckpointStore;

// --- LOB Time Series ---

typedef struct {
    CheckpointStore ckpts;
    DeltaLog deltas;
    // Reconstruction cache (LRU)
    int64_t *cache_keys;
    LobBook **cache_vals;
    int cache_len;
    int cache_cap;  // _CACHE_MAXSIZE = 32
    // Timestamp range view (for slicing)
    int64_t ts_lo;
    int64_t ts_hi;
    // Metadata
    double tick_size;
    char name[64];
    int mode;  // 0=delta, 1=latest, 2=lazy
} LobTimeSeries;

// --- Trade Log ---

typedef struct {
    int64_t *ts;
    uint8_t *side;    // 0=buy, 1=sell
    double *price;
    double *volume;
    int len;
    int cap;
} TradeLog;

// --- Timeline (TL) ---

typedef struct {
    LobTimeSeries lob;
    TradeLog trades;
    double tick_size;
    char name[64];
    int8_t timestamp_unit;  // 0=ns, 1=us, 2=ms, 3=s
    int8_t lob_mode;        // 0=delta, 1=snapshot
} Timeline;
```

---

## C Public API (functions exposed to Python via cffi)

```c
// --- Lifecycle ---
Timeline *tl_create(const char *name, double tick_size, int8_t ts_unit,
                    int8_t lob_mode, int8_t update_type);
void tl_destroy(Timeline *tl);

// --- LOB mutation ---
void tl_add_snapshot(Timeline *tl, int64_t ts,
                     const double *bid_p, const double *bid_q, int n_bids,
                     const double *ask_p, const double *ask_q, int n_asks);
void tl_add_updates(Timeline *tl, int64_t ts,
                    const uint8_t *sides, const double *prices,
                    const double *qtys, int n_updates);

// --- Trade mutation ---
void tl_add_trades(Timeline *tl, int64_t ts,
                   const uint8_t *sides, const double *prices,
                   const double *volumes, int n_trades);

// --- Bulk load (replaces _from_parquet_lazy) ---
// event_type: 0=book_level, 1=book_update, 2=trade
// side: 0=bid, 1=ask (for LOB); 0=buy, 1=sell (for trades)
void tl_load_from_arrays(Timeline *tl,
                         const int64_t *ts, const int32_t *event_types,
                         const int32_t *sides, const double *prices,
                         const double *qtys, int n_rows,
                         int build_checkpoints);

// --- Checkpoint management ---
void tl_build_checkpoints(Timeline *tl, int n_checkpoints);

// --- LOB access ---
LobBook *tl_get_lob(Timeline *tl, int64_t ts);  // borrowed pointer (cached)

// --- Standalone LOB ---
LobBook *lob_create(const char *name, double tick_size);
void lob_destroy(LobBook *lob);
void lob_set_snapshot(LobBook *lob, int64_t ts,
                      const double *bid_p, const double *bid_q, int n_bids,
                      const double *ask_p, const double *ask_q, int n_asks);
void lob_set_updates(LobBook *lob, int64_t ts,
                     const uint8_t *sides, const double *prices,
                     const double *qtys, int n);
double lob_spread(const LobBook *lob);
double lob_midprice(const LobBook *lob);
double lob_vw_midprice(const LobBook *lob);
double lob_vi(const LobBook *lob, int nlevels);
double lob_aggq(const LobBook *lob, int side, int nlevel, int ticks, double price);
double lob_slippage(const LobBook *lob, double volume, int side);
double lob_at_price(const LobBook *lob, int side, double price);

// Export LOB to flat arrays
int lob_to_arrays(const LobBook *lob,
                  double **out_bid_p, double **out_bid_q, int *out_n_bids,
                  double **out_ask_p, double **out_ask_q, int *out_n_asks);

// --- Time series analytics ---
int tl_lob_timestamps(Timeline *tl, int64_t **out_ts);
int tl_trade_timestamps(Timeline *tl, int64_t **out_ts);

int tl_bid_ts(Timeline *tl, int64_t **out_ts, double **out_val);
int tl_ask_ts(Timeline *tl, int64_t **out_ts, double **out_val);
int tl_spread_ts(Timeline *tl, int64_t **out_ts, double **out_val);
int tl_midprice_ts(Timeline *tl, int64_t **out_ts, double **out_val);
int tl_bidq_ts(Timeline *tl, int64_t **out_ts, double **out_val);
int tl_askq_ts(Timeline *tl, int64_t **out_ts, double **out_val);

// --- OHLC ---
int tl_ohlc(Timeline *tl, int64_t period_ns,
            int64_t **out_ts, double **out_o, double **out_h,
            double **out_l, double **out_c, double **out_v, int **out_count);

// --- Realized Volatility ---
double tl_realized_vol_scalar(Timeline *tl);
int tl_realized_vol_rolling(Timeline *tl, int64_t window_size,
                            int64_t **out_ts, double **out_val);

// --- VPIN ---
double tl_vpin_scalar(Timeline *tl, double bucket_size);
int tl_vpin_rolling(Timeline *tl, int64_t window_size, double bucket_size,
                    int64_t **out_ts, double **out_val);

// --- Guéant ---
int tl_gueant_ask_rolling(Timeline *tl, int64_t window_size,
                          const int *bucket_thresholds, int n_buckets,
                          int64_t **out_ts, double **out_A, double **out_k);
int tl_gueant_bid_rolling(Timeline *tl, int64_t window_size,
                          const int *bucket_thresholds, int n_buckets,
                          int64_t **out_ts, double **out_A, double **out_k);
int tl_gueant_ask_scalar(Timeline *tl,
                         const int *bucket_thresholds, int n_buckets,
                         double *out_A, double *out_k);
int tl_gueant_bid_scalar(Timeline *tl,
                         const int *bucket_thresholds, int n_buckets,
                         double *out_A, double *out_k);

// Guéant raw buckets (for inspection)
int tl_gueant_buckets(Timeline *tl, int side,
                      const int *bucket_thresholds, int n_buckets,
                      int **out_delta, int64_t **out_N, double **out_T,
                      double **out_lambda);

// --- Slicing ---
Timeline *tl_get_range(Timeline *tl, int64_t start, int64_t stop);

// --- Metadata ---
int tl_lob_snapshot_count(Timeline *tl);
int tl_trade_count(Timeline *tl);
int tl_total_event_count(Timeline *tl);

// --- Memory cleanup ---
void tl_free_i64(int64_t *p);
void tl_free_f64(double *p);
void tl_free_i32(int *p);
void tl_free_lob(LobBook *p);
```

---

## Python Wrapper Design

```python
# lobpy/_cext/__init__.py
from lobpy._cext._core import ffi, lib

# lobpy/lob.py (rewritten)
import numpy as np
from lobpy._cext import ffi, lib

class LOB:
    def __init__(self, name=None, tick_size=1, *, bids=None, asks=None):
        name = name or ""
        self._ptr = lib.lob_create(name.encode(), float(tick_size))
        if bids or asks:
            self.set_snapshot(bids or [], asks or [])

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.lob_destroy(self._ptr)

    def set_snapshot(self, bids, asks, timestamp=0):
        bp, bq = _to_flat_arrays(bids)
        ap, aq = _to_flat_arrays(asks)
        lib.lob_set_snapshot(self._ptr, timestamp,
                            ffi.cast("double*", bp.ctypes.data),
                            ffi.cast("double*", bq.ctypes.data), len(bids),
                            ffi.cast("double*", ap.ctypes.data),
                            ffi.cast("double*", aq.ctypes.data), len(asks))

    @property
    def spread(self):
        return lib.lob_spread(self._ptr)

    @property
    def midprice(self):
        return lib.lob_midprice(self._ptr)

    # ... etc. Properties delegate to C via ffi.

# lobpy/tl.py (rewritten)
class TL:
    def __init__(self, name=None, tick_size=1, lob_mode="delta", ...):
        self._ptr = lib.tl_create(name.encode(), float(tick_size),
                                  _ts_unit_int(timestamp_unit),
                                  _lob_mode_int(lob_mode), ...)

    def from_parquet(self, path, mode="lazy"):
        # parquet reading stays in Python (pyarrow), then bulk-load into C
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=[...]).sort_by("timestamp")
        ts = table.column("timestamp").to_numpy()
        et = _encode_event_types(table.column("event_type").to_numpy())
        sides = _encode_sides(table.column("side").to_numpy())
        prices = table.column("price").to_numpy()
        qtys = table.column("quantity").to_numpy()
        lib.tl_load_from_arrays(self._ptr,
                                ffi.cast("int64_t*", ts.ctypes.data), ...)

    def ohlc(self, period):
        period_ns = _period_to_ns(period, self.timestamp_unit)
        ts_p = ffi.new("int64_t**")
        ...
        n = lib.tl_ohlc(self._ptr, period_ns, ts_p, o_p, h_p, l_p, c_p, v_p, cnt_p)
        return _ohlc_arrays_to_df(n, ts_p, o_p, h_p, l_p, c_p, v_p, cnt_p)
```

---

## File Structure

```
lobpy/
  _cext/
    __init__.py         # import _core; expose ffi, lib
    build.py            # cffi ffibuilder (cdef + set_source)
    _core.c             # all C implementations (~2500-3500 lines)
    _core.h             # public struct/function declarations (~300 lines)
    internal.h          # internal helpers, constants
  __init__.py           # unchanged exports
  lob.py                # rewritten: thin cffi wrapper
  lobts.py              # rewritten: thin cffi wrapper
  tl.py                 # rewritten: thin cffi wrapper + parquet I/O
  gueant.py             # rewritten: thin cffi wrapper
  ohlc.py               # rewritten: thin cffi wrapper
  realized_volatility.py # rewritten: thin cffi wrapper
  vpin.py               # rewritten: thin cffi wrapper
  itch.py               # unchanged (ITCH parser, separate concern)
  sorteddict.py         # REMOVED (no longer needed)
  sortedlist.py         # REMOVED (no longer needed)
  sortedset.py          # REMOVED (no longer needed)

setup.py                # modified: cffi_modules + ext_modules
pyproject.toml          # modified: cffi build dependency
MANIFEST.in             # modified: include _cext/*.c, _cext/*.h
requirements-dev.txt    # modified: add cffi>=1.0.0
```

---

## Build & Distribution

### setup.py additions

```python
from cffi import FFI

setup(
    ...
    setup_requires=["cffi>=1.0.0"],
    install_requires=["numpy>=1.20.0", "pandas>=1.3.0", "cffi>=1.0.0"],
    cffi_modules=["lobpy/_cext/build.py:ffibuilder"],
)
```

### cibuildwheel (GitHub Actions)

```yaml
# .github/workflows/wheels.yml
jobs:
  build_wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install cibuildwheel
      - run: cibuildwheel --output-dir wheelhouse
      - uses: actions/upload-artifact@v4
        with:
          path: wheelhouse/*.whl
```

Target platforms:
- Linux: `manylinux2014` + `musllinux` (x86_64, aarch64)
- macOS: x86_64 + arm64 (Apple Silicon)
- Windows: x86_64
- Python: 3.8, 3.9, 3.10, 3.11, 3.12

---

## Phased Implementation

### Phase 1: Core Data Structures + Build Chain (~1-2 weeks)

**C code:**
- `SideBook` init/free/insert/update/delete/at/binary_search
- `LobBook` create/destroy/set_snapshot/set_updates/spread/midprice/vw_midprice/vi/aggq/slippage/at
- `DeltaLog` init/free/append
- `CheckpointStore` init/free/insert
- `TradeLog` init/free/append
- `LobTimeSeries` init/destroy (checkpoint + delta + cache management)
- `Timeline` create/destroy
- `tl_add_snapshot`, `tl_add_updates`, `tl_add_trades`
- `tl_build_checkpoints`
- `tl_get_lob` (checkpoint-based reconstruction with cache)
- `tl_load_from_arrays`
- Memory management: `tl_free_*`

**Build:**
- `lobpy/_cext/build.py` (cffi builder)
- `lobpy/_cext/__init__.py`
- Modified `setup.py`, `pyproject.toml`, `MANIFEST.in`

**Python wrappers:**
- `lob.py` → `LOB` class backed by `LobBook*`
- `tl.py` → `TL.__init__`, `add_lob_snapshot`, `add_lob_update`, `add_trade`, `from_parquet`

**Testing:**
- All existing `tests/test_lob.py` tests pass with C-backed `LOB`
- Basic `TL` construction + data loading works
- `examples/large_input.py` loads data successfully

### Phase 2: Time Series Extraction + Slicing (~1 week)

**C code:**
- Forward pass engine (`lob_forward_best` in C)
- `tl_lob_timestamps`, `tl_trade_timestamps`, `tl_trade_count`
- `tl_bid_ts`, `tl_ask_ts`, `tl_spread_ts`, `tl_midprice_ts`
- `tl_bidq_ts`, `tl_askq_ts`
- `tl_get_range` (slice Timeline by timestamp range)

**Python wrappers:**
- `lobts.py` → `LOBts` class backed by `LobTimeSeries*`
- All `*_ts` methods return pandas Series

**Testing:**
- All existing `tests/test_lobts.py` tests pass
- `bid_ts`, `ask_ts`, `spread_ts` output matches Python baseline within tolerance

### Phase 3: Analytics (~1-2 weeks)

**C code:**
- `tl_ohlc` (time-bucketed OHLC from trades)
- `tl_realized_vol_scalar`, `tl_realized_vol_rolling`
- `tl_vpin_scalar`, `tl_vpin_rolling` (volume buckets + sliding window)
- `tl_gueant_ask_scalar`, `tl_gueant_bid_scalar` (full-timeline fit)
- `tl_gueant_ask_rolling`, `tl_gueant_bid_rolling` (rolling with incremental least-squares)
- `tl_gueant_buckets` (raw N/T/lambda inspection)

**Key algorithmic change for Guéant:**
Replace per-step `polyfit` (O(K²) via Vandermonde + lstsq) with incremental least-squares:
- Maintain running sums `S_x`, `S_xx`, `S_y`, `S_xy` for the log-linear fit `log(lambda) = log(A) - k * delta`
- Per-step update is O(K) where K = number of delta buckets (typically 4-5)
- Fit coefficients: `k = (N*S_xy - S_x*S_y) / (N*S_xx - S_x^2)`, `A = exp((S_y + k*S_x) / N)`

**Python wrappers:**
- `gueant.py` → `GueantAccessor` backed by C
- `ohlc.py` → returns DataFrame
- `realized_volatility.py` → returns scalar or Series
- `vpin.py` → returns scalar or Series

**Testing:**
- All existing tests pass
- Numerical results match Python baseline within `atol=1e-8`
- `examples/large_input.py` runs end-to-end

### Phase 4: Polish + Distribution (~1 week)

- `cibuildwheel` GitHub Actions workflow for all platforms
- Full `examples/large_input.py` benchmark: verify ~5-7s target
- Valgrind / ASAN pass for memory safety
- Remove `sorteddict.py`, `sortedlist.py`, `sortedset.py`
- Update `README.md`, `CONTRIBUTING.md`
- Tag release

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Memory leaks in C | `tl_destroy` / `lob_destroy` frees all owned memory. Python `__del__` calls destroy. Add valgrind/ASAN to CI. |
| Numerical differences vs Python | After each phase, run `large_input.py` and compare outputs with `atol=1e-8`. |
| Platform-specific issues | `cibuildwheel` tests on all target platforms. Stick to C99, no platform-specific intrinsics. |
| Breaking public API | Keep `TL`, `LOB`, `LOBts` class interfaces identical. Only internal implementation changes. |
| Build complexity | `cffi` is well-supported; `cibuildwheel` is the standard wheel builder. Minimal custom build logic. |
| Dangling pointers | Python objects own the C struct exclusively; slicing creates new C-owned copies. |

---

## Reproducing the Baseline

```bash
source .env/bin/activate
python examples/large_input.py

# Profiling
python -m cProfile -s cumulative examples/large_input.py 2>&1 | head -60
py-spy record -o optimization/lob_pyspy.svg --rate 100 -- python examples/large_input.py
python -m memray run -o optimization/lob_memray.bin examples/large_input.py
```
