# `LOBts` API (Time Series LOB)

### Initialization
`LOBts(name=None, tick_size=1, mode='lazy')`
- `name`: Optional identifier for the time series (auto-generated if `None`)
- `tick_size`: Minimum price increment (default: `1`)
- `mode`: Storage mode — `'lazy'` (default), `'eager'`, or `'latest'`

### Storage modes

#### `mode='lazy'` (default)
Stores only **checkpoints** (full snapshots at sparse intervals) plus a **delta log** (compact numpy structured array of `(ts, side, price, qty)` changes). LOB states at arbitrary timestamps are **reconstructed on demand** from the nearest preceding checkpoint + delta replay, and cached in an LRU cache (max 32 entries).

- **Memory**: O(C + D) where C = total checkpoint levels and D = total delta rows — dramatically lower than eager for large files.
- **Random access** (`lobts[ts]`): O(D/C) amortised (replay from nearest checkpoint).
- **Sequential analytics** (`spread_ts`, `bid_ts`, etc.): O(N + D) via a single C forward pass (`lobts_seq_extract_best`), avoiding per-timestamp reconstruction entirely.
- **`build_checkpoints(n=100)`**: generates `n` evenly spaced checkpoints from the delta log to bound random-access cost. Called automatically after `from_parquet(..., mode='lazy')`.
- `set_snapshot` stores the bids/asks as a checkpoint (numpy arrays).
- `set_updates` appends rows to the delta log; returns `None`.

#### `mode='eager'`
Every snapshot/update is stored as a **full `LobBook` object in C memory**. No reconstruction needed.

- **Memory**: O(N × L) where N = number of timestamps, L = average levels per snapshot.
- **Random access**: O(log N) binary search in C array.
- `set_updates` copies the previous LOB state, applies deltas, and stores a new full snapshot. Returns the new `LOB` object.

#### `mode='latest'`
Keeps **only the most recent snapshot** in C memory — previous snapshots are discarded.

- **Memory**: O(L) — constant regardless of history length.
- `set_snapshot` replaces the single stored LOB.
- `set_updates` copies the current LOB, applies deltas, and replaces it.
- Random access only works for the latest timestamp; all others return `None`.

### Core Methods
- `set_snapshot(bids, asks, timestamp=0, force=False)`: create a LOB snapshot at the given timestamp
  - `bids`: list of `(price, size)` tuples for bid side
  - `asks`: list of `(price, size)` tuples for ask side
  - `timestamp`: timestamp for this snapshot
  - `force`: if `True`, overwrite existing timestamp (default raises `ValueError`)
  - In **lazy** mode: stores as a checkpoint array pair
  - In **eager/latest** mode: stored as a C `LobBook`

- `set_updates(updates, timestamp=0)`: apply incremental updates to create a new LOB state
  - `updates`: list of `(side, price, size)` tuples
    - `side`: `'b'`/`'bid'` for bids, `'a'`/`'ask'` for asks
    - `size`: quantity (`0` to delete level)
  - `timestamp`: timestamp for this snapshot
  - **Lazy**: appends rows to the delta log; returns `None`
  - **Eager/latest**: copies previous state, applies deltas, stores new snapshot; returns the new `LOB`

- `update(side, price_level, size, timestamp=0)`: apply a single update (delegates to `set_updates`)
  - Returns: the new `LOB` object (eager/latest) or `None` (lazy)

- `build_checkpoints(n=100)`: generate `n` evenly spaced checkpoints from the delta log (lazy mode only)
  - Makes random access via `lobts[ts]` O(D/n) instead of O(D)
  - Called automatically after `from_parquet(..., mode='lazy')`

### Properties
- `mode`: the storage mode string (`'lazy'`, `'eager'`, or `'latest'`)
- `timestamps`: sorted list of all LOB timestamps
- `len`: number of timestamps (same as `len(lobts)`)
- `len_ts`: duration — `last_timestamp - first_timestamp`

### Time Indexing
- `lobts[timestamp]`: access LOB at a specific timestamp
  - Returns: `LOB` object or `None` if not found
  - **Lazy**: reconstructs from nearest checkpoint + delta replay (cached)
  - **Eager/latest**: binary search in C array

- `lobts[start:end]`: slice time range (both inclusive)
  - Returns: new `LOBts` with filtered snapshots
  - **Lazy**: copies the nearest preceding checkpoint (as reconstruction anchor) plus relevant deltas; result is also lazy

- `timestamp in lobts`: check if a timestamp exists (`__contains__`)
  - **Lazy**: checks checkpoints and delta log
  - **Eager/latest**: binary search in C

- `for lob in lobts`: iterate over `LOB` objects in timestamp order (`__iter__`)

- `lobts.get_at_timestamp(timestamp)`: alias for `lobts[timestamp]`

- `lobts.get_range(start_ts, end_ts)`: alias for `lobts[start_ts:end_ts]`

### LOB Properties (at specific timestamp)
Access via `lobts[timestamp]` — returns a full `LOB` object (see [LOB docs](lob.md)):
- `lobts[ts].bid`, `lobts[ts].ask`, `lobts[ts].bidq`, `lobts[ts].askq`, `lobts[ts].vi`
- `lobts[ts].spread`, `lobts[ts].spread_tick`, `lobts[ts].spread_rel`
- `lobts[ts].midprice`, `lobts[ts].vw_midprice`
- `lobts[ts].check()`

### Time Series Statistics
Properties returning pandas Series indexed by timestamp:

- `lobts.spread`: spread time series
- `lobts.bid`: best bid price time series
- `lobts.ask`: best ask price time series
- `lobts.bidq`: best bid quantity time series
- `lobts.askq`: best ask quantity time series
- `lobts.midprice`: mid-price time series
- `lobts.vw_midprice`: volume-weighted mid-price time series
- `lobts.vi`: volume imbalance time series

**Lazy mode**: these are computed via a single C forward pass (`lobts_seq_extract_best`) — O(N + D).
**Eager/latest mode**: iterates stored LOB objects.

### Time-Based Statistics
- `lobts.arrival_frequency`: total order arrivals (L2 quantity-based)
  - Counts quantity added to the order book across all transitions
  - Includes: new levels and quantity increases at existing levels

- `lobts.cancel_frequency`: total order cancellations (L2 quantity-based)
  - Counts quantity removed from the order book across all transitions
  - Includes: full cancellations (level → 0) and partial cancellations (quantity decreases)

- `lobts.update_frequency()`: total updates (arrivals + cancellations)

### Utility Methods
- `lobts.diff(other)`: calculate differences between two `LOBts`
  - `other`: another `LOBts` object to compare with
  - Returns: list of `(timestamp, bid_deltas, ask_deltas)` tuples

### Conversion Methods
- `lobts.to_np(start_ts=None, end_ts=None)`: export to numpy array
  - Returns: array with shape `(n, 5)` — `[timestamp, side, level, price, size]`

- `lobts.to_pd(start_ts=None, end_ts=None)`: export to pandas DataFrame
  - Returns: DataFrame with columns `['timestamp', 'side', 'level', 'price', 'size']`

- `lobts.to_csv(path, start_ts=None, end_ts=None)`: export to CSV
- `lobts.to_xlsx(path, start_ts=None, end_ts=None)`: export to XLSX
- `lobts.to_parquet(path, start_ts=None, end_ts=None)`: export to Parquet

### L2 Order Book Semantics
`LOBts` uses L2 (level 2) order book semantics for frequency calculations:

**Arrival Frequency**: counts quantity added to the book
- New level arrival: full quantity at new price level
- Quantity increase: difference when existing level grows (X → Y, where Y > X)

**Cancel Frequency**: counts quantity removed from the book
- Full cancellation: complete quantity at removed level (X → 0)
- Partial cancellation: difference when existing level shrinks (X → Y, where Y < X)

Example:
```
t=1000:  bid@100.00: 10
t=1100:  bid@100.00: 7   (partial cancel: -3)
t=1200:  bid@100.00: 15  (quantity increase: +8)
```
- Arrival from t=1000→1100: 0 (no increases)
- Cancel from t=1000→1100: 3 (10→7)
- Arrival from t=1100→1200: 8 (7→15)
