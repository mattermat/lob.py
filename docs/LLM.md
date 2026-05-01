# `lobpy` — Complete API Reference

> **v1.2.1** · Three core objects: `LOB` (static book), `LOBts` (book time series), `TL` (book + trades timeline).

---

## `LOB` — Static Limit Order Book

### Constructor

```python
LOB(name=None, tick_size=1, *, bids=None, asks=None)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | auto | Identifier string |
| `tick_size` | `float` | `1` | Minimum price increment |
| `bids` | `list[(price, size)] \| None` | `None` | Initial bid levels |
| `asks` | `list[(price, size)] \| None` | `None` | Initial ask levels |

### Mutation Methods

| Method | Signature | Notes |
|---|---|---|
| **set_snapshot** | `set_snapshot(bids, asks, timestamp=0)` | Replace full book state |
| **set_updates** | `set_updates(updates, timestamp=0)` | Apply incremental deltas; `updates` = `[(side, price, size), …]`; `size=0` removes level |
| **update** | `update(side, price_level, size, timestamp=0)` | Single-level update; delegates to `set_updates` |

> **Side values**: `'b'` / `'bid'` for bids, `'a'` / `'ask'` for asks.

### Scalar Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Identifier |
| `tick_size` | `float` | Min price increment (read/write) |
| `timestamp` | `float` | Current timestamp (read/write) |
| `spread` | `float` | `ask − bid` (absolute) |
| `spread_tick` | `float` | Spread in number of ticks |
| `spread_rel` | `float` | Spread as % of best bid |
| `midprice` | `float` | `(bid + ask) / 2` |
| `vw_midprice` | `float` | Volume-weighted mid-price |

### Indexable Accessors

Accessed like `lob.bid[i]`, `lob.askq[i]`, etc. Support full arithmetic (`+`, `-`, `*`, `/`, comparisons, `float()`, `int()`).

| Accessor | `[0]` | `[i]` |
|---|---|---|
| `bid` | Best bid price | Bid price at level *i* |
| `ask` | Best ask price | Ask price at level *i* |
| `bidq` | Best bid qty | Bid qty at level *i* |
| `askq` | Best ask qty | Ask qty at level *i* |
| `vi` | Volume imbalance (1 lvl) | Imbalance of top *i+1* levels |

> `lob.bid` ≡ `lob.bid[0]`. Out-of-range indices return `0.0`.

### Query Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| **at** | `at(side, price)` | `float` | Qty at exact price level |
| **check** | `check()` | `bool` | `True` if not crossed |
| **get_slippage** | `get_slippage(volume, side='midprice')` | `float` | Per-unit price impact vs midprice |
| **aggq** | `aggq(side, nlevel=None, ticks=None, price=None)` | `float` | Aggregated qty on one side; exactly one kwarg required |
| **len_in_tick** | `len_in_tick(side, price)` | `int` | Distance in ticks from best to *price* |
| **diff** | `diff(other)` | `list[(side, price, size)]` | Updates to go from `self` → `other` |
| **get_delta** | `get_delta(bids, asks, timestamp=0)` | `(bid_deltas, ask_deltas)` | Replace book with new snapshot and return deltas |

### Export

| Method | Output |
|---|---|
| `to_np(side=None, nlevels=None)` | `ndarray` — shape `(n,2)` [price, size] or `(n,3)` [side, price, size] if `side=None` |
| `to_pd(side=None, nlevels=None)` | `DataFrame` — columns `['price','size']` or `['price','size','side']` |
| `to_csv(path, side=None, nlevels=None)` | CSV file |
| `to_xlsx(path, side=None, nlevels=None)` | XLSX file |
| `to_parquet(path, side=None, nlevels=None)` | Parquet file |

---

## `LOBts` — Limit Order Book Time Series

### Constructor

```python
LOBts(name=None, tick_size=1, mode='lazy', timestamp_unit='ns')
```

`timestamp_unit` — `'s'` / `'ms'` / `'us'` / `'ns'` — governs per-second rate calculations in frequency properties.

### Storage Modes

| Mode | Storage | Memory | Random Access | Sequential Analytics |
|---|---|---|---|---|
| **`'lazy'`** *(default)* | Checkpoints + delta log (`np` structured array) | O(C + D) | O(D/C) amortised, cached (LRU 32) | O(N + D) single C pass |
| **`'eager'`** | Full `LobBook` per timestamp | O(N × L) | O(log N) binary search | O(N) |
| **`'latest'`** | Only most recent snapshot | O(L) | Latest only; others → `None` | N/A |

### Core Methods

| Method | Signature | Lazy returns | Eager/Latest returns |
|---|---|---|---|
| **set_snapshot** | `set_snapshot(bids, asks, timestamp=0, force=False)` | `None` | `None` |
| **set_updates** | `set_updates(updates, timestamp=0)` | `None` | `LOB` |
| **update** | `update(side, price, size, timestamp=0)` | `None` | `LOB` |
| **build_checkpoints** | `build_checkpoints(n=100)` | — | — *(lazy only)* |

> `force=True` overwrites existing timestamp (default: raises `ValueError`).

### Properties

| Property | Type | Description |
|---|---|---|
| `mode` | `str` | `'lazy'` / `'eager'` / `'latest'` |
| `timestamps` | `list` | Sorted list of all timestamps |
| `len` / `len(lobts)` | `int` | Number of timestamps |
| `len_ts` | `float` | Duration = `last_ts − first_ts` |

### Indexing & Iteration

| Syntax | Returns | Description |
|---|---|---|
| `lobts[ts]` | `LOB \| None` | LOB at timestamp |
| `lobts[start:end]` | `LOBts` | Slice (both inclusive) |
| `ts in lobts` | `bool` | Timestamp existence check |
| `for lob in lobts` | — | Iterate `LOB` objects in order |
| `lobts.get_at_timestamp(ts)` | `LOB \| None` | Alias for `lobts[ts]` |
| `lobts.get_range(a, b)` | `LOBts` | Alias for `lobts[a:b]` |

### Time Series Statistics → `pd.Series`

Each statistic is available in two forms:
- **Property** (full series): `lobts.spread`, `lobts.bid`, etc. — covers the whole time range.
- **Method** (filtered range): `lobts.spread_ts(start_ts=None, end_ts=None)`, etc. — both bounds inclusive, `None` means unbounded.

| Property | Explicit method | Computed via |
|---|---|---|
| `lobts.spread` | `lobts.spread_ts(start_ts, end_ts)` | C forward pass *(lazy)* / iteration *(eager)* |
| `lobts.bid` | `lobts.bid_ts(start_ts, end_ts)` | " |
| `lobts.ask` | `lobts.ask_ts(start_ts, end_ts)` | " |
| `lobts.bidq` | `lobts.bidq_ts(start_ts, end_ts)` | " |
| `lobts.askq` | `lobts.askq_ts(start_ts, end_ts)` | " |
| `lobts.midprice` | `lobts.midprice_ts(start_ts, end_ts)` | " |
| `lobts.vw_midprice` | *(property only)* | per-timestamp LOB access |
| `lobts.vi` | *(property only)* | per-timestamp LOB access |

> Use the explicit `_ts` methods when you only need a sub-range — they avoid processing the full series.

### Per-Timestamp LOB Access

`lobts[ts]` returns a full `LOB` object — all `LOB` properties and methods are available:

```python
lobts[ts].bid[0]      # best bid
lobts[ts].spread       # spread at ts
lobts[ts].check()      # consistency check
lobts[ts].get_slippage(100, side='a')  # slippage for 100 units on ask side
```

### Order Book Activity (L2 Quantity-Based)

**Volume** (total quantity moved across all transitions):

| Property | Description |
|---|---|
| `order_arrival_volume` | Total qty added (new levels + qty increases) |
| `order_cancel_volume` | Total qty removed (full + partial cancels) |
| `update_volume()` | `order_arrival_volume + order_cancel_volume` |

**Frequency** (events per second; one event = one price level that changed):

| Property | Description |
|---|---|
| `order_arrival_frequency` | Arrival events / second (both sides) |
| `order_cancel_frequency` | Cancel events / second (both sides) |
| `bid_order_arrival_frequency` | Bid-side arrival events / second |
| `ask_order_arrival_frequency` | Ask-side arrival events / second |
| `bid_order_cancel_frequency` | Bid-side cancel events / second |
| `ask_order_cancel_frequency` | Ask-side cancel events / second |

**Order flow imbalance**:

| Property | Type | Description |
|---|---|---|
| `order_flow_imbalance` | `pd.Series` | `OFI(t) = (bid_arr_vol − bid_can_vol) − (ask_arr_vol − ask_can_vol)` per transition; positive = bullish pressure |

### Utility & Export

| Method | Output |
|---|---|
| `diff(other)` | `list[(ts, bid_deltas, ask_deltas)]` |
| `to_np(start=None, end=None)` | `ndarray (n,5)` — `[ts, side, level, price, size]` |
| `to_pd(start=None, end=None)` | `DataFrame` — columns `[timestamp, side, level, price, size]` |
| `to_csv(path, …)` | CSV file |
| `to_xlsx(path, …)` | XLSX file |
| `to_parquet(path, …)` | Parquet file |

---

## `TL` — Timeline (LOB + Trades)

Unified container for order book events and trade records.

### Constructor

```python
TL(name=None, tick_size=1, lob_mode='delta', update_type='realtime', timestamp_unit='ns')
```

| Param | Values | Description |
|---|---|---|
| `lob_mode` | `'delta'` / `'snapshot'` | Incremental updates vs full snapshots |
| `update_type` | `'realtime'` / `'fixed'` | Sparse vs regular-interval updates |
| `timestamp_unit` | `'s'` / `'ms'` / `'us'` / `'ns'` | Timestamp resolution; used to scale `ohlc` periods & rolling windows |

### LOB Methods

```python
tl.add_lob_snapshot(timestamp, bids, asks)    # full book snapshot
tl.add_lob_update(timestamp, updates)          # incremental deltas
```

### LOB Access

| Access | Description |
|---|---|
| `tl.lob` | Underlying `LOBts` — supports all `LOBts` indexing/statistics |
| `tl.lob[ts]` | `LOB \| None` — exact-match lookup; returns `None` if `ts` has no LOB event |

### Trade Methods

```python
tl.add_trade(timestamp, side, price, volume)         # single trade
tl.add_trades(timestamp, trades)                       # batch: [(side, price, vol), …]
```

> **Trade side**: `'b'` = buy aggressor (takes asks), `'s'` = sell aggressor (takes bids).

### Trade Access

| Property | Type | Description |
|---|---|---|
| `trades` | `list[Trade]` | All trades in insertion order |
| `Trade.timestamp` | `float` | — |
| `Trade.side` | `str` | `'b'` / `'s'` |
| `Trade.price` | `float` | — |
| `Trade.volume` | `float` | — |

### General Access

| Property | Description |
|---|---|
| `timestamps` | Sorted, deduplicated list of all event timestamps (LOB + trades) |
| `len(tl)` | Total event count (LOB snapshots/updates + individual trades) |

### Slicing & Rolling

| Syntax | Returns | Description |
|---|---|---|
| `tl[start:stop]` | `TL` | Inclusive range slice; inherits all config |
| `tl.rolling(window_size)` | generator of `TL` | Yields `tl[ts − window : ts]` for each event `ts` |

### Export

| Method | Output |
|---|---|
| `to_pd()` | `DataFrame` — columns: `timestamp, type, side, level, price, size` |
| `to_np()` | `ndarray (n, 6)` — same columns; empty → shape `(0, 6)` |

> `type` is `'lob'` or `'trade'`; `level` is `NaN` for trades.

### Analytics — Trade Activity

**Frequency** (trades per second; scaled by `timestamp_unit`):

| Property | Description |
|---|---|
| `trade_frequency` | All trades / second |
| `ask_trade_frequency` | Buy-aggressor trades / second (`side='b'`, hits the ask) |
| `bid_trade_frequency` | Sell-aggressor trades / second (`side='s'`, hits the bid) |

**Volume imbalance**:

| Property | Type | Range | Description |
|---|---|---|---|
| `trade_volume_imbalance` | `float` | `[−1, +1]` | `(buy_vol − sell_vol) / (buy_vol + sell_vol)`; `0.0` if no trades |

**Order book activity** (delegated to `tl.lob`):

```python
tl.lob.order_arrival_volume        # total qty added
tl.lob.order_cancel_volume         # total qty removed
tl.lob.order_arrival_frequency     # arrival events/s
tl.lob.order_cancel_frequency      # cancel events/s
tl.lob.bid_order_arrival_frequency # bid-side arrivals/s
tl.lob.ask_order_arrival_frequency # ask-side arrivals/s
tl.lob.bid_order_cancel_frequency  # bid-side cancels/s
tl.lob.ask_order_cancel_frequency  # ask-side cancels/s
tl.lob.order_flow_imbalance        # pd.Series OFI per LOB transition
```

### Analytics — OHLC

```python
tl.ohlc(period)   # → DataFrame [open, high, low, close, volume, count]
```

| Period | |
|---|---|
| `'1s'` · `'5s'` · `'1m'` · `'15m'` · `'1h'` · `'24h'` | Scaled by `timestamp_unit` |

### Analytics — Realized Volatility

```python
tl.realized_vol(window_size=None)   # scalar (all trades) or pd.Series (rolling, indexed by end ts)
```

Formula: `σ = √(Σ log²(pᵢ / pᵢ₋₁))`. Returns `nan` if < 2 trades.

### Analytics — Volume Buckets & VPIN

```python
tl.volume_buckets(bucket_size=None, include_partial=False)
# → DataFrame [buy_volume, sell_volume], indexed by bucket number
```

```python
tl.vpin(window_size=None, bucket_size=None)
# → scalar or pd.Series (rolling)
```

> `VPIN = Σ|V_buy[i] − V_sell[i]| / (n × bucket_size)`. Near 0 = balanced flow; near 1 = directional/informed.

### Analytics — Fill Rate

```python
tl.fill_rate(holding_time, side='a', buckets=None)
# → pd.DataFrame [delta, N, T, lambda, fill_rate]
```

`P(fill | δ, T) = 1 − exp(−λ̂(δ) · holding_time)` using the empirical rate λ̂(δ) = N(δ)/T(δ) (no model fitting). `holding_time` is in the same timestamp units as the TL.

| Param | Description |
|---|---|
| `holding_time` | resting duration before cancellation |
| `side` | `'a'` (ask, filled by buy-aggressors) / `'b'` (bid, filled by sell-aggressors) |
| `buckets` | optional δ threshold list; same as `tl.gueant.buckets()` |

`fill_rate` column is nan where `lambda` is nan. Assumes Poisson arrivals (trade clustering means slightly underestimated for short windows, overestimated for long ones).

### Analytics — Kyle's Lambda

Price impact coefficient `λ` from `ΔP_mid = λ · Q_signed + α` (OLS). `Q_signed = buy_volume − sell_volume`; zero-flow intervals excluded.

```python
tl.kyle_lambda(interval=None, window_size=None)
# → float                   (window_size=None)
# → pd.Series (end_ts idx)  (window_size=W)
```

| Param | Description |
|---|---|
| `interval=None` | One obs per consecutive LOB-update pair `(t₁, t₂)`: Q=trades in `(t₁,t₂]`, ΔP=mid(t₂)−mid(t₁) |
| `interval=N` | Non-overlapping fixed buckets `[t, t+N)`: Q=trades in bucket, ΔP=mid(t+N)−mid(t) |
| `window_size=None` | Scalar over all obs; `W` = rolling Series indexed by obs end timestamps |

Returns `nan` / empty Series when fewer than 2 non-zero-flow observations are available. Higher `λ` = thinner liquidity or stronger adverse selection.

### Analytics — Hawkes Process

Models **trade arrival** self-excitation: `λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ))`. Fitted on `tl.trades` only — order arrivals/cancellations require a separate model.

```python
tl.hawkes(side=None, window_size=None)
# → dict {'mu', 'alpha', 'beta', 'branching_ratio'}     (window_size=None)
# → pd.DataFrame [mu, alpha, beta, branching_ratio]      (window_size=N)
```

| Param | Description |
|---|---|
| `side` | `None` = all trades; `'b'` = buy-aggressors; `'s'` = sell-aggressors |
| `window_size` | `None` = scalar fit; `N` = rolling fit at each trade ts (same units as timestamps) |

Parameters always in SI units (events/s for μ, α; 1/s for β). `branching_ratio = α/β`; must be < 1 for stationarity. Returns nan if < 3 trades in scope.

### Analytics — Guéant Intensity

Models `λ(δ) = A · exp(−k · δ)` where `δ` = distance in ticks from best.

```python
tl.gueant.ask(window_size=None, buckets=None)   # → (A, k) or (Series, Series)
tl.gueant.bid(window_size=None, buckets=None)    # → (A, k) or (Series, Series)
tl.gueant.buckets(side, buckets=None)             # → DataFrame [delta, N, T, lambda]
```

| Param | Description |
|---|---|
| `window_size=None` | Scalar over all trades; `N` → rolling window |
| `buckets` | List of δ thresholds (e.g. `[1, 3, 5, 10]`); trades above last threshold excluded from fit; `None` = one point per integer δ |

**Distance convention**: ask side `δ = (trade_price − best_bid) / tick_size`; bid side `δ = (best_ask − trade_price) / tick_size`.

### I/O — Parquet

```python
tl.from_parquet(path, mode='lazy')
```

#### Full column schema

| # | Column | Type | Nullable | Notes |
|---|---|---|---|---|
| 1 | `timestamp` | `int64` | no | Arrival timestamp (ns since epoch). Primary sort key. |
| 2 | `exchange_timestamp` | `int64` | no | Exchange-reported timestamp (ns). Metadata only — not used by `lobpy` internals. |
| 3 | `exchange` | `string` | no | Exchange identifier, e.g. `'blofin'`. Metadata only. |
| 4 | `symbol` | `string` | no | Trading pair, e.g. `'BTC-USDC'`. Metadata only. |
| 5 | `event_type` | `string` | no | `'book_level'` / `'book_update'` / `'trade'` |
| 6 | `price` | `double` | no | Price of the level or trade. Must be `> 0`. |
| 7 | `quantity` | `double` | no | Size or trade volume. `≥ 0` for `book_update`; `> 0` for `book_level` and `trade`. |
| 8 | `side` | `string` | no | `'bid'`/`'ask'` for LOB rows; `'buy'`/`'sell'` for trade rows. |
| 9 | `sequence` | `string` | no | Exchange sequence / connection ID. Rows sharing `(timestamp, sequence)` come from the same message. Metadata only. |

> Columns 2–4 and 9 are optional extras read when present; only columns 1, 5–8 drive `lobpy` behaviour.

#### `event_type` values

| Value | Meaning | `side` domain | `quantity = 0` |
|---|---|---|---|
| `book_level` | Full LOB snapshot — all rows at the same timestamp form **one complete book state** | `bid`, `ask` | invalid |
| `book_update` | Incremental delta — rows at the same timestamp applied as one atomic batch | `bid`, `ask` | means *remove this price level* |
| `trade` | Single executed fill | `buy`, `sell` | invalid |

> `book_level` and `book_update` rows **must** use `bid`/`ask`; `trade` rows **must** use `buy`/`sell`.

#### Invariants

- **Ordering**: rows sorted by `timestamp` ascending; within the same timestamp: `book_level` → `book_update` → `trade`.
- **Uniqueness**: within one `book_level` batch, `(side, price)` must be unique.
- **No nulls**: every column is non-nullable.
- **Initial state**: a `book_update`-only file is valid but implies the book starts empty.

### CLI — Parquet validation

```bash
lobpy validate path/to/events.parquet
lobpy validate --strict --load path/to/events.parquet
```

- Exit code `0`: valid.
- Exit code `1`: validation errors (or warnings with `--warnings-as-errors`).
- `--strict`: require the full documented schema and strict nullable/type metadata.
- `--load`: additionally verifies that `TL.from_parquet(..., mode='lazy')` can load the file.
- `--json`: emit machine-readable JSON.

---

## Quick Reference — Module Exports

```python
from lobpy import LOB, LOBts, TL, itch_parser
```

| Export | Description |
|---|---|
| `LOB` | Static limit order book |
| `LOBts` | LOB time series |
| `TL` | Timeline (LOB + trades) |
| `itch_parser` | ITCH protocol parser |
