# Parquet Data Contract

> The canonical schema for `lobpy` time-series parquet files — LOB snapshots, incremental updates, and trades in a single flat table.

---

## Schema

| # | Column | Type | Nullable | Description |
|---|---|---|---|---|
| 1 | `timestamp` | `int64` | no | Arrival timestamp (nanoseconds since epoch). Primary sort key. Events sharing the same timestamp are processed as one batch. |
| 2 | `exchange_timestamp` | `int64` | no | Exchange-reported timestamp (nanoseconds since epoch). May lag `timestamp`. Metadata only — not used by `lobpy` internals. |
| 3 | `exchange` | `string` | no | Exchange identifier, e.g. `'blofin'`, `'mexc'`. Metadata only. |
| 4 | `symbol` | `string` | no | Trading pair, e.g. `'BTC-USDC'`, `'BTCUSDT'`. Metadata only. |
| 5 | `event_type` | `string` | no | Event classification — see [values](#event_type-values) below. |
| 6 | `price` | `double` | no | Price of the level or trade. Always **strictly positive** (`> 0`). |
| 7 | `quantity` | `double` | no | Size at the price level, or trade volume. **Non-negative** (`≥ 0`). `0` means *delete this level* (only for `book_update`). |
| 8 | `side` | `string` | no | Side qualifier — semantics depend on `event_type`, see [values](#side-values) below. |
| 9 | `sequence` | `string` | no | Exchange sequence / connection ID for the event batch. Rows sharing the same `(timestamp, sequence)` pair originate from the same exchange message. Metadata only. |

### PyArrow schema definition

```python
import pyarrow as pa

SCHEMA = pa.schema([
    pa.field("timestamp",           pa.int64(),  nullable=False),
    pa.field("exchange_timestamp",  pa.int64(),  nullable=False),
    pa.field("exchange",            pa.string(), nullable=False),
    pa.field("symbol",              pa.string(), nullable=False),
    pa.field("event_type",          pa.string(), nullable=False),
    pa.field("price",               pa.float64(),nullable=False),
    pa.field("quantity",            pa.float64(),nullable=False),
    pa.field("side",                pa.string(), nullable=False),
    pa.field("sequence",            pa.string(), nullable=False),
])
```

---

## Enumerated Values

### `event_type` values

| Value | Meaning | Rows per batch | `side` domain | `quantity = 0` |
|---|---|---|---|---|
| `book_level` | Full LOB snapshot — all rows at the same `(timestamp, sequence)` together describe the **entire** order book at that moment. | ≥ 1 | `bid`, `ask` | invalid (level should not exist) |
| `book_update` | Incremental delta — a change to a single price level. Multiple rows at the same `(timestamp, sequence)` form one update batch applied atomically. | ≥ 1 | `bid`, `ask` | means *remove this price level* |
| `trade` | An executed trade (fill). Each row is one fill; multiple fills can share a timestamp. | ≥ 1 | `buy`, `sell` | invalid (trade has zero volume) |

### `side` values

| Value | Used by | Meaning |
|---|---|---|
| `bid` | `book_level`, `book_update` | Bid side of the LOB |
| `ask` | `book_level`, `book_update` | Ask side of the LOB |
| `buy` | `trade` | Buy aggressor — taker lifted the ask |
| `sell` | `trade` | Sell aggressor — taker hit the bid |

> **Cross-constraint**: `book_level` and `book_update` rows **must** use `bid`/`ask`. `trade` rows **must** use `buy`/`sell`.

---

## Invariants

### Ordering

- Rows **must** be sorted by `timestamp` in ascending order.
- Within the same `timestamp`, processing order is: `book_level` → `book_update` → `trade`.

### Uniqueness

- Within a single `book_level` batch (same `timestamp`), the combination `(side, price)` **must** be unique — a price level appears at most once per side.

### Nulls

- **No column may contain null values.** All fields are non-nullable.

### Numeric ranges

| Column | Constraint |
|---|---|
| `price` | `> 0` (strictly positive) |
| `quantity` | `≥ 0` for `book_update`; `> 0` for `book_level` and `trade` |
| `timestamp` | `≥ 0`; monotonically non-decreasing across rows |
| `exchange_timestamp` | `≥ 0` |

### Referential

- `book_update` rows **must** be preceded by at least one `book_level` batch (the initial snapshot) at an earlier or equal timestamp, unless the application explicitly handles an initially empty book.
- There is no constraint linking `trade.price` to existing LOB levels (trades can occur at any price).

---

## Processing Semantics (how `lobpy` reads this)

```
for each unique timestamp (ascending):
    1. collect all book_level rows  → build full snapshot
    2. collect all book_update rows → apply deltas to current book
    3. collect all trade rows       → record as Trade events
```

`TL.from_parquet(path, mode='lazy')` reads the file once, vectorises by `event_type`, and reconstructs LOB states on demand.

---

## Minimal Valid File

The smallest valid file must contain at least one `book_level` batch:

| timestamp | exchange_timestamp | exchange | symbol | event_type | price | quantity | side | sequence |
|---|---|---|---|---|---|---|---|---|
| `1000` | `1000` | `any` | `ANY` | `book_level` | `100.0` | `1.0` | `bid` | `1` |
| `1000` | `1000` | `any` | `ANY` | `book_level` | `101.0` | `1.0` | `ask` | `1` |

A file with **no** `book_level` rows (only `book_update` / `trade`) is parseable but implies the book starts empty and updates are applied to a blank slate.
