# `LOB` API (static data structure for the limit order book)

### Constructor
`LOB(name=None, tick_size=1, *, bids=None, asks=None)`
- `name`: Optional identifier (auto-generated if `None`)
- `tick_size`: Minimum price increment (default: `1`)
- `bids`, `asks`: Optional initial levels as lists of `(price, size)` tuples

### Methods
- `set_snapshot(bids, asks, timestamp=0)`: Replace the full book state
  - `bids`, `asks`: lists of `(price, size)` tuples
- `set_updates(updates, timestamp=0)`: Apply incremental updates to the current book
  - `updates`: list of `(side, price, size)` tuples
    - `side`: `'b'`/`'bid'` for bids, `'a'`/`'ask'` for asks
    - `size=0` removes the price level
- `update(side, price_level, size, timestamp=0)`: Apply a single update
  - `side`: `'b'`/`'bid'` or `'a'`/`'ask'`
  - `size=0` deletes the level

### Properties
- `name`: identifier string
- `tick_size`: minimum price increment (read/write; setting it updates the C layer)
- `timestamp`: current timestamp (read/write)
- `bid`: best bid price (via `PriceAccessor`)
- `ask`: best ask price (via `PriceAccessor`)
- `bidq`: best bid size (via `QuantityAccessor`)
- `askq`: best ask size (via `QuantityAccessor`)
- `vi`: volume imbalance (via `VolumeImbalanceAccessor`)
- `spread`: spread in absolute value
- `spread_tick`: spread in ticks
- `spread_rel`: spread as percentage of the best bid
- `midprice`: mid-price
- `vw_midprice`: volume-weighted mid-price

### Indexable accessors
- `bid[0]`: best bid price (equals `bid`); `bid[i]`: bid price at level `i` (0.0 if out of range)
- `ask[0]`: best ask price (equals `ask`); `ask[i]`: ask price at level `i`
- `bidq[0]`: best bid quantity (equals `bidq`); `bidq[i]`: bid quantity at level `i`
- `askq[0]`: best ask quantity (equals `askq`); `askq[i]`: ask quantity at level `i`
- `vi[0]`: volume imbalance of the first level (equals `vi`); `vi[i]`: imbalance of the top `i+1` levels

All accessors support arithmetic (`+`, `-`, `*`, `/`, comparisons, `float()`, `int()`).

### Other methods
- `at(side, price)`: return quantity at an exact price level on the given side
  - `side`: `'b'`/`'bid'` or `'a'`/`'ask'`
- `check()`: check consistency of the order book
  - Returns `True` if consistent (best bid < best ask or one side is empty)
  - Returns `False` if crossed
- `get_slippage(volume, side='midprice')`: calculate slippage for a given volume
  - `side`: `'midprice'` (returns 0), `'ask'`/`'a'`, or `'bid'`/`'b'`
  - Returns the per-unit price impact relative to midprice
- `aggq(side, nlevel=None, ticks=None, price=None)`: aggregate quantity on one side
  - Exactly one of `nlevel` (top N levels), `ticks` (within N ticks of best), or `price` (at or better than given price) must be specified
- `len_in_tick(side, price)`: distance in ticks from the best price to the given price
- `diff(other)`: compute the updates needed to go from `self` to `other`
  - Returns a list of `(side, price, size)` tuples
- `get_delta(bids, asks, timestamp=0)`: replace the book with a new snapshot and return `(bid_deltas, ask_deltas)` describing the change

### Export Methods
#### Numpy and Pandas Export
- `to_np(side=None, nlevels=None)`: export order book to numpy array
  - `side`: `'b'` for bids, `'a'` for asks, or `None` for both sides
  - `nlevels`: number of top levels to export (default: all levels)
  - Returns 2D array with shape `(n, 2)` [price, size] when side specified
  - Returns 2D array with shape `(n, 3)` [side, price, size] when side=None
  - When both sides, bids come first (best to worst), then asks (best to worst)

- `to_pd(side=None, nlevels=None)`: export order book to pandas DataFrame
  - `side`: `'b'` for bids, `'a'` for asks, or `None` for both sides
  - `nlevels`: number of top levels to export (default: all levels)
  - Returns DataFrame with columns `['price', 'size']` when side specified
  - Returns DataFrame with columns `['price', 'size', 'side']` when side=None

#### File Export
- `to_csv(path, side=None, nlevels=None)`: export order book to CSV file
- `to_xlsx(path, side=None, nlevels=None)`: export order book to XLSX file
- `to_parquet(path, side=None, nlevels=None)`: export order book to Parquet file
