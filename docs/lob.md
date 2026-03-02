# `LOB` API (static data structure for the limit order book)

### Methods
- `set_snapshot`
- `update`
- `set_update`

### Properties
- `bid`: best bid price
- `ask`: best ask price
- `bidq`: best bid size
- `askq`: best ask size
- `vi`: volume imbalance
- `bid[0]`: bid price at level 0 - equals to `bid`
- `bid[i]`: bid price at level i
- `ask[0]`: ask price at level 0 - equals to `ask`
- `ask[i]`: ask price at level i
- `bidq[0]`: bid quantity at level 0 - equals to `bidq`
- `bidq[i]`: bid quantity at level i
- `askq[0]`: ask quantity at level 0 - equals to `askq`
- `askq[i]`: ask quantity at level i
- `vi[0]`: volume imbalance of the first level - equals to `vi`
- `vi[i]`: volume imbalance of the top i levels
- `spread`: spread in absolute value
- `spread_tick`: spread in ticks
- `spread_rel`: spread in percentage of the bid level
- `midprice`: mid-price
- `vw_midprice`: volume-weighted mid-price

### Other methods
- `check()`: Check consistency of the order book
  - Returns `True` if the book is consistent (best bid < best ask or one side is empty)
  - Returns `False` if the book is crossed (best bid >= best ask)
  - Useful for validating order book state before processing or after updates
- `get_slippage(volume, side=['midprice', 'ask', 'bid'])`: calculate the slippage from the top level (from the midprice is not declared)

### Export Methods
#### Numpy and Pandas Export
- `to_np(side=None, nlevels=None)`: Export order book to numpy array
  - `side`: `'b'` for bids, `'a'` for asks, or `None` for both sides
  - `nlevels`: number of top levels to export (default: all levels)
  - Returns 2D array with shape `(n, 2)` [price, size] when side specified
  - Returns 2D array with shape `(n, 3)` [side, price, size] when side=None
  - When both sides, bids come first (best to worst), then asks (best to worst)

- `to_pd(side=None, nlevels=None)`: Export order book to pandas DataFrame
  - `side`: `'b'` for bids, `'a'` for asks, or `None` for both sides
  - `nlevels`: number of top levels to export (default: all levels)
  - Returns DataFrame with columns `['price', 'size']` when side specified
  - Returns DataFrame with columns `['price', 'size', 'side']` when side=None
  - Side column contains `'b'` for bids and `'a'` for asks

#### File Export
- `to_csv(path, side=None, nlevels=None)`: Export order book to CSV file
  - `path`: file path for CSV output
  - `side`, `nlevels`: same as to_np/to_pd
  - Saves current snapshot with appropriate columns

- `to_xlsx(path, side=None, nlevels=None)`: Export order book to XLSX file
  - `path`: file path for XLSX output
  - `side`, `nlevels`: same as to_np/to_pd
  - Saves current snapshot with appropriate columns

- `to_parquet(path, side=None, nlevels=None)`: Export order book to Parquet file
  - `path`: file path for Parquet output
  - `side`, `nlevels`: same as to_np/to_pd
  - Saves current snapshot with appropriate columns
  - Efficient binary format for large order books

