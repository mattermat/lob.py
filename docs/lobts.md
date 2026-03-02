# `LOBts` API (Time Series LOB)

### Initialization
- `LOBts(name=None, tick_size=1, mode='delta')`: Create time series LOB
  - `name`: Optional identifier for the time series
  - `tick_size`: Minimum price increment (default: 1)
  - `mode`: Storage mode - `'delta'` (store all snapshots) or `'latest'` (keep only current state)

### Core Methods
- `set_snapshot(bids, asks, timestamp=0, force=False)`: Create LOB snapshot at timestamp
  - `bids`: List of `(price, size)` tuples for bid side
  - `asks`: List of `(price, size)` tuples for ask side
  - `timestamp`: Timestamp for this snapshot
  - `force`: If `True`, overwrite existing timestamp (default raises error)

- `set_updates(updates, timestamp=0)`: Apply updates to create new snapshot
  - `updates`: List of `(side, price, size)` tuples
    - `side`: `'b'`/`'bid'` for bids, `'a'`/`'ask'` for asks
    - `price`: Price level
    - `size`: Quantity (0 to delete level)
  - `timestamp`: Timestamp for this snapshot
  - Returns: The new LOB object

- `update(side, price_level, size, timestamp=0)`: Apply single update
  - `side`: `'b'`/`'bid'` or `'a'`/`'ask'`
  - `price_level`: Price level
  - `size`: Quantity (0 to delete level)
  - `timestamp`: Timestamp for this snapshot
  - Returns: The new LOB object

### Time Indexing
- `lobts[timestamp]`: Access LOB at specific timestamp
  - Returns: `LOB` object or `None` if not found

- `lobts[start:end]`: Slice time range
  - `start`: Start timestamp (inclusive)
  - `end`: End timestamp (inclusive)
  - Returns: New `LOBts` with filtered snapshots

- `lobts.timestamps`: Property returning sorted list of timestamps
  - Returns: List of timestamps in chronological order

- `lobts.len`: Property returning number of snapshots
  - Returns: Integer count of LOB objects stored

- `lobts.len_ts`: Property returning time duration
  - Returns: `last_timestamp - first_timestamp`

- `len(lobts)`: Get number of snapshots (same as `lobts.len`)

### LOB Properties (at specific timestamp)
Access LOB properties via `lobts[timestamp]`:
- `lobts[ts].bid`: Best bid price (indexable: `bid[0]`, `bid[1]`, ...)
- `lobts[ts].ask`: Best ask price (indexable: `ask[0]`, `ask[1]`, ...)
- `lobts[ts].bidq`: Best bid quantity (indexable)
- `lobts[ts].askq`: Best ask quantity (indexable)
- `lobts[ts].vi`: Volume imbalance (indexable: `vi[0]` for top level, `vi[i]` for top i levels)
- `lobts[ts].spread`: Spread in absolute value
- `lobts[ts].spread_tick`: Spread in ticks
- `lobts[ts].spread_rel`: Spread as percentage of bid level
- `lobts[ts].midprice`: Mid-price
- `lobts[ts].vw_midprice`: Volume-weighted mid-price
- `lobts[ts].check()`: Check book consistency (returns `True`/`False`)

### Time Series Statistics
Properties returning pandas Series with timestamps as index:

- `lobts.spread`: Spread time series
- `lobts.bid`: Best bid price time series
- `lobts.ask`: Best ask price time series
- `lobts.midprice`: Mid-price time series
- `lobts.vw_midprice`: Volume-weighted mid-price time series
- `lobts.vi`: Volume imbalance time series

### Time-Based Statistics
- `lobts.arrival_frequency`: Total order arrivals (L2 quantity-based)
  - Counts quantity added to order book across all transitions
  - Includes: new levels and quantity increases at existing levels

- `lobts.cancel_frequency`: Total order cancellations (L2 quantity-based)
  - Counts quantity removed from order book across all transitions
  - Includes: full cancellations (level→0) and partial cancellations (quantity decreases)

- `lobts.update_frequency()`: Total updates (arrivals + cancellations)
  - Returns: `arrival_frequency + cancel_frequency`

### Utility Methods
- `lobts.diff(other)`: Calculate differences between two LOBts
  - `other`: Another LOBts object to compare with
  - Returns: List of `(timestamp, bid_deltas, ask_deltas)` tuples
  - Useful for comparing order book evolution

- `lobts.get_at_timestamp(timestamp)`: Get LOB at specific timestamp
  - Returns: `LOB` object or `None` if not found

- `lobts.get_range(start_ts, end_ts)`: Get time range
  - `start_ts`: Start timestamp (inclusive)
  - `end_ts`: End timestamp (inclusive)
  - Returns: New `LOBts` with filtered snapshots

### Conversion Methods
- `lobts.to_np(start_ts=None, end_ts=None)`: Export to numpy array
  - `start_ts`, `end_ts`: Optional time range filter
  - Returns: Array with shape `(n, 5)`: `[timestamp, side, level, price, size]`

- `lobts.to_pd(start_ts=None, end_ts=None)`: Export to pandas DataFrame
  - `start_ts`, `end_ts`: Optional time range filter
  - Returns: DataFrame with columns `['timestamp', 'side', 'level', 'price', 'size']`

### Export Methods
- `lobts.to_csv(path, start_ts=None, end_ts=None)`: Export to CSV file
  - `path`: File path for CSV output
  - `start_ts`, `end_ts`: Optional time range filter
  - Saves entire time series with appropriate columns

- `lobts.to_xlsx(path, start_ts=None, end_ts=None)`: Export to XLSX file
  - `path`: File path for XLSX output
  - `start_ts`, `end_ts`: Optional time range filter
  - Saves entire time series with appropriate columns

- `lobts.to_parquet(path, start_ts=None, end_ts=None)`: Export to Parquet file
  - `path`: File path for Parquet output
  - `start_ts`, `end_ts`: Optional time range filter
  - Saves entire time series with appropriate columns
  - Efficient binary format for large time series

### L2 Order Book Semantics
LOBts uses L2 (level 2) order book semantics for frequency calculations:

**Arrival Frequency**: Counts quantity added to the book
- New level arrival: full quantity at new price level
- Quantity increase: difference when existing level grows (X → Y, where Y > X)

**Cancel Frequency**: Counts quantity removed from the book
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
