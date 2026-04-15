# `TL` API (TimeLine — LOB time series + trade events)

`TL` is a unified container for limit order book snapshots/updates and trade events. It combines `LOBts` (order book time series) with trade records to enable mixed analysis of order book state and execution data.

### Constructor
`TL(name=None, tick_size=1, lob_mode='delta', update_type='realtime', timestamp_unit='ns')`
- `name`: Optional identifier for this timeline (auto-generated if `None`)
- `tick_size`: Minimum price increment (default: `1`)
- `lob_mode`: `'delta'` (incremental updates) or `'snapshot'` (full book at each update)
- `update_type`: `'realtime'` (sparse updates) or `'fixed'` (regular intervals)
- `timestamp_unit`: Unit of all timestamps — `'s'`, `'ms'`, `'us'`, or `'ns'` (default: `'ns'`)
  - Used to convert `ohlc` periods and rolling window sizes to the correct scale

### LOB methods
- `add_lob_snapshot(timestamp, bids, asks)`: Record a full order book snapshot
  - `bids`, `asks`: lists of `(price, quantity)` tuples
  - Replaces the full book state at this timestamp
- `add_lob_update(timestamp, updates)`: Apply incremental LOB changes at the given timestamp
  - In `lob_mode='delta'`: updates are applied on top of the previous snapshot
  - In `lob_mode='snapshot'`: updates replace the book entirely (full snapshot semantics)
  - `updates`: list of `(side, price, quantity)` tuples
    - `side`: `'b'`/`'bid'` for bids, `'a'`/`'ask'` for asks
    - `quantity=0` removes the price level

### LOB properties
- `lob`: the underlying `LOBts`, indexable by timestamp
  - `tl.lob[ts]`: returns the `LOB` object at that timestamp (last known state at or before `ts`)
  - `tl.lob[ts].bid[0]`: best bid price at `ts`
  - `tl.lob[ts].ask[0]`: best ask price at `ts`
  - `tl.lob[ts].bidq[0]`: best bid quantity at `ts`
  - `tl.lob[ts].askq[0]`: best ask quantity at `ts`
  - `tl.lob[ts].spread`: bid-ask spread at `ts`
  - `tl.lob[ts].midprice`: mid-price at `ts`

### Trade methods
- `add_trade(timestamp, side, price, volume)`: Record a single trade event
  - `side`: `'b'` (buy aggressor, takes from asks) or `'s'` (sell aggressor, takes from bids)
  - `price`: execution price
  - `volume`: trade size
- `add_trades(timestamp, trades)`: Record multiple trades at the same timestamp
  - `trades`: list of `(side, price, volume)` tuples

### Trade properties
- `trades`: list of `Trade` objects in insertion order
  - each `Trade` has `.timestamp`, `.side`, `.price`, `.volume`

### Data access
- `timestamps`: sorted list of all event timestamps (LOB and trades, deduplicated)
- `len(tl)`: total number of events (LOB snapshots/updates + individual trades)

### Slicing
- `tl[start:stop]`: returns a new `TL` containing only events in the inclusive range `[start, stop]`
  - `start` or `stop` can be `None` for open-ended ranges
  - the sliced `TL` inherits `tick_size`, `lob_mode`, `update_type`, and `timestamp_unit`
- `rolling(window_size)`: generator yielding `TL` slices over a rolling window
  - for each event timestamp `ts`, yields `tl[ts - window_size : ts]`
  - window size is in the same units as timestamps

### Export
- `to_pd()`: export the full timeline as a pandas DataFrame sorted by timestamp
  - columns: `timestamp`, `type`, `side`, `level`, `price`, `size`
  - `type`: `'lob'` or `'trade'`
  - `level`: integer depth level for LOB rows; `NaN` for trade rows
- `to_np()`: export as a numpy object array with the same 6 columns
  - returns shape `(n_rows, 6)`; empty timeline returns shape `(0, 6)`

### Analytics
- `ohlc(period)`: compute OHLC candles from trade data bucketed into fixed time windows
  - `period`: one of `'1s'`, `'5s'`, `'1m'`, `'15m'`, `'1h'`, `'24h'`
  - period durations are scaled by `timestamp_unit` (e.g. `'1m'` = 60 × units-per-second)
  - returns a DataFrame indexed by candle-open timestamp with columns: `open`, `high`, `low`, `close`, `volume`, `count`
  - returns an empty DataFrame if no trades are recorded

- `realized_vol(window_size=None)`: realized volatility = `sqrt(Σ log²(pᵢ/pᵢ₋₁))` over trade prices
  - `window_size=None`: returns a scalar over all trades
  - `window_size=N`: returns a `pd.Series` of realized vol values over rolling windows of size `N`, indexed by end timestamp
  - returns `nan` if fewer than 2 trades are available in the window

- `volume_buckets(bucket_size=None, include_partial=False)`: partition trades into fixed-volume buckets
  - trades are processed in timestamp order; a trade is split across bucket boundaries when it would overflow the current bucket
  - `bucket_size`: volume per bucket; if `None`, computed as `total_volume / 50`
  - `include_partial`: if `True`, the last partial bucket (volume < `bucket_size`) is included
  - returns a DataFrame indexed by bucket number (0-based) with columns: `buy_volume`, `sell_volume`

- `vpin(window_size=None, bucket_size=None)`: VPIN (Volume-Synchronized Probability of Informed Trading)
  - formula: `VPIN = Σ|V_buy[i] - V_sell[i]| / (n × bucket_size)` where the sum runs over all complete volume buckets
  - `window_size=None`: returns a scalar over all trades
  - `window_size=N`: returns a `pd.Series` of VPIN values over rolling windows of size `N`, indexed by end timestamp
  - `bucket_size=None`: computed once as `total_volume / 50` and reused across all windows
  - values near 0 indicate balanced order flow; values near 1 indicate strong directional (potentially informed) flow

### Guéant intensity function — `tl.gueant`
Models trade arrival intensity as `λ(δ) = A · exp(−k · δ)`, where `δ` is the distance in ticks from best bid/ask.

Reference: Guéant, Lehalle, Fernandez-Tapia (2013).

- `tl.gueant.ask(window_size=None, buckets=None)`: estimate `(A, k)` for the ask side (buy-aggressor trades)
  - `window_size=None`: returns `(A, k)` scalars over the full timeline
  - `window_size=N`: returns `(pd.Series_A, pd.Series_k)` over rolling windows, indexed by end timestamp
  - `buckets`: optional list of δ thresholds (e.g. `[1, 3, 5, 10]`); each threshold defines a bin upper bound; trades above the last threshold go into a silent overflow bin excluded from the fit; if `None`, one data point per integer δ is used
- `tl.gueant.bid(window_size=None, buckets=None)`: same as `ask`, but for the bid side (sell-aggressor trades)
- `tl.gueant.buckets(side, buckets=None)`: inspect the empirical intensity table `λ̂(δ) = N(δ) / T(δ)` before fitting
  - `side`: `'a'` (ask) or `'b'` (bid)
  - returns a DataFrame with columns `[delta, N, T, lambda]`
    - `N(δ)`: number of trades at tick distance `δ`
    - `T(δ)`: total time the book exposed liquidity at distance `δ`
    - `lambda`: empirical intensity `N/T`; `nan` if no exposure or no trades
  - distance conventions:
    - ask side: `δ = (trade_price − best_bid) / tick_size`
    - bid side: `δ = (best_ask − trade_price) / tick_size`

### I/O
- `from_parquet(path, mode='lazy')`: load LOB and trade events from a Parquet file into this `TL` instance
  - expected columns: `timestamp`, `event_type`, `side`, `price`, `quantity`
  - `event_type` values:
    - `'book_level'`: rows forming a full LOB snapshot; all rows at the same timestamp form one snapshot
    - `'book_update'`: incremental LOB changes; all rows at the same timestamp are one update batch
    - `'trade'`: individual trade events
  - `side` values: LOB rows use `'bid'`/`'ask'`; trade rows use `'buy'`/`'sell'`
  - within the same timestamp, processing order is: `book_level` → `book_update` → `trade`
  - `mode`: `'lazy'` (default) or `'eager'`
    - **`'lazy'`**: LOB data is loaded as checkpoints + a delta log (vectorised via pyarrow); `build_checkpoints()` is called automatically. Dramatically lower memory for large files.
    - **`'eager'`**: every snapshot/update is expanded into a full `LobBook` in C memory immediately. Simpler but uses O(N × L) memory.
