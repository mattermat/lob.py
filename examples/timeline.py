"""
Example showing TimeLine (TL) usage.

TL combines LOBts (LOB time series) with trade events to enable analysis
of statistics that mix order book and execution data, such as:
- Liquidity profile (available vs. taken liquidity)
- Trade impact on the order book
- Spread dynamics around trades
- Volume-weighted statistics
"""

from lobpy import TL

# Create a TimeLine with configuration
tl = TL(
    name="BTC-USD",
    tick_size=0.5,
    lob_mode="realtime",  # 'realtime' or 'fixed'
    # 'realtime': LOB updates are in real time (sparse in time)
    # 'fixed': LOB updates at every a fixed  full snapshots
)

# Simulate a sequence of market events
# Timestamps represent microseconds (or any consistent time unit)
timestamps = [
    1000,  # t0: initial LOB snapshot
    1100,  # t1: LOB update (price changes)
    1150,  # t2: trade (buy aggressor)
    1200,  # t3: LOB update
    1250,  # t4: trade (sell aggressor)
    1300,  # t5: LOB update
    1350,  # t6: multiple trades
    1400,  # t7: LOB update
    1450,  # t8: trade
    1500,  # t9: final LOB update
]

# --- Event 1: Initial LOB snapshot ---
print("=== Adding Initial LOB Snapshot ===")
tl.add_lob_snapshot(
    timestamp=timestamps[0],
    bids=[(100.00, 1.5), (99.50, 2.3), (99.00, 1.8), (98.50, 3.0)],
    asks=[(101.00, 2.1), (101.50, 1.7), (102.00, 2.5), (102.50, 1.2)],
)
lob0 = tl.lob[timestamps[0]]
print(f"LOB at {timestamps[0]}: Bid={lob0.bid[0]}, Ask={lob0.ask[0]}")
print()

# --- Event 2: LOB update (incremental change) ---
print("=== Adding LOB Update ===")
tl.add_lob_update(
    timestamp=timestamps[1],
    updates=[
        ("b", 100.00, 2.0),  # Bid quantity increased
        ("b", 100.50, 1.0),  # New bid level
        ("a", 101.00, 1.5),  # Ask quantity decreased
    ],
)
lob1 = tl.lob[timestamps[1]]
print(f"LOB at {timestamps[1]}: Bid={lob1.bid[0]}, Ask={lob1.ask[0]}")
print()

# --- Event 3: Trade (buy aggressor takes liquidity from ask side) ---
print("=== Adding Trade (Buy Aggressor) ===")
tl.add_trade(
    timestamp=timestamps[2],
    side="b",  # 'b' = buy aggressor (takes from asks)
    price=101.00,  # Execution price
    volume=0.5,  # Trade size
)
print(f"Trade at {timestamps[2]}: Buy {0.5} @ {101.00}")
print()

# --- Event 4: LOB update ---
tl.add_lob_update(
    timestamp=timestamps[3],
    updates=[
        ("a", 101.00, 1.0),  # Ask replenished after trade
        ("b", 100.50, 0),  # Bid level cancelled
    ],
)

# --- Event 5: Trade (sell aggressor takes liquidity from bid side) ---
print("=== Adding Trade (Sell Aggressor) ===")
tl.add_trade(
    timestamp=timestamps[4],
    side="s",  # 's' = sell aggressor (takes from bids)
    price=100.00,
    volume=1.0,
)
print(f"Trade at {timestamps[4]}: Sell {1.0} @ {100.00}")
print()

# --- Event 6: LOB update ---
tl.add_lob_update(
    timestamp=timestamps[5],
    updates=[
        ("b", 100.00, 1.0),  # Bid replenished
        ("a", 101.50, 0),  # Ask cancelled
        ("a", 103.00, 2.0),  # New deeper ask level
    ],
)

# --- Event 7: Multiple trades (can batch trades at same timestamp) ---
print("=== Adding Multiple Trades ===")
tl.add_trades(
    timestamp=timestamps[6],
    trades=[
        ("b", 101.00, 0.3),
        ("b", 101.50, 0.2),
        ("s", 100.00, 0.5),
    ],
)
print(f"Multiple trades at {timestamps[6]}")
print()

# --- Events 8-9: More LOB updates and trade ---
tl.add_lob_update(
    timestamp=timestamps[7],
    updates=[
        ("b", 99.50, 3.0),
        ("a", 101.00, 2.5),
    ],
)

tl.add_trade(timestamp=timestamps[8], side="b", price=101.00, volume=1.5)

tl.add_lob_update(
    timestamp=timestamps[9],
    updates=[
        ("b", 100.00, 2.5),
        ("a", 101.00, 1.0),
    ],
)

# ==============================================================================
# 3. ACCESSING DATA
# ==============================================================================

print("=== Data Access ===")

# Get all timestamps
print(f"Total events: {len(tl)}, in {len(tl.timestamps)} timestamps")
print(f"Timestamps: {tl.timestamps[:5]}... (showing first 5)")
print()

# exit()
# print(tl.to_np())
print(tl.to_pd())

# Access LOB at specific timestamp (last LOB state at or before timestamp)
lob_at_1200 = tl.lob[timestamps[3]]
print(f"LOB at timestamp {timestamps[3]}:")
print(f"  Best bid: {lob_at_1200.bid[0]} x {lob_at_1200.bidq[0]}")
print(f"  Best ask: {lob_at_1200.ask[0]} x {lob_at_1200.askq[0]}")
print(f"  Spread: {lob_at_1200.spread}")
print()

# sliced timeline
print(f"sliced [{timestamps[7]}:{timestamps[9]}]")
print(tl[timestamps[7] : timestamps[9]].to_np())

# Get events in time range
range_tl = tl[timestamps[2] : timestamps[6]]
print(f"Events in range [{timestamps[2]}, {timestamps[6]}]: {len(range_tl)}")
print()

# Access all trades
print(f"Total trades: {len(tl.trades)}")
for trade in tl.trades[:3]:  # Show first 3
    print(f"  {trade.timestamp}: {trade.side} {trade.volume} @ {trade.price}")
print("  ...")
print()

print(tl.lob)
print(tl.trades)  # TODO: change what __repr__ returns

# Access events by type
"""
print(f"LOB events: {len(tl.lob_events)}")
print(f"Trade events: {len(tl.trade_events)}")
print()
"""


# ==============================================================================
# 4. COMBINED STATISTICS (LOB + Trades)
# ==============================================================================

# Gueant's A and k of bid and ask side on the complete timeline
A_ask, k_ask = tl.gueant.ask()
A_bid, k_bid = tl.gueant.bid()
print("λ(δ) = A * exp(-k * δ)")
print(f"  - λ(δ_ask) = {A_ask} * exp(-{k_ask} * δ_ask)")
print(f"  - λ(δ_bid) = {A_bid} * exp(-{k_bid} * δ_bid)")
print(tl.gueant.buckets("b"))


# Gueant's but with custom buckets
custom_buckets = [1, 3, 5, 10]
A_ask, k_ask = tl.gueant.ask(buckets=custom_buckets)
print(f"Custom buckets: {custom_buckets}")
print(f"λ(δ_ask) = {A_ask} * exp(-{k_ask} * δ_ask)")

# rolling Gueant's intensity function parameters (pd.Series)
Aa, ka = tl.gueant.ask(200)  # same unit as per timestamps
Ab, kb = tl.gueant.bid(200)  # same unit as per timestamps
print(ka)

# rolling Gueant's intensity function and custom buckets
Aa, ka = tl.gueant.ask(200, buckets=custom_buckets)

print()
# OHLC Candles
print("ohcl 1 second")
print(tl.ohlc("1s"))
print()
print("ohcl 1 minute")
print(tl.ohlc("1m"))

print()
# Realized Volatility
tl.realized_vol()  # realized volatility on all the trade data
tl[1150:1350].realized_vol()  # realized volatility on the sliced data
tl.realized_vol(200)  # rolling realized volatility

print(f"realized vol - full timeline: {tl.realized_vol()}")
print(f"realized vol - from 1150 to 1350: {tl[1150:1350].realized_vol()}")
print(f"rolling realized vol - 200ns window: {tl.realized_vol(200)}")

print()
# VPIN
vpin = tl.vpin()
print(vpin)
# rolling window VPIN (pd.Series)
vpin_ts = tl.vpin(200)
print(vpin_ts)
