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
import numpy as np  # For intensity function examples

# Create a TimeLine with configuration
tl = TL(
    name="BTC-USD",
    tick_size=0.5,
    lob_mode='realtime',  # 'realtime' or 'fixed'
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
    asks=[(101.00, 2.1), (101.50, 1.7), (102.00, 2.5), (102.50, 1.2)]
)
print(f"LOB at {timestamps[0]}: Bid={tl.lob[timestamps[0]].bid[0]}, Ask={tl.lob[timestamps[0]].ask[0]}")
print()

# --- Event 2: LOB update (incremental change) ---
print("=== Adding LOB Update ===")
tl.add_lob_update(
    timestamp=timestamps[1],
    updates=[
        ('b', 100.00, 2.0),   # Bid quantity increased
        ('b', 100.50, 1.0),   # New bid level
        ('a', 101.00, 1.5),   # Ask quantity decreased
    ]
)
print(f"LOB at {timestamps[1]}: Bid={tl.lob[timestamps[1]].bid[0]}, Ask={tl.lob[timestamps[1]].ask[0]}")
print()

# --- Event 3: Trade (buy aggressor takes liquidity from ask side) ---
print("=== Adding Trade (Buy Aggressor) ===")
tl.add_trade(
    timestamp=timestamps[2],
    side='b',        # 'b' = buy aggressor (takes from asks)
    price=101.00,    # Execution price
    volume=0.5       # Trade size
)
print(f"Trade at {timestamps[2]}: Buy {0.5} @ {101.00}")
print()

# --- Event 4: LOB update ---
tl.add_lob_update(
    timestamp=timestamps[3],
    updates=[
        ('a', 101.00, 1.0),   # Ask replenished after trade
        ('b', 100.50, 0),     # Bid level cancelled
    ]
)

# --- Event 5: Trade (sell aggressor takes liquidity from bid side) ---
print("=== Adding Trade (Sell Aggressor) ===")
tl.add_trade(
    timestamp=timestamps[4],
    side='s',        # 's' = sell aggressor (takes from bids)
    price=100.00,
    volume=1.0
)
print(f"Trade at {timestamps[4]}: Sell {1.0} @ {100.00}")
print()

# --- Event 6: LOB update ---
tl.add_lob_update(
    timestamp=timestamps[5],
    updates=[
        ('b', 100.00, 1.0),   # Bid replenished
        ('a', 101.50, 0),     # Ask cancelled
        ('a', 103.00, 2.0),   # New deeper ask level
    ]
)

# --- Event 7: Multiple trades (can batch trades at same timestamp) ---
print("=== Adding Multiple Trades ===")
tl.add_trades(
    timestamp=timestamps[6],
    trades=[
        ('b', 101.00, 0.3),
        ('b', 101.50, 0.2),
        ('s', 100.00, 0.5),
    ]
)
print(f"Multiple trades at {timestamps[6]}")
print()

# --- Events 8-9: More LOB updates and trade ---
tl.add_lob_update(
    timestamp=timestamps[7],
    updates=[
        ('b', 99.50, 3.0),
        ('a', 101.00, 2.5),
    ]
)

tl.add_trade(
    timestamp=timestamps[8],
    side='b',
    price=101.00,
    volume=1.5
)

tl.add_lob_update(
    timestamp=timestamps[9],
    updates=[
        ('b', 100.00, 2.5),
        ('a', 101.00, 1.0),
    ]
)

# ==============================================================================
# 3. ACCESSING DATA
# ==============================================================================

print("=== Data Access ===")

# Get all timestamps
print(f"Total events: {len(tl)}, in {len(tl.timestamps)} timestamps")
print(f"Timestamps: {tl.timestamps[:5]}... (showing first 5)")
print()

#print(tl.to_np())
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
print(tl[timestamps[7]:timestamps[9]].to_np())

# Get events in time range
range_tl = tl[timestamps[2]:timestamps[6]]
print(f"Events in range [{timestamps[2]}, {timestamps[6]}]: {len(range_tl)}")
print()

# Access all trades
print(f"Total trades: {len(tl.trades)}")
for trade in tl.trades[:3]:  # Show first 3
    print(f"  {trade.timestamp}: {trade.side} {trade.volume} @ {trade.price}")
print("  ...")
print()

print(tl.lob)
print(tl.trades) # TODO: change what __repr__ returns

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
print('λ(δ) = A * exp(-k * δ)')
print(f'  - λ(δ_ask) = {A_ask} * exp(-{k_ask} * δ_ask)')
print(f'  - λ(δ_bid) = {A_bid} * exp(-{k_bid} * δ_bid)')
print(tl.gueant.buckets('b'))


# Gueant's but with custom buckets
custom_buckets = [1, 3, 5, 10]
A_ask, k_ask = tl.gueant.ask(buckets=custom_buckets)
print(f"Custom buckets: {custom_buckets}")
print(f'λ(δ_ask) = {A_ask} * exp(-{k_ask} * δ_ask)')

# rolling Gueant's intensity function parameters (pd.Series)
Aa, ka = tl.gueant.ask(200) # same unit as per timestamps
Ab, kb = tl.gueant.bid(200) # same unit as per timestamps
print(ka)

# rolling Gueant's intensity function and custom buckets
Aa, ka = tl.gueant.ask(200, buckets=custom_buckets)

print()
# OHLC Candles
print("ohcl 1 second")
print(tl.ohlc('1s'))
print()
print("ohcl 1 minute")
print(tl.ohlc('1m'))

print()
# Realized Volatility
tl.realized_vol() # realized volatility on all the trade data
tl[1150:1350].realized_vol() # realized volatility on the sliced data
tl.realized_vol(200) # rolling realized volatility

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

exit()

print("=== Combined Statistics ===")

# --- Liquidity profile: how much liquidity available vs. taken ---
# This analyzes the distribution of liquidity across price levels:
# - Available liquidity: total quantity posted in the LOB at each price
# - Taken liquidity: total volume executed (trades) at each price
# - Useful for understanding where liquidity concentrates vs. where it's consumed
liquidity_profile = tl.liquidity_profile(
    price_range=(99.0, 103.0),  # Price range to analyze
    bin_size=0.5,                # Bin prices into 0.5 intervals
    side=None,                   # None (both), 'b' (bids only), 'a' (asks only)
    normalize_time=True          # Normalize by time spent at each level
)

print("Liquidity Profile (available vs. taken):")
print(liquidity_profile.head())
# Expected output (pandas DataFrame):
#     price  available_qty  taken_qty  lob_appearances  trade_count  avg_available  ratio_taken
# 0   99.00          15.3        0.0               5            0          3.06         0.00
# 1   99.50          18.4        0.0               6            0          3.07         0.00
# 2  100.00          25.5        1.5               8            2          3.19         0.06
# 3  100.50          12.0        0.0               3            0          4.00         0.00
# 4  101.00          32.1        2.3               9            3          3.57         0.07
#
# Columns explanation:
# - price: Price level (binned)
# - available_qty: Sum of all quantities posted at this level across all LOB snapshots
# - taken_qty: Sum of all trade volumes executed at this level
# - lob_appearances: Number of LOB snapshots containing this price level
# - trade_count: Number of trades executed at this price
# - avg_available: average quantity when level is present (available_qty / lob_appearances)
# - ratio_taken: taken_qty / available_qty (how much of posted liquidity was consumed)
"""
  Concrete Example (from appendix in the file):

  For price level 101.00:
  - Appeared in 5 LOB snapshots with quantities: 2.1, 1.5, 1.0, 2.5, 1.0
  - Had 3 trades: 0.5, 0.3, 1.5
  - Result: available_qty=8.1, taken_qty=2.3, ratio_taken=0.284 (28.4%)

  Key Insights:

  - High ratio_taken (>0.5): Price level is actively traded ("hot zone")
  - Low ratio_taken (<0.1): Stale/passive liquidity
  - High available, low trades: Resistance/support level
  - Asymmetry bid/ask: Directional pressure
"""
print()

# --- Trade impact: how trades affected spread/midprice ---
trade_impact = tl.trade_impact(window_after=100)  # Look 100 time units after each trade
print("Trade Impact Analysis:")
print(trade_impact.head())
print()

# --- Spread at trade time ---
spread_at_trades = tl.spread_at_trades()
print("Spread when trades occurred:")
print(spread_at_trades.head())
print()

# --- Volume statistics ---
volume_stats = tl.volume_stats()
print("Volume Statistics:")
print(f"  Total buy volume: {volume_stats['total_buy_volume']}")
print(f"  Total sell volume: {volume_stats['total_sell_volume']}")
print(f"  Buy trade count: {volume_stats['buy_count']}")
print(f"  Sell trade count: {volume_stats['sell_count']}")
print(f"  Average trade size: {volume_stats['avg_trade_size']:.2f}")
print()

# --- Aggressive vs. passive volume (based on trade side) ---
aggressive_volume = tl.aggressive_volume()
print("Aggressive Volume (by side):")
print(f"  Buy aggressive (took asks): {aggressive_volume['buy']}")
print(f"  Sell aggressive (took bids): {aggressive_volume['sell']}")
print(f"  Ratio (buy/sell): {aggressive_volume['ratio']:.2f}")
print()

# --- Time-weighted spread ---
tw_spread = tl.time_weighted_spread(start_ts=timestamps[0], end_ts=timestamps[-1])
print(f"Time-weighted average spread: {tw_spread:.3f}")
print()

# --- Order flow imbalance (combining LOB imbalance with trade flow) ---
ofi = tl.order_flow_imbalance()
print("Order Flow Imbalance:")
print(ofi.head())
print()

# --- Price impact per trade ---
price_impact = tl.price_impact_per_trade()
print("Price Impact (midprice change after each trade):")
print(price_impact.head())
print()

# --- Intensity function parameters (Guéant et al.) ---
# From "Dealing with the Inventory Risk: A solution to the market making problem"
# Guéant, Lehalle, Fernandez-Tapia (2013)
# Models trade arrival intensity as: λ(δ) = A * exp(-k * δ)
# where δ = distance in ticks from best bid/ask
print("=== Intensity Function Parameters (Guéant) ===")
print("Exponential model: λ(δ) = A * exp(-k * δ)")
print(f"  A (scale parameter): {tl.gueant_A:.4f}")
print(f"  k (decay rate): {tl.gueant_k:.4f}")
print()
print("Interpretation:")
print(f"  - Base intensity (at best price): {tl.gueant_A:.4f} trades/time unit")
print(f"  - At δ=1 tick: {tl.gueant_A * np.exp(-tl.gueant_k):.4f} trades/time unit")
print(f"  - At δ=5 ticks: {tl.gueant_A * np.exp(-5*tl.gueant_k):.4f} trades/time unit")
print(f"  - Half-life distance: {np.log(2)/tl.gueant_k:.2f} ticks")
print()

# Access per-side parameters
print("Side-specific parameters:")
print(f"  Ask side: A={tl.gueant_A_ask:.4f}, k={tl.gueant_k_ask:.4f}")
print(f"  Bid side: A={tl.gueant_A_bid:.4f}, k={tl.gueant_k_bid:.4f}")
print()

# Get full intensity function (callable)
intensity_ask = tl.get_intensity_function(side='a', model='gueant')
intensity_bid = tl.get_intensity_function(side='b', model='gueant')
print("Intensity at different tick distances:")
print(f"  Ask: λ(0)={intensity_ask(0):.4f}, λ(1)={intensity_ask(1):.4f}, λ(5)={intensity_ask(5):.4f}")
print(f"  Bid: λ(0)={intensity_bid(0):.4f}, λ(1)={intensity_bid(1):.4f}, λ(5)={intensity_bid(5):.4f}")
print()

# Plot intensity decay (if matplotlib available)
# import matplotlib.pyplot as plt
# delta_range = np.arange(0, 20)
# plt.plot(delta_range, [intensity_ask(d) for d in delta_range], label='Ask')
# plt.plot(delta_range, [intensity_bid(d) for d in delta_range], label='Bid')
# plt.xlabel('Distance (ticks)')
# plt.ylabel('Intensity λ(δ)')
# plt.title('Guéant Intensity Function')
# plt.legend()
# plt.show()

# --- Linear intensity parameters (Ho-Stoll) ---
# From "Optimal dealer pricing under transactions and return uncertainty"
# Ho and Stoll (1981)
# Models trade arrival intensity as: λ(δ) = A - k * δ
print("=== Intensity Function Parameters (Ho-Stoll) ===")
print("Linear model: λ(δ) = A - k * δ")
print(f"  k (linear decay rate): {tl.ho_k:.4f}")
print()
print("Interpretation:")
print(f"  - Intensity decreases by {tl.ho_k:.4f} per tick away from best price")
print(f"  - At δ=1 tick: intensity reduced by {tl.ho_k:.4f}")
print(f"  - At δ=5 ticks: intensity reduced by {5*tl.ho_k:.4f}")
print()

# Side-specific Ho-Stoll parameters
print("Side-specific parameters:")
print(f"  Ask side: k={tl.ho_k_ask:.4f}")
print(f"  Bid side: k={tl.ho_k_bid:.4f}")
print()

# Get Ho-Stoll intensity function
intensity_ho_ask = tl.get_intensity_function(side='a', model='ho-stoll')
intensity_ho_bid = tl.get_intensity_function(side='b', model='ho-stoll')
print("Intensity at different tick distances:")
print(f"  Ask: λ(0)={intensity_ho_ask(0):.4f}, λ(1)={intensity_ho_ask(1):.4f}, λ(5)={intensity_ho_ask(5):.4f}")
print(f"  Bid: λ(0)={intensity_ho_bid(0):.4f}, λ(1)={intensity_ho_bid(1):.4f}, λ(5)={intensity_ho_bid(5):.4f}")
print()

# --- VPIN (Volume-Synchronized Probability of Informed Trading) ---
# From "VPIN and the Flash Crash" Easley, López de Prado, O'Hara (2012)
# Measures order flow toxicity / probability of informed trading
print("=== VPIN (Volume-Synchronized Probability of Informed Trading) ===")
print(f"VPIN: {tl.VPIN:.4f}")
print()
print("Interpretation:")
print(f"  - VPIN ∈ [0, 1]: Current value {tl.VPIN:.4f}")
if tl.VPIN < 0.3:
    print("  - LOW toxicity: Order flow is relatively balanced and uninformed")
elif tl.VPIN < 0.7:
    print("  - MODERATE toxicity: Some directional order flow imbalance")
else:
    print("  - HIGH toxicity: Strong directional pressure, potential informed trading")
print()

# VPIN calculation with custom parameters
vpin_custom = tl.calculate_vpin(
    bucket_size=50,      # Volume per bucket
    num_buckets=50,      # Number of buckets for VPIN calculation
    method='bulk'        # 'bulk' or 'trade' classification
)
print(f"VPIN (custom params): {vpin_custom:.4f}")
print()

# VPIN time series
vpin_ts = tl.vpin_ts(
    bucket_size=50,
    num_buckets=50,
    rolling=True         # Calculate rolling VPIN
)
print("VPIN Time Series (rolling):")
print(vpin_ts.head())
print()

# Trade classification for VPIN (Bulk Volume Classification)
# Classifies each volume bucket as buy or sell based on price movement
trade_classification = tl.get_trade_classification(method='bulk')
print("Trade Classification (for VPIN):")
print(trade_classification.head())
print()

print("VPIN Use Cases:")
print("  1. Risk management: High VPIN signals increased adverse selection risk")
print("  2. Market making: Widen spreads when VPIN is high")
print("  3. Execution: Avoid trading during high VPIN periods (toxic flow)")
print("  4. Liquidity provision: Reduce inventory when VPIN spikes")
print("  5. Market monitoring: Detect potential flash crash conditions")
print()

# ==============================================================================
# 5. FILTERING AND QUERIES
# ==============================================================================

print("=== Filtering and Queries ===")

# Filter by event type
buy_trades = tl.filter_trades(side='b')
print(f"Buy trades only: {len(buy_trades)}")

sell_trades = tl.filter_trades(side='s')
print(f"Sell trades only: {len(sell_trades)}")

# Filter by price range
trades_near_best = tl.filter_trades(
    min_price=100.0,
    max_price=101.5
)
print(f"Trades in price range [100.0, 101.5]: {len(trades_near_best)}")
print()

# Get LOB events only
lob_snapshots = tl.filter_events(event_type='lob')
print(f"LOB events: {len(lob_snapshots)}")
print()

# ==============================================================================
# 6. EXPORT METHODS
# ==============================================================================

print("=== Export Examples ===")

# Export to pandas DataFrame
df_all = tl.to_pd()
print("Full timeline as DataFrame:")
print(df_all.head())
print()

# Export trades only
df_trades = tl.trades_to_pd()
print("Trades as DataFrame:")
print(df_trades.head())
print()

# Export LOB time series
df_lob = tl.lob_to_pd()
print("LOB time series as DataFrame:")
print(df_lob.head())
print()

# Export combined data (LOB state + trades)
df_combined = tl.to_pd_combined()
print("Combined view (each row = event with LOB state):")
print(df_combined.head())
print()

# Export to files
# tl.to_csv("timeline.csv")
# tl.to_parquet("timeline.parquet")
# tl.trades_to_csv("trades.csv")
# tl.lob_to_csv("lob.csv")

# ==============================================================================
# 7. ADVANCED FEATURES
# ==============================================================================

print("=== Advanced Features ===")

# Reconstruct LOB at any timestamp (even between updates)
lob_at_1175 = tl.lob_at(timestamps[2] - 25)  # Between update and trade
print(f"LOB reconstructed at {timestamps[2] - 25}: spread = {lob_at_1175.spread}")
print()

# Get nearest trade to a timestamp
nearest_trade = tl.nearest_trade(timestamp=1180, direction='before')
print(f"Nearest trade before 1180: {nearest_trade.timestamp} @ {nearest_trade.price}")
print()

# Resample to fixed intervals (aggregate trades in time bins)
resampled = tl.resample_trades(interval=200)  # 200 time units per bin
print("Resampled trades (200 time unit bins):")
print(resampled.head())
print()

# Calculate VWAP (volume-weighted average price) over time window
vwap = tl.vwap(start_ts=timestamps[0], end_ts=timestamps[-1])
print(f"VWAP over entire period: {vwap:.2f}")
print()

# Get trade arrival rate (trades per time unit)
arrival_rate = tl.trade_arrival_rate()
print(f"Trade arrival rate: {arrival_rate:.6f} trades/time unit")
print()

# Detect sweep events (multiple trades at same timestamp across price levels)
sweeps = tl.detect_sweeps(min_trades=2)
print(f"Sweep events detected: {len(sweeps)}")
if len(sweeps) > 0:
    print(f"  Example sweep at {sweeps[0]['timestamp']}: {sweeps[0]['trade_count']} trades")
print()

# ==============================================================================
# 8. SUMMARY
# ==============================================================================

print("=== Summary ===")
print(f"Timeline: {tl.name}")
print(f"Time range: {tl.timestamps[0]} to {tl.timestamps[-1]}")
print(f"Duration: {tl.duration} time units")
print(f"Total events: {len(tl)}")
print(f"  - LOB updates: {len(tl.lob_events)}")
print(f"  - Trades: {len(tl.trades)}")
print(f"Total volume traded: {sum(t.volume for t in tl.trades):.2f}")
print(f"Number of unique price levels traded: {len(set(t.price for t in tl.trades))}")
print()

print("=== TL API Showcase Complete ===")
print("This example demonstrates the proposed API for TimeLine (TL),")
print("a unified interface for analyzing LOB and trade data together.")

# ==============================================================================
# APPENDIX: DETAILED LIQUIDITY PROFILE EXPLANATION
# ==============================================================================

print("\n" + "="*80)
print("APPENDIX: How liquidity_profile() Works")
print("="*80 + "\n")

print("The liquidity profile answers the question:")
print("'Where is liquidity posted vs. where is it actually consumed?'\n")

print("Example calculation for price level 101.00:")
print("-" * 60)
print("LOB snapshots containing 101.00:")
print("  t=1000: 101.00 x 2.1  (ask side)")
print("  t=1100: 101.00 x 1.5  (ask side, updated)")
print("  t=1200: 101.00 x 1.0  (ask side, after trade)")
print("  t=1300: (not present)")
print("  t=1400: 101.00 x 2.5  (ask side)")
print("  t=1500: 101.00 x 1.0  (ask side)")
print()
print("Trades at 101.00:")
print("  t=1150: Buy 0.5 @ 101.00  (took from ask)")
print("  t=1350: Buy 0.3 @ 101.00  (took from ask)")
print("  t=1450: Buy 1.5 @ 101.00  (took from ask)")
print()
print("Calculated metrics:")
print(f"  available_qty = 2.1 + 1.5 + 1.0 + 2.5 + 1.0 = 8.1")
print(f"  taken_qty = 0.5 + 0.3 + 1.5 = 2.3")
print(f"  lob_appearances = 5")
print(f"  trade_count = 3")
print(f"  avg_available = 8.1 / 5 = 1.62")
print(f"  ratio_taken = 2.3 / 8.1 = 0.284 (28.4% of posted liquidity was taken)")
print()

print("Use cases:")
print("  1. Identify 'hot zones' - price levels with high taken/available ratio")
print("  2. Find stale liquidity - high available, low taken (not attractive)")
print("  3. Market making - place orders at levels with good turnover")
print("  4. Execution analysis - understand where your trades fit in the profile")
print("  5. Spread analysis - compare bid vs ask liquidity profiles")
print()

print("Alternative analysis modes:")
print("  - Time-weighted: Weight each snapshot by duration (normalize_time=True)")
print("  - Volume-weighted: Weight by actual quantities")
print("  - Side-specific: Analyze bids vs asks separately")
print("  - Dynamic: Track how profile changes over time windows")
print()

print("Example insights from liquidity profile:")
print("  • If ratio_taken is high (>0.5): Price level is actively traded")
print("  • If ratio_taken is low (<0.1): Price level has stale/passive liquidity")
print("  • If available_qty is high but trade_count is low: Resistance/support level")
print("  • Asymmetry between bid/ask profiles: Directional pressure")
print()

print("="*80)

# ==============================================================================
# APPENDIX 2: INTENSITY FUNCTION ESTIMATION
# ==============================================================================

print("\n" + "="*80)
print("APPENDIX 2: Intensity Function Parameter Estimation")
print("="*80 + "\n")

print("1. GUÉANT INTENSITY FUNCTION (Exponential Model)")
print("-" * 80)
print()
print("Model: λ(δ) = A * exp(-k * δ)")
print("where:")
print("  - λ(δ) = trade arrival intensity at distance δ ticks from best price")
print("  - A = base intensity (trades per time unit at best price)")
print("  - k = decay rate (how quickly intensity decreases with distance)")
print("  - δ = distance in ticks from best bid/ask")
print()

print("Estimation method (Maximum Likelihood):")
print("  1. For each trade, compute δ = distance from best bid/ask at trade time")
print("     - Buy trades: δ = (trade_price - best_bid) / tick_size")
print("     - Sell trades: δ = (best_ask - trade_price) / tick_size")
print()
print("  2. Count trades at each tick distance:")
print("     - N(δ) = number of trades at distance δ")
print("     - T(δ) = total time exposed at distance δ")
print()
print("  3. Estimate intensity: λ̂(δ) = N(δ) / T(δ)")
print()
print("  4. Fit exponential model via regression:")
print("     - log(λ̂(δ)) = log(A) - k * δ")
print("     - Linear regression of log(λ̂) vs δ")
print("     - Slope = -k, Intercept = log(A)")
print()

print("Example calculation:")
print("  Observed data (ask side):")
print("    δ=0: 10 trades in 100 time units → λ̂(0) = 0.100")
print("    δ=1: 6 trades in 100 time units → λ̂(1) = 0.060")
print("    δ=2: 3 trades in 100 time units → λ̂(2) = 0.030")
print("    δ=3: 2 trades in 100 time units → λ̂(3) = 0.020")
print()
print("  Log transformation:")
print("    log(λ̂(0)) = log(0.100) = -2.303")
print("    log(λ̂(1)) = log(0.060) = -2.813")
print("    log(λ̂(2)) = log(0.030) = -3.507")
print("    log(λ̂(3)) = log(0.020) = -3.912")
print()
print("  Linear regression:")
print("    Slope (k̂) ≈ -0.51  →  k = 0.51")
print("    Intercept ≈ -2.30  →  A = exp(-2.30) = 0.100")
print()
print("  Fitted model: λ(δ) = 0.100 * exp(-0.51 * δ)")
print()

print("Interpretation of k:")
print(f"  - k = 0.51 means intensity decays by factor of e^(-0.51) ≈ 0.60 per tick")
print(f"  - Half-life: δ_half = ln(2)/k = 1.36 ticks")
print(f"  - At δ=2: intensity is ~0.36 of base intensity")
print()

print("-" * 80)
print("2. HO-STOLL INTENSITY FUNCTION (Linear Model)")
print("-" * 80)
print()
print("Model: λ(δ) = A - k * δ")
print("where k represents the linear decrease in intensity per tick")
print()

print("Estimation method (Ordinary Least Squares):")
print("  1. Compute empirical intensities λ̂(δ) = N(δ) / T(δ) (same as above)")
print()
print("  2. Fit linear model directly:")
print("     - λ̂(δ) = A - k * δ")
print("     - OLS regression of λ̂ vs δ")
print("     - Slope = -k, Intercept = A")
print()

print("Example calculation:")
print("  Same observed data:")
print("    (δ=0, λ̂=0.100), (δ=1, λ̂=0.060), (δ=2, λ̂=0.030), (δ=3, λ̂=0.020)")
print()
print("  Linear regression:")
print("    Slope: k̂ ≈ 0.028")
print("    Intercept: Â ≈ 0.095")
print()
print("  Fitted model: λ(δ) = 0.095 - 0.028 * δ")
print()

print("Interpretation:")
print("  - k = 0.028 means intensity decreases by 0.028 per tick")
print("  - At δ=1: λ(1) = 0.095 - 0.028 = 0.067")
print("  - Model becomes negative at δ > A/k ≈ 3.4 ticks (need to truncate)")
print()

print("-" * 80)
print("3. VPIN (Volume-Synchronized Probability of Informed Trading)")
print("-" * 80)
print()
print("Measures order flow toxicity based on buy/sell volume imbalance")
print()

print("Algorithm:")
print("  1. Divide trading into volume buckets of size V")
print("     - Each bucket contains exactly V volume (not time-based)")
print()
print("  2. Classify each bucket as net buy or net sell:")
print("     - Bulk Volume Classification (BVC):")
print("       * If average price in bucket > average price in previous: BUY")
print("       * If average price in bucket < average price in previous: SELL")
print("     - Trade Classification (TC):")
print("       * Directly use trade side (aggressor side)")
print()
print("  3. For each bucket i, compute:")
print("     - V_buy[i] = total buy volume in bucket")
print("     - V_sell[i] = total sell volume in bucket")
print()
print("  4. Calculate VPIN over n buckets:")
print("     - VPIN = Σ|V_buy[i] - V_sell[i]| / (n * V)")
print()

print("Example calculation:")
print("  Parameters: bucket_size = 10, num_buckets = 5")
print()
print("  Bucket 1: V_buy=7, V_sell=3  → |7-3| = 4")
print("  Bucket 2: V_buy=5, V_sell=5  → |5-5| = 0")
print("  Bucket 3: V_buy=8, V_sell=2  → |8-2| = 6")
print("  Bucket 4: V_buy=3, V_sell=7  → |3-7| = 4")
print("  Bucket 5: V_buy=6, V_sell=4  → |6-4| = 2")
print()
print("  VPIN = (4 + 0 + 6 + 4 + 2) / (5 * 10)")
print("       = 16 / 50")
print("       = 0.32")
print()

print("Interpretation:")
print("  - VPIN = 0.32 (moderate order flow imbalance)")
print("  - Higher VPIN → more toxic order flow → higher adverse selection risk")
print("  - VPIN > 0.7: high toxicity (informed trading likely)")
print("  - VPIN < 0.3: low toxicity (uninformed/balanced flow)")
print()

print("Applications:")
print("  1. Market Making:")
print("     - Widen spread when VPIN is high")
print("     - Reduce inventory when VPIN spikes")
print()
print("  2. Execution:")
print("     - Avoid large trades during high VPIN periods")
print("     - Use VPIN to time order execution")
print()
print("  3. Risk Management:")
print("     - High VPIN predicts higher short-term volatility")
print("     - VPIN spikes can signal flash crash conditions")
print()
print("  4. Liquidity Provision:")
print("     - Increase posted quotes when VPIN is low")
print("     - Pull liquidity when VPIN exceeds threshold")
print()

print("="*80)

"""
 1. Guéant Intensity Function (Exponential)

  Based on "Dealing with the Inventory Risk" by Guéant, Lehalle, Fernandez-Tapia (2013)

  Model: λ(δ) = A * exp(-k * δ)

  API Properties:
  - tl.gueant_A - Base intensity (aggregate both sides)
  - tl.gueant_k - Decay rate (aggregate both sides)
  - tl.gueant_A_ask, tl.gueant_k_ask - Ask side parameters
  - tl.gueant_A_bid, tl.gueant_k_bid - Bid side parameters

  Methods:
  - tl.get_intensity_function(side='a'|'b', model='gueant') - Returns callable λ(δ)

  Estimation: Maximum likelihood via log-linear regression on empirical intensities

  2. Ho-Stoll Intensity Function (Linear)

  Based on "Optimal dealer pricing" by Ho and Stoll (1981)

  Model: λ(δ) = A - k * δ

  API Properties:
  - tl.ho_k - Linear decay rate (aggregate)
  - tl.ho_k_ask, tl.ho_k_bid - Side-specific parameters

  Methods:
  - tl.get_intensity_function(side='a'|'b', model='ho-stoll') - Returns callable λ(δ)

  Estimation: Ordinary Least Squares regression

  3. VPIN (Volume-Synchronized Probability of Informed Trading)

  Based on Easley, López de Prado, O'Hara (2012)

  API Properties:
  - tl.VPIN - Current VPIN value ∈ [0, 1]

  Methods:
  - tl.calculate_vpin(bucket_size, num_buckets, method='bulk'|'trade') - Custom VPIN
  - tl.vpin_ts(bucket_size, num_buckets, rolling=True) - VPIN time series
  - tl.get_trade_classification(method='bulk') - Buy/sell classification for VPIN

  Calculation:
  VPIN = Σ|V_buy[i] - V_sell[i]| / (n * V)

  Interpretation:
  - VPIN < 0.3: Low toxicity (balanced flow)
  - VPIN 0.3-0.7: Moderate toxicity
  - VPIN > 0.7: High toxicity (informed trading likely)

  All three features include detailed appendices (see lines 533-672 in timeline.py) with:
  - Mathematical formulations
  - Step-by-step estimation procedures
  - Worked examples with numbers
  - Interpretation guidelines
  - Practical use cases for market making and risk management
"""