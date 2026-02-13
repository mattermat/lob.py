"""
Example showing LOBts (Time Series LOB) usage.
"""

from lobpy import LOBts
import time

# array of timestamps in microseconds
times = [
    1770990337346989262321,
    1770990337346989262322,
    1770990337346989262323
]

# Create a time-series LOB
lobts = LOBts(name="BTC-USD", tick_size=0.01, mode='history')

# Set initial snapshot at timestamp 1000
lobts.set_snapshot(
    bids=[(49900.00, 1.5), (49899.00, 2.3), (49898.50, 1.8)],
    asks=[(49901.00, 2.1), (49902.00, 1.7), (49903.00, 2.5)],
    timestamp=1000
)

# Add updates at timestamp 1001
lobts.set_updates([
    ('bid', 49900.00, 2.0),  # Update size
    ('bid', 49901.00, 1.0),  # New level (arrival)
    ('ask', 49901.00, 0),    # Cancel level
    ('ask', 49904.00, 1.5),  # New level
], timestamp=1001)

# Another update at timestamp 1002
lobts.set_updates([
    ('bid', 49899.00, 0),    # Cancel
    ('bid', 49897.00, 3.0),  # New
], timestamp=1002)

# Access LOB at specific timestamp
print("=== LOB at timestamp 1000 ===")
lob_1000 = lobts[1000]
print(f"Best bid: {lob_1000.bid}")
print(f"Best ask: {lob_1000.ask}")
print(f"Spread: {lob_1000.spread}")

print("\n=== LOB at timestamp 1001 ===")
lob_1001 = lobts[1001]
print(f"Best bid: {lob_1001.bid}")
print(f"Best ask: {lob_1001.ask}")
print(f"Spread: {lob_1001.spread}")

print("\n=== Latest LOB ===")
print(f"Best bid: {lobts.bid}")
print(f"Best ask: {lobts.ask}")
print(f"Spread: {lobts.spread}")

# Get time-series stats
print("\n=== Time Series Statistics ===")
spread_ts = lobts.spread_ts()
print(f"Spread time series: {spread_ts}")

midprice_ts = lobts.midprice_ts()
print(f"Mid-price time series: {midprice_ts}")

# Arrival and cancel frequency (total)
print("\n=== Event Frequencies ===")
arrival_freq = lobts.arrival_frequency()
print(f"Total arrivals: {arrival_freq}")

cancel_freq = lobts.cancel_frequency()
print(f"Total cancels: {cancel_freq}")

# Frequency per time window
print("\n=== Window-based Frequencies ===")
arrival_window = lobts.arrival_frequency(window=1)  # Per timestamp unit
print(f"Arrivals per window: {arrival_window}")

# Export to pandas
print("\n=== Export to Pandas ===")
df = lobts.to_pd()
print(df)

# Get a range of timestamps
print("\n=== Time Range ===")
lob_range = lobts.get_range(start_ts=1000, end_ts=1001)
print(f"Timestamps in range: {list(lob_range._timestamps)}")

# Update mode='latest' - keeps only latest state
lobts_latest = LOBts(name="Latest Only", mode='latest')
lobts_latest.set_snapshot(
    bids=[(100, 10)],
    asks=[(101, 8)],
    timestamp=2000
)
lobts_latest.set_updates([('bid', 99, 5)], timestamp=2001)
lobts_latest.set_updates([('bid', 102, 3)], timestamp=2002)

print("\n=== Latest Mode ===")
print(f"Number of stored snapshots: {len(lobts_latest)}")
print(f"Best bid at latest: {lobts_latest.bid}")
