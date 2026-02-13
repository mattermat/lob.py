"""
Example showing LOBts (Time Series LOB) usage.
"""


from lobpy import LOBts

# array of timestamps in microseconds
times = [
    1770990337346989262321,
    1770990337346989262322,
    1770990337346989262323
]

# Create a time-series LOB
lobts = LOBts(tick_size=0.01) # mode not specified, default is 'delta'

lobts.set_snapshot(
    bids=[(49900.00, 1.5), (49899.00, 2.3), (49898.50, 1.8)],
    asks=[(49901.00, 2.1), (49902.00, 1.7), (49903.00, 2.5)],
    timestamp=times[0]
)

# Set updates. It create another point in the timeseries
lobts.set_updates([
    ('b', 49900.00, 2.0),
    ('b', 49901.00, 1.0),
    ('a', 49901.00, 0),
    ('a', 49904.00, 1.5),
], timestamp=times[1])
# providing an existing timestamp should throw an error, unless `force=True` is set

# Set updates. It create another point in the timeseries
lobts.set_updates([
    ('b', 49899.00, 0),
    ('b', 49897.00, 3.0),
], timestamp=times[2])

# Access LOB at specific timestamp
print(f"LOB at timestamp {times[0]}:")
print(lobts[times[0]].to_np())
print(f"Bid: {lobts[times[0]].bid}")
print(f"Ask: {lobts[times[0]].ask}")
print(f"Spread: {lobts[times[0]].spread}")

print(f"LOB at timestamp {times[1]}:")
print(lobts[times[1]].to_np())
print(f"Bid: {lobts[times[1]].bid}")
print(f"Ask: {lobts[times[1]].ask}")
print(f"Spread: {lobts[times[1]].spread}")

# LOB lengths
print({lobts.len}) # return the lenght in number of timestamps
print({lobts.len_ts}) # last timestamp - first timestamp

# Get time-series stats
print("\n=== Time Series Statistics ===")
print(f"Spread time series: {lobts.spread}")
print(f"Bid time series: {lobts.ask}")
print(f"Ask time series: {lobts.bid}")
print(f"Mid-price time series: {lobts.midprice}")

# Arrival and cancel frequency (total)
print("\n=== Event Frequencies ===")
print(f"Total arrivals: {lobts.arrival_frequency}")
print(f"Total cancels: {lobts.cancel_frequency}")

# Convert to pandas DataFrame
print(lobts.to_pd())

# Time windows slicing
sliced_lobts = lobts[times[1]:times[2]]
print(lobts.to_pd())

# Get time-series stats
print("\n=== Time Series Statistics ===")
print(f"Spread time series: {sliced_lobts.spread}")
print(f"Bid time series: {sliced_lobts.ask}")
print(f"Ask time series: {sliced_lobts.bid}")
print(f"Mid-price time series: {sliced_lobts.spread}")

# Arrival and cancel frequency (total)
print("\n=== Event Frequencies ===")
print(f"Total arrivals: {sliced_lobts.arrival_frequency}")
print(f"Total cancels: {sliced_lobts.cancel_frequency}")
