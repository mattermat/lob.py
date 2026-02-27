"""
Example showing LOBts (Time Series LOB) usage.
"""

from lobpy import LOBts

# array of timestamps in microseconds
times = [1770990337346989262321, 1770990337346989262322, 1770990337346989262323]

# Create a time-series LOB
lobts = LOBts(tick_size=0.01)  # mode not specified, default is 'delta'

lobts.set_snapshot(
    bids=[(49900.00, 1.5), (49899.00, 2.3), (49898.50, 1.8)],
    asks=[(49901.00, 2.1), (49902.00, 1.7), (49903.00, 2.5)],
    timestamp=times[0],
)
lobts.set_updates(
    [
        ("b", 49900.00, 2.0),
        ("b", 49901.00, 1.0),
        ("a", 49901.00, 0),
        ("a", 49904.00, 1.5),
    ],
    timestamp=times[1],
)
lobts.set_updates(
    [
        ("b", 49899.00, 0),
        ("b", 49897.00, 3.0),
    ],
    timestamp=times[2],
)

for i in lobts.timestamps:
    print(i)
    print(lobts[i].to_np())
    print(f"vi: {lobts[i].vi}")
    print(lobts[i].bidq - 1)
    print((lobts[i].bidq - lobts[i].askq) / (lobts[i].bidq + lobts[i].askq))
    print()
