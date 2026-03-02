"""
Example showing LOBts (Time Series LOB) usage.
"""

from lobpy import LOB

# Create a time-series LOB

l = LOB(
    #tick_size=0.01,
    bids=[(49900.00, 1.5), (49899.00, 2.3), (49898.50, 1.8)],
    asks=[(49901.00, 2.1), (49902.00, 1.7), (49903.00, 2.5)]
)

print(l.bidq)
print(l.bidq + 1)
print(l.bidq[1])
print((l.bidq - l.askq) / (l.bidq + l.askq))
print(l.vi + l.vi)
