from lobpy import LOB
import numpy as np

lob = LOB(tick_size=0.1)
lob.set_snapshot(bids=[(99.8, 10), (99.7, 20), (99.2, 30)], asks=[(100.1, 15), (101.0, 25)])

print(lob.to_np())
assert lob.check()

# bid and ask levels
assert lob.bid == 99.8
assert lob.bidq == 10
assert lob.bid[0] == lob.bid == 99.8
assert lob.bid[1] == 99.7 # bid at level 1
assert lob.ask == lob.ask[0] == 100.1
assert lob.askq == 15
assert lob.ask[1] == 101.0 # ask at level 1

# volume imbalance
assert lob.vi == lob.vi[0] == (10 - 15) / (10 + 15) == -0.2
# volume imbalance of the top 1 levels (cumulated)
print(f"volume imbalance: {lob.vi[1]:.4f}")

bid_v1 = lob.bidq[0] + lob.bidq[1]  # cumulated bid quantity at level 0 and 1
ask_v1 = lob.askq[0] + lob.askq[1]  # cumulated ask quantity at level 0 and 1
assert lob.vi[1] == (bid_v1 - ask_v1) / (bid_v1 + ask_v1)

# spread and midprice
np.testing.assert_almost_equal(lob.spread, abs(lob.ask - lob.bid))
np.testing.assert_almost_equal(lob.spread, 0.3)

np.testing.assert_almost_equal(
    lob.spread_tick,
    int(round(lob.spread / lob.tick_size,0))
)
np.testing.assert_almost_equal(lob.spread_tick, 3)

np.testing.assert_almost_equal(lob.midprice, (lob.bid + lob.ask) / 2)
np.testing.assert_almost_equal(lob.midprice, 99.95)

np.testing.assert_almost_equal(lob.spread_rel, lob.spread / lob.midprice)
np.testing.assert_almost_equal(lob.spread_rel, 0.3 / 99.95)

np.testing.assert_almost_equal(lob.vw_midprice, (lob.bid * lob.bidq + lob.ask * lob.askq) / (lob.bidq + lob.askq))
np.testing.assert_almost_equal(lob.vw_midprice, 99.98)

# aggregated quantities

# aggregate by level
lob.aggq("a", nlevel=3)

# aggregate by ticks from the best level
lob.aggq("b", ticks=5)

# aggregate by price
lob.aggq("b", price=99.7)
np.testing.assert_almost_equal(
    lob.aggq("b", price=99.7), # should include first two bid levels
    lob.bidq[0] + lob.bidq[1]
)

# lob.to_xlsx() requires openpyxl
# lob.to_xlsx('lob.xlsx')

print()
# impossible update
lob.set_updates(
    [("b", 99.7, 0), ("b", 101.5, 10), ("a", 100.1, 10), ("a", 101.0, 0), ("a", 102.0, 10)]
)
print(lob.to_np())
assert lob.check() == False

print()
# comparison between two LOBs
base_book = LOB(
    tick_size=0.1, bids=[(99.8, 10), (99.7, 20), (99.2, 30)], asks=[(100.1, 15), (101.0, 25)]
)
comp_book = LOB(
    tick_size=0.1,
    bids=[(99.7, 20), (99.2, 30)],
    asks=[(99.9, 15), (101.0, 25)]
)

# comparison
print(base_book.diff(comp_book))  # noqa: E501

# TODO: fix the following: should return one char for the side ('a' instead of 'ask') and the quantity should be np.float64 as well
# returns
# [('bid', np.float64(99.8), 0), ('ask', np.float64(99.9), np.float64(15.0)), ('ask', np.float64(100.1), 0)]