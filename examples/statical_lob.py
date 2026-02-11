from lobpy import LOB

lob = LOB(tick_size=0.1)

lob.set_snapshot(
    [(99.8, 10), (99.7, 20), (99.2, 30)],
    [(100.1, 15), (101.0, 25)]
)
print(lob.to_np())
print(lob.check())

# book data
print(lob.bid)
print(lob.bidq)
print(lob.bid[0])
print(lob.bid[1]) # bid at level 1
print(lob.ask)
assert lob.ask == lob.ask[0]
assert lob.bid == lob.bid[0]
print(lob.bid[1]) # bid at level 1
print(lob.ask[1]) # ask at level 1
print(lob.askq)
print(lob.vi)
assert lob.vi == lob.vi[0]
print(lob.vi[1]) # volume imbalance of the top 1 levels (cumulated)
print(lob.spread)
print(lob.spread_tick)
print(lob.spread_rel)
print(lob.midprice)
print(lob.vw_midprice)

#lob.to_xlsx('lob.xlsx')

print()
# impossible update
lob.set_updates([
    ('bid', 99.7, 0),
    ('bid', 101.5, 10),
    ('ask', 100.1, 10),
    ('ask', 101.0, 0),
    ('ask', 102.0, 10)
])
print(lob.to_np())
print(lob.check())

# TODO: change the arguments of `get_delta`
print(
    lob.get_delta(bids=[
        ('bid', 99.7, 0),
        ('bid', 101.5, 10)],
        asks=[('ask', 100.1, 10),
        ('ask', 101.0, 0),
        ('ask', 102.0, 10)])
)