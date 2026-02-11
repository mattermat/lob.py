# lob.py

![Tests](https://github.com/mattermat/lob.py/actions/workflows/tests.yml/badge.svg)
![Coverage](https://codecov.io/gh/mattermat/lob.py/branch/main/graph/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/lobpy)

Limit Order Book in python

- LOB is the basic limit order book.
- LOB has the following methods:
  - basics:
    - set snapshot: push multiple levels (should be equal to set updates)
    - set updates: push multiple updates
    - update: update a single level
  - utils:
    - diff: difference between two lob (it returns the updates needed to change the lob 1 to the lob 2)
    - track_queue_position: to define how and why
    - len in tick: you provide side and price, it return the number of tick the provided price is far from the top of the book
    - methods to convert numpy/pandas
  - stats:
    - spread
    - limit order book

- LOBts: time series of LOB
  - it has the same basic methods of LOB
  - at any group of updates (pushed via set_updates), it produce another LOB in the inner data structure
  - LOB(t) are indexed by timestamp
  - methods to convert numpy/pandas
  - stats: it can have further stats (time-based stats)
    - basic LOB stats in form of time series
    - arrival frequency
    - cancel frequency

We just need one dep: sortedcontainers. Consider to implement it.


### LOB API
#### Methods
- `set_snapshot`
- `update`
- `set_update`
#### Properties
- `bid`: best bid price
- `ask`: best ask price
- `vi`: volume imbalance
- `bidq`: best bid size
- `askq`: best ask size
- `bid[0]`: bid at level 0 - equals to `bid`
- `bid[i]`: bid at level i
- `ask[0]`: ask at level 0 - equals to `ask`
- `ask[i]`: ask at level i
- `vi[0]`: volume imbalance of the first level - equals to `vi`
- `vi[i]`: volume imbalance of the top i levels
- `spread`: spread in absolute value
- `spread_tick`: spread in ticks
- `spread_rel`: spread in percentage of the bid level
- `midprice`: mid-price
- `vw_midprice`: volume-weighted mid-price
#### Other methods
- `get_slippage(volume, side=['midprice', 'ask', 'bid'])`: calculate the slippage from the top level (from the midprice is not declared)
