# lob.py
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

- tsLOB: time series of LOB
  - it has the same basic methods of LOB
  - at any group of updates (pushed via set_updates), it produce another LOB in the inner data structure
  - LOB(t) are indexed by timestamp
  - methods to convert numpy/pandas
  - stats: it can have further stats (time-based stats)
    - basic LOB stats in form of time series
    - arrival frequency
    - cancel frequency

We just need one dep: sortedcontainers. Consider to implement it.
