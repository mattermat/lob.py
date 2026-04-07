from lobpy import TL, itch_parser

filepath = "test_data/01302019.NASDAQ_ITCH50.gz"

# mode one: create a TL for a specific symbol
tl = TL()

parser = itch_parser("test_data/S101819-v50.txt.gz")

# TODO: manage the following internally via TL.load_itch("path")
"""
  ┌───────────┬─────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │ ITCH type │                   Meaning                   │                                      TL call(s)                                       │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ A         │ Add Order (anonymous)                       │ add_lob_update(ts, [(side, price, +shares)])                                          │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ F         │ Add Order with MPID                         │ add_lob_update(ts, [(side, price, +shares)])                                          │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ D         │ Delete Order (full cancel)                  │ add_lob_update(ts, [(side, price, -original_shares)])                                 │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ X         │ Order Cancel (partial)                      │ add_lob_update(ts, [(side, price, -cancelled_shares)])                                │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ U         │ Order Replace                               │ add_lob_update(ts, [(side, old_price, -orig_shares), (side, new_price, +new_shares)]) │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ E         │ Order Executed (at limit price)             │ add_lob_update(...) + add_trade(...)                                                  │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ C         │ Order Executed with Price (non-displayable) │ add_lob_update(...) + add_trade(...)                                                  │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ P         │ Non-Cross Trade (off-book print)            │ add_trade(ts, side, price, shares) only                                               │
  ├───────────┼─────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Q         │ Cross Trade (auction)                       │ add_trade(ts, '', price, shares) only                                                 │
  └───────────┴─────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘
"""

print("WARNING: IT CONSUMES TOO MUCH RAM!!!")

i = 0
for msg in parser.messages():
    if not (msg.symbol == "AAPL"):
        continue
    match msg.type:
        case "A" | "F":
            # case A: AddOrder(type='A', timestamp=16798097622248, order_ref=605001, side='B', shares=300, symbol='AAPL', price=234.0, mpid='')
            # case F: AddOrder(type='F', timestamp=25846948158830, order_ref=2592649, side='B', shares=100, symbol='AAPL', price=0.01, mpid='NITE')
            side = "b" if msg.side == "B" else "a"
            if len(tl.lob.timestamps) == 0:  # first update
                size = msg.shares
            else:
                last_ts = tl.lob.timestamps[-1]
                lob = tl.lob[last_ts]
                size = lob.at(side, msg.price) + msg.shares
            tl.add_lob_update(timestamp=msg.timestamp, updates=[(side, msg.price, size)])
        case "D":
            # DeleteOrder(type='D', timestamp=14431128988366, order_ref=96189, symbol='AAPL', side='S', price=236.89, shares=30)
            side = "b" if msg.side == "B" else "a"
            last_ts = tl.lob.timestamps[-1]
            lob = tl.lob[last_ts]
            size = lob.at(side, msg.price) - msg.shares
            tl.add_lob_update(timestamp=msg.timestamp, updates=[(side, msg.price, size)])
        case "E":
            # ExecuteOrder(type='E', timestamp=14765741715460, order_ref=109653, executed_shares=10, match_number=17905, symbol='AAPL', side='S', price=235.0)
            side = "b" if msg.side == "B" else "a"
            trade_side = "s" if msg.side == "B" else "b"
            last_ts = tl.lob.timestamps[-1]
            lob = tl.lob[last_ts]
            size = lob.at(side, msg.price) - msg.executed_shares
            tl.add_lob_update(msg.timestamp, updates=[(side, msg.price, size)])
            tl.add_trade(msg.timestamp, trade_side, msg.price, msg.executed_shares)
        case "P":
            # Trade(type='P', timestamp=25435670444121, shares=100, symbol='AAPL', price=235.3, match_number=20418, side='B')
            side = "b" if msg.side == "B" else "s"
            tl.add_trade(msg.timestamp, side, msg.price, msg.shares)
        case "Q":
            # Trade(type='Q', timestamp=34200489571720, shares=1664440, symbol='AAPL', price=234.53, match_number=111703, side='')
            tl.add_trade(msg.timestamp, "", msg.price, msg.shares)
        case "X":
            # CancelOrder(type='X', timestamp=33760585292120, order_ref=6432793, cancelled_shares=1, symbol='AAPL', side='S', price=234.9)
            side = "b" if msg.side == "B" else "a"
            last_ts = tl.lob.timestamps[-1]
            lob = tl.lob[last_ts]
            size = lob.at(side, msg.price) - msg.cancelled_shares
            tl.add_lob_update(msg.timestamp, updates=[(side, msg.price, size)])
        case "U":
            # ReplaceOrder(type='U', timestamp=34141647578478, order_ref=7944949, new_order_ref=7945153, orig_shares=100, shares=100, old_price=241.0, price=241.83, symbol='AAPL', side='S')
            side = "b" if msg.side == "B" else "a"
            last_ts = tl.lob.timestamps[-1]
            lob = tl.lob[last_ts]
            old_size = lob.at(side, msg.old_price) - msg.orig_shares
            new_size = lob.at(side, msg.price) + msg.shares
            tl.add_lob_update(
                msg.timestamp,
                updates=[(side, msg.old_price, old_size), (side, msg.price, new_size)],
            )
        case "C":
            # ExecuteOrder(type='C', timestamp=34201071214588, order_ref=9031469, executed_shares=30, match_number=174003, symbol='AAPL', side='S', price=234.5)
            side = "b" if msg.side == "B" else "a"
            trade_side = "s" if msg.side == "B" else "b"
            last_ts = tl.lob.timestamps[-1]
            lob = tl.lob[last_ts]
            size = lob.at(side, msg.price) - msg.executed_shares
            tl.add_lob_update(msg.timestamp, updates=[(side, msg.price, size)])
            tl.add_trade(msg.timestamp, trade_side, msg.price, msg.executed_shares)
        case _:
            print(msg)
            break
    if (i % 100) == 0:
        print(i)
    i += 1
