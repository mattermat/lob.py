"""
TimeLine (TL) with real data.
"""

import matplotlib.pyplot as plt

from lobpy import TL

# Create a TimeLine with configuration
tl = TL(name="BTC-USD", tick_size=0.1, lob_mode="realtime")

tl.from_parquet("test_data/blofin_example_data.parquet")

WINDOW = 30_000_000_000  # 30s rolling window

ohlc = tl.ohlc("5s")

print("ohcl 5 seconds chart")
# plotting: ohlc
fig, ax = plt.subplots(figsize=(12, 4))
for ts, row in ohlc.iterrows():
    color = "green" if row["close"] >= row["open"] else "red"
    ax.plot([ts, ts], [row["low"], row["high"]], color=color, linewidth=1)
    ax.plot([ts, ts], [row["open"], row["close"]], color=color, linewidth=4)
ax.set_xlabel("timestamp (ns)")
ax.set_ylabel("price")
ax.set_title("BTC-USD OHLC (5s)")
plt.tight_layout()
plt.show()

bid = tl.lob.bid_ts()
ask = tl.lob.ask_ts()
vpin = tl.vpin(window_size=WINDOW)
rvol = tl.realized_vol(window_size=WINDOW)
custom_buckets = [1, 3, 5, 10]
Aa, ka = tl.gueant.ask(WINDOW, buckets=custom_buckets)
Ab, kb = tl.gueant.bid(WINDOW, buckets=custom_buckets)


def _plot_bid_ask(ax):
    ax.plot(bid.index, bid.values, label="best bid", linewidth=0.8)
    ax.plot(ask.index, ask.values, label="best ask", linewidth=0.8)
    ax.set_xlabel("timestamp (ns)")
    ax.set_ylabel("price")


print("bid ask vs VPIN")
# chart 1: bid, ask + VPIN
fig, ax1 = plt.subplots(figsize=(12, 4))
_plot_bid_ask(ax1)
ax2 = ax1.twinx()
ax2.plot(vpin.index, vpin.values, label="VPIN (30s)", color="orange", linewidth=1, alpha=0.8)
ax2.set_ylabel("VPIN")
ax2.set_ylim(0, 1)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_title("BTC-USD bid & ask + rolling VPIN (30s)")
plt.tight_layout()
plt.show()

print("bid ask vs realized volatility")
# chart 2: bid, ask + realized vol
fig, ax1 = plt.subplots(figsize=(12, 4))
_plot_bid_ask(ax1)
ax2 = ax1.twinx()
ax2.plot(
    rvol.index, rvol.values, label="realized vol (30s)", color="purple", linewidth=1, alpha=0.8
)
ax2.set_ylabel("realized vol")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_title("BTC-USD bid & ask + rolling realized vol (30s)")
plt.tight_layout()
plt.show()

print("bid ask vs Gueant A and k")
# chart 3: bid, ask + Gueant A and k (ask and bid sides)
fig, ax1 = plt.subplots(figsize=(12, 4))
_plot_bid_ask(ax1)
ax2 = ax1.twinx()
ax2.plot(ka.index, ka.values, label="k ask (30s)", color="red", linewidth=1, alpha=0.8)
ax2.plot(kb.index, kb.values, label="k bid (30s)", color="green", linewidth=1, alpha=0.8)
ax2.plot(
    Aa.index, Aa.values, label="A ask (30s)", color="red", linewidth=1, alpha=0.8, linestyle="--"
)
ax2.plot(
    Ab.index, Ab.values, label="A bid (30s)", color="green", linewidth=1, alpha=0.8, linestyle="--"
)
ax2.set_ylabel("Guéant A / k")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_title("BTC-USD bid & ask + rolling Guéant parameters (30s)")
plt.tight_layout()
plt.show()
