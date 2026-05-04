import warnings
import numpy as np
import pandas as pd
from lobpy import TL

warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)

tl = TL(name='BTC-USDT', tick_size=0.1)
tl.from_parquet('test_data/blofin_BTC-USDT_20260330_194305.parquet', mode="lazy")

total_duration = tl.timestamps[-1] - tl.timestamps[0]
window = total_duration // 10
custom_buckets = [1, 3, 5, 10]
holding_time_ns = int(1e9)  # 1 second

# ── per-timestamp LOBts stats ─────────────────────────────────────────────────
lob_ts = pd.DataFrame({
    'spread':      tl.lob.spread,
    'bid':         tl.lob.bid,
    'ask':         tl.lob.ask,
    'midprice':    tl.lob.midprice,
    'vw_midprice': tl.lob.vw_midprice,
    'bidq':        tl.lob.bidq,
    'askq':        tl.lob.askq,
    'vi':          tl.lob.vi,
    'ofi':         tl.lob.order_flow_imbalance,
})

# ── rolling analytics ─────────────────────────────────────────────────────────
hawkes_roll = tl.hawkes(window_size=window).rename(columns=lambda c: f'hawkes_{c}')
A_ask_roll, k_ask_roll = tl.gueant.ask(window_size=window, buckets=custom_buckets)
A_bid_roll, k_bid_roll = tl.gueant.bid(window_size=window, buckets=custom_buckets)

# ── main DataFrame: outer-join on timestamp, ffill LOBts BBO ─────────────────
df = pd.concat([
    lob_ts,
    tl.realized_vol(window_size=window).rename('realized_vol'),
    tl.vpin(window_size=window),
    tl.kyle_lambda(window_size=window).rename('kyle_lambda'),
    hawkes_roll,
    A_ask_roll.rename('gueant_A_ask'),
    k_ask_roll.rename('gueant_k_ask'),
    A_bid_roll.rename('gueant_A_bid'),
    k_bid_roll.rename('gueant_k_bid'),
], axis=1)
df.index.name = 'timestamp'
df[list(lob_ts.columns)] = df[list(lob_ts.columns)].ffill()

# ── OHLC ──────────────────────────────────────────────────────────────────────
ohlc = tl.ohlc('1s')

# ── volume buckets & fill rate ────────────────────────────────────────────────
vol_buckets = tl.volume_buckets()
fill_rate_ask = tl.fill_rate(holding_time=holding_time_ns, side='a')
fill_rate_bid = tl.fill_rate(holding_time=holding_time_ns, side='b')

# ── Guéant bucket tables ──────────────────────────────────────────────────────
gueant_buckets_ask = tl.gueant.buckets('a', buckets=custom_buckets)
gueant_buckets_bid = tl.gueant.buckets('b', buckets=custom_buckets)

# ── global scalars ────────────────────────────────────────────────────────────
A_ask_s, k_ask_s = tl.gueant.ask(buckets=custom_buckets)
A_bid_s, k_bid_s = tl.gueant.bid(buckets=custom_buckets)
scalars = pd.Series({
    'trade_frequency':          tl.trade_frequency,
    'ask_trade_frequency':      tl.ask_trade_frequency,
    'bid_trade_frequency':      tl.bid_trade_frequency,
    'trade_volume_imbalance':   tl.trade_volume_imbalance,
    'order_arrival_volume':     tl.lob.order_arrival_volume,
    'order_cancel_volume':      tl.lob.order_cancel_volume,
    'order_arrival_frequency':  tl.lob.order_arrival_frequency,
    'order_cancel_frequency':   tl.lob.order_cancel_frequency,
    'bid_arrival_frequency':    tl.lob.bid_order_arrival_frequency,
    'ask_arrival_frequency':    tl.lob.ask_order_arrival_frequency,
    'bid_cancel_frequency':     tl.lob.bid_order_cancel_frequency,
    'ask_cancel_frequency':     tl.lob.ask_order_cancel_frequency,
    'realized_vol':             tl.realized_vol(),
    'vpin':                     tl.vpin(),
    'kyle_lambda':              tl.kyle_lambda(),
    **{f'hawkes_{k}': v for k, v in tl.hawkes().items()},
    'gueant_A_ask':             A_ask_s,
    'gueant_k_ask':             k_ask_s,
    'gueant_A_bid':             A_bid_s,
    'gueant_k_bid':             k_bid_s,
}, name='value')

# ── summary DataFrame ────────────────────────────────────────────────────────
window_5s  = 5_000_000_000   # ns
window_10s = 10_000_000_000  # ns
window_60s = 60_000_000_000  # ns

# df + latest ohlc bar (backward merge)
summary = pd.merge_asof(
    df.sort_index().reset_index(),
    ohlc.add_prefix('ohlc_').reset_index(),
    on='timestamp',
    direction='backward',
).set_index('timestamp')

# realized_vol at finer windows
summary = summary.join(tl.realized_vol(window_size=window_10s).rename('realized_vol_10s'), how='left')
summary = summary.join(tl.realized_vol(window_size=window_60s).rename('realized_vol_60s'), how='left')

# rolling 5s: order arrival/cancel rates, trade frequencies, fill_rate (10s holding)
records, ts_out = [], []
for tl_w in tl.rolling(window_5s):
    ts_out.append(tl_w.timestamps[-1])
    fr_a = tl_w.fill_rate(holding_time=window_10s, side='a', buckets=custom_buckets)
    fr_b = tl_w.fill_rate(holding_time=window_10s, side='b', buckets=custom_buckets)
    fr_a_map = fr_a.set_index('delta')['fill_rate'].to_dict()
    fr_b_map = fr_b.set_index('delta')['fill_rate'].to_dict()
    records.append({
        'order_arrival_rate_5s':     tl_w.lob.order_arrival_frequency,
        'order_cancel_rate_5s':      tl_w.lob.order_cancel_frequency,
        'bid_order_arrival_rate_5s': tl_w.lob.bid_order_arrival_frequency,
        'ask_order_arrival_rate_5s': tl_w.lob.ask_order_arrival_frequency,
        'bid_order_cancel_rate_5s':  tl_w.lob.bid_order_cancel_frequency,
        'ask_order_cancel_rate_5s':  tl_w.lob.ask_order_cancel_frequency,
        'trade_freq_5s':             tl_w.trade_frequency,
        'ask_trade_freq_5s':         tl_w.ask_trade_frequency,
        'bid_trade_freq_5s':         tl_w.bid_trade_frequency,
        **{f'fill_rate_ask_d{d}': fr_a_map.get(float(d), np.nan) for d in custom_buckets},
        **{f'fill_rate_bid_d{d}': fr_b_map.get(float(d), np.nan) for d in custom_buckets},
    })

rolling_5s = pd.DataFrame(records, index=pd.Index(ts_out, name='timestamp'))
summary = summary.join(rolling_5s, how='left')

# rolling Guéant at 5s window
A_ask_5s, k_ask_5s = tl.gueant.ask(window_size=window_5s, buckets=custom_buckets)
A_bid_5s, k_bid_5s = tl.gueant.bid(window_size=window_5s, buckets=custom_buckets)
summary = summary.join(pd.DataFrame({
    'gueant_A_ask_5s': A_ask_5s,
    'gueant_k_ask_5s': k_ask_5s,
    'gueant_A_bid_5s': A_bid_5s,
    'gueant_k_bid_5s': k_bid_5s,
}), how='left')

# ── output ────────────────────────────────────────────────────────────────────
print("=== main df ===")
print(df)
print(f"\nShape: {df.shape}")

print("\n=== ohlc (1s) ===")
print(ohlc)

print("\n=== volume buckets ===")
print(vol_buckets)

print("\n=== fill rate — ask ===")
print(fill_rate_ask)

print("\n=== fill rate — bid ===")
print(fill_rate_bid)

print(f"\n=== guéant buckets — ask {custom_buckets} ===")
print(gueant_buckets_ask)

print(f"\n=== guéant buckets — bid {custom_buckets} ===")
print(gueant_buckets_bid)

# print("\n=== scalars ===")
# print(scalars)

print("\n=== summary ===")
print(summary.head())
print(f"\nShape: {summary.shape}")
