# Goal
**Raw / HFT features**:
- [x] best bid
- [x] best ask
- [x] spread / spread in tick
- [x] LOB Volume Imbalance top level
- [ ] LOB Volume Imbalance top 10 ticks

**Rolling window features**:
- [ ] Realized Volatility (5s)
- [x] Realized Volatility (60s)
- [x] OHLC (1s)
- [x] Rates (5s)
  - [x] bid order arrival
  - [x] ask order arrival
  - [x] order arrival
  - [x] bid order cancel
  - [x] ask order cancel
  - [x] order cancel
  - [x] buy trade arrival
  - [x] sell trade arrival
  - [x] trade arrival
- [x] VPIN (60s)
- [x] Kyle's Lambda (60s)
- [x] Hawkes (5s)
- [x] Hawkes (60s)
- [x] Order Flow Imbalance (5s)
- [x] Gueant's Intensity function (5s)
- [x] Gueant's Intensity function (60s)
- [ ] Price momentum (5s) [momentum = log(mid_t / mid_{t-5s})]
- [ ] Price momentum (60s) [momentum = log(mid_t / mid_{t-5s})]
- [ ] Regime context[latest trade price] (60s) [HMM]
- [ ] Regime context[spread] (60s) [HMM or spread / rolling median spread]

~From the backtester~
- [ ] inventory (raw)
- [ ] open quotes (raw)
- [ ] estimated queue position (raw)

### Current available features
```
╔══════════════════════════════════════════════╤═══════════╤═══════════╗
║ Column                                       │ Type      │ Window    ║
╠══════════════════════════════════════════════╪═══════════╪═══════════╣
║ timestamp                                    │ immediate │ —         ║
║ spread                                       │ immediate │ —         ║
║ bid                                          │ immediate │ —         ║
║ ask                                          │ immediate │ —         ║
║ midprice                                     │ immediate │ —         ║
║ bidq                                         │ immediate │ —         ║
║ askq                                         │ immediate │ —         ║
║ vw_midprice                                  │ immediate │ —         ║
║ vi                                           │ immediate │ —         ║
║ realized_vol                                 │ rolling   │ 60s       ║
║ gueant_A_ask                                 │ rolling   │ 60s       ║
║ gueant_k_ask                                 │ rolling   │ 60s       ║
║ gueant_A_bid                                 │ rolling   │ 60s       ║
║ gueant_k_bid                                 │ rolling   │ 60s       ║
║ gueant_A_ask_5s                              │ rolling   │ 5s        ║
║ gueant_k_ask_5s                              │ rolling   │ 5s        ║
║ gueant_A_bid_5s                              │ rolling   │ 5s        ║
║ gueant_k_bid_5s                              │ rolling   │ 5s        ║
║ ofi                                          │ immediate │ —         ║
║ order_arrival_rate_5s                        │ rolling   │ 5s        ║
║ order_cancel_rate_5s                         │ rolling   │ 5s        ║
║ bid_order_arrival_rate_5s                    │ rolling   │ 5s        ║
║ ask_order_arrival_rate_5s                    │ rolling   │ 5s        ║
║ bid_order_cancel_rate_5s                     │ rolling   │ 5s        ║
║ ask_order_cancel_rate_5s                     │ rolling   │ 5s        ║
║ trade_freq_5s                                │ rolling   │ 5s        ║
║ ask_trade_freq_5s                            │ rolling   │ 5s        ║
║ bid_trade_freq_5s                            │ rolling   │ 5s        ║
║ hawkes_mu                                    │ rolling   │ 60s       ║
║ hawkes_alpha                                 │ rolling   │ 60s       ║
║ hawkes_beta                                  │ rolling   │ 60s       ║
║ hawkes_branching_ratio                       │ rolling   │ 60s       ║
║ hawkes_mu_5s                                 │ rolling   │ 5s        ║
║ hawkes_alpha_5s                              │ rolling   │ 5s        ║
║ hawkes_beta_5s                               │ rolling   │ 5s        ║
║ hawkes_branching_ratio_5s                    │ rolling   │ 5s        ║
║ vpin                                         │ rolling   │ 60s       ║
║ kyle_lambda                                  │ rolling   │ 60s       ║
║ ohlc_open                                    │ rolling   │ 1s        ║
║ ohlc_high                                    │ rolling   │ 1s        ║
║ ohlc_low                                     │ rolling   │ 1s        ║
║ ohlc_close                                   │ rolling   │ 1s        ║
║ ohlc_volume                                  │ rolling   │ 1s        ║
╚══════════════════════════════════════════════╧═══════════╧═══════════╝
```