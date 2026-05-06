#ifndef LOBCORE_H
#define LOBCORE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ================================================================== */
/* Side — one side of the order book (bids or asks)                   */
/* ================================================================== */

#define SIDE_INITIAL_CAP 16

typedef struct {
    double *data;   /* interleaved [price, qty, price, qty, ...] */
    int     len;    /* number of price levels */
    int     cap;    /* allocated capacity (in levels) */
    int     is_bid; /* 1 = bids (descending), 0 = asks (ascending) */
} Side;

void  side_init(Side *s, int is_bid);
void  side_free(Side *s);
int   side_find(const Side *s, double price);
int   side_insert_pos(const Side *s, double price);
void  side_insert_at(Side *s, int pos, double price, double qty);
void  side_delete_at(Side *s, int pos);
void  side_update(Side *s, double price, double qty);
void  side_set(Side *s, const double *prices, const double *qtys, int n);
void  side_clear(Side *s);
double side_at(const Side *s, double price);
void  side_sort_desc(double *data, int n);
void  side_sort_asc(double *data, int n);
double side_aggq_nlevel(const Side *s, int nlevel);
double side_aggq_price(const Side *s, double price);
void  side_copy(Side *dst, const Side *src);
void  side_load_flat(Side *s, const double *flat, int n_levels);

/* ================================================================== */
/* LobBook — the double-sided order book (bids + asks)                */
/* ================================================================== */

typedef struct {
    Side       bids;
    Side       asks;
    double     tick_size;
    long long  timestamp;
    char       name[64];
} LobBook;

LobBook    *lob_create(const char *name, double tick_size, long long timestamp);
void        lob_destroy(LobBook *lob);
LobBook    *lob_copy(const LobBook *src);
void        lob_set_snapshot(LobBook *lob, long long ts,
                             const double *bid_p, const double *bid_q, int n_bids,
                             const double *ask_p, const double *ask_q, int n_asks);
void        lob_set_updates(LobBook *lob, long long ts,
                            const uint8_t *sides, const double *prices,
                            const double *qtys, int n);
double      lob_spread(const LobBook *lob);
double      lob_midprice(const LobBook *lob);
double      lob_vw_midprice(const LobBook *lob);
double      lob_vi(const LobBook *lob, int nlevels);
int         lob_check(const LobBook *lob);

/* ================================================================== */
/* LOBts — time-series of LobBooks                                    */
/* ================================================================== */

#define LOBTS_DELTA   0
#define LOBTS_LATEST  1
#define LOBTS_INITIAL_CAP 16

typedef struct {
    char       name[64];
    double     tick_size;
    int        mode;
    long long *ts_arr;
    LobBook  **lob_arr;
    int        len;
    int        cap;
} LOBts;

void lobts_grow(LOBts *lt);

/* ================================================================== */
/* Trades — dynamic array of trade events                             */
/* ================================================================== */

#define TRADES_INITIAL_CAP 64

typedef struct {
    long long timestamp;
    uint8_t   side;    /* 0 = buy aggressor ('b'), 1 = sell aggressor ('s') */
    double    price;
    double    volume;
} TradeEntry;

typedef struct {
    TradeEntry *data;
    int         len;
    int         cap;
} Trades;

Trades *trades_create(void);
void    trades_destroy(Trades *tr);
void    trades_clear(Trades *tr);
int     trades_len(const Trades *tr);
void    trades_append(Trades *tr, long long ts, int side, double price, double volume);
void    trades_get_at(const Trades *tr, int i,
                      long long *ts, int *side, double *price, double *volume);
int     trades_cmp_by_ts(const void *a, const void *b);
void    trades_sort_by_ts(Trades *tr);

/* ================================================================== */
/* Sequential best-price extractor (from checkpoints + delta log)     */
/* ================================================================== */

int bsearch_left_ll(const long long *arr, int n, long long val);

void lobts_seq_extract_best(
    const long long *all_ts,    int n_ts,
    const long long *ckpt_ts,   int n_ckpts,
    const int       *bid_offs,  const double *ckpt_bids,
    const int       *ask_offs,  const double *ckpt_asks,
    const long long *dl_ts,
    const uint8_t   *dl_side,
    const double    *dl_price,
    const double    *dl_qty,
    int              n_deltas,
    int              anchor_idx,
    double          *out_bids,
    double          *out_asks,
    double          *out_bidqs,
    double          *out_askqs);

/* ================================================================== */
/* Sequential full-state extractor (two-pass, produces CSR arrays)    */
/* ================================================================== */

void lobts_iter_seq_states(
    const long long *all_ts,    int n_ts,
    const long long *ckpt_ts,   int n_ckpts,
    const int       *bid_offs,  const double *ckpt_bids,
    const int       *ask_offs,  const double *ckpt_asks,
    const long long *dl_ts,
    const uint8_t   *dl_side,
    const double    *dl_price,
    const double    *dl_qty,
    int              n_deltas,
    int              anchor_idx,
    long long       *out_ts,
    int             *out_bid_offs,
    int             *out_ask_offs,
    double          *out_bid_data,
    double          *out_ask_data,
    int             *out_total_bid,
    int             *out_total_ask);

/* ================================================================== */
/* Guéant rolling intensity estimation                                */
/* ================================================================== */

void gueant_rolling_buckets(
    const long long *lob_ts,       int n_lob,
    const int       *bid_offs,     const double *bid_data,
    const int       *ask_offs,     const double *ask_data,
    long long        end_ts,
    const long long *trade_ts,     const double *trade_prices,
    const uint8_t   *trade_sides,  int n_trades,
    double           tick_size,    int gueant_side,  long long window_size,
    const int       *thresholds,   int n_buckets,
    const long long *all_ts,       int n_all_ts,
    double          *out_A,        double *out_k);

/* ================================================================== */
/* Hawkes MLE fitting                                                 */
/* ================================================================== */

int hawkes_mle_fit(
    const double *ts, const double *dt, int n,
    double *out_mu, double *out_alpha, double *out_beta);

#endif /* LOBCORE_H */
