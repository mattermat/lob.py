import os
import sys

import cffi

ffibuilder = cffi.FFI()

ffibuilder.cdef("""
    /* ---- LobBook (opaque) ---- */
    typedef ... LobBook;

    LobBook *lob_create(const char *name, double tick_size, long long timestamp);
    void     lob_destroy(LobBook *lob);
    LobBook *lob_copy(const LobBook *lob);

    void lob_set_snapshot(LobBook *lob, long long ts,
                          const double *bid_p, const double *bid_q, int n_bids,
                          const double *ask_p, const double *ask_q, int n_asks);

    void lob_set_updates(LobBook *lob, long long ts,
                         const uint8_t *sides, const double *prices,
                         const double *qtys, int n);

    void lob_update_level(LobBook *lob, long long ts, int side, double price, double qty);
    void lob_delete_bid(LobBook *lob, long long ts, double price);
    void lob_delete_ask(LobBook *lob, long long ts, double price);

    double lob_at(const LobBook *lob, int side, double price);

    void lob_get_bids(const LobBook *lob, const double **out_data, int *out_len);
    void lob_get_asks(const LobBook *lob, const double **out_data, int *out_len);

    double lob_spread(const LobBook *lob);
    double lob_midprice(const LobBook *lob);
    double lob_vw_midprice(const LobBook *lob);
    double lob_spread_tick(const LobBook *lob);
    double lob_spread_rel(const LobBook *lob);
    double lob_vi(const LobBook *lob, int nlevels);
    double lob_aggq_nlevel(const LobBook *lob, int side, int nlevel);
    double lob_aggq_ticks(const LobBook *lob, int side, int ticks);
    double lob_aggq_price(const LobBook *lob, int side, double price);
    double lob_slippage(const LobBook *lob, double volume, int side);
    int    lob_len_in_tick(const LobBook *lob, int side, double price);
    int    lob_check(const LobBook *lob);

    void        lob_set_tick_size(LobBook *lob, double tick_size);
    double      lob_get_tick_size(const LobBook *lob);
    long long   lob_timestamp(const LobBook *lob);
    void        lob_set_timestamp(LobBook *lob, long long ts);
    const char *lob_name(const LobBook *lob);

    /* ---- LOBts (opaque) ---- */
    typedef ... LOBts;

    LOBts *lobts_create(const char *name, double tick_size, int mode);
    void   lobts_destroy(LOBts *lobts);
    void   lobts_clear(LOBts *lobts);
    int    lobts_len(const LOBts *lobts);
    long long   lobts_last_ts(const LOBts *lobts);
    const char *lobts_get_name(const LOBts *lobts);

    /* Returns 0 on success, -1 if ts exists and force==0. */
    int lobts_set_snapshot(LOBts *lobts, long long ts,
                           const double *bid_p, const double *bid_q, int n_bids,
                           const double *ask_p, const double *ask_q, int n_asks,
                           int force);

    void lobts_set_updates(LOBts *lobts, long long ts,
                           const uint8_t *sides, const double *prices,
                           const double *qtys, int n);

    /* Borrowed pointer — do NOT call lob_destroy on the result. */
    LobBook *lobts_get(const LOBts *lobts, long long ts);
    LobBook *lobts_get_at(const LOBts *lobts, int i);
    long long lobts_ts_at(const LOBts *lobts, int i);

    void lobts_get_timestamps(const LOBts *lobts,
                               const long long **out_ts, int *out_len);

    /* Returns a new LOBts (caller must lobts_destroy). */
    LOBts *lobts_get_range(const LOBts *lobts,
                            long long start_ts, long long end_ts,
                            int has_start, int has_end);

    /* ---- Sequential full-state extractor (lazy-mode _iter_seq_states) ---- */
    void lobts_iter_seq_states(
        const long long *all_ts,   int n_ts,
        const long long *ckpt_ts,  int n_ckpts,
        const int       *bid_offs, const double *ckpt_bids,
        const int       *ask_offs, const double *ckpt_asks,
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
        int             *out_total_ask
    );

    /* ---- Sequential best-price extractor (lazy-mode analytics) ---- */
    void lobts_seq_extract_best(
        const long long *all_ts,   int n_ts,
        const long long *ckpt_ts,  int n_ckpts,
        const int       *bid_offs, const double *ckpt_bids,
        const int       *ask_offs, const double *ckpt_asks,
        const long long *dl_ts,
        const uint8_t   *dl_side,
        const double    *dl_price,
        const double    *dl_qty,
        int              n_deltas,
        int              anchor_idx,
        double          *out_bids,
        double          *out_asks,
        double          *out_bidqs,
        double          *out_askqs
    );

    /* ---- Guéant rolling estimation ---- */
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
        double          *out_A,        double *out_k
    );

    /* ---- Trades (opaque) ---- */
    typedef ... Trades;

    Trades *trades_create(void);
    void    trades_destroy(Trades *tr);
    void    trades_clear(Trades *tr);
    int     trades_len(const Trades *tr);

    void trades_append(Trades *tr, long long ts, int side, double price, double volume);
    void trades_get_at(const Trades *tr, int i,
                       long long *ts, int *side, double *price, double *volume);
    void trades_append_bulk(Trades *tr,
                             const long long *ts, const uint8_t *sides,
                             const double *prices, const double *volumes, int n);

    /* ---- Hawkes MLE ---- */
    /* ts  : timestamps shifted to start at 0, length n
       dt  : inter-arrival times (dt[0]=0, dt[i]=ts[i]-ts[i-1]), length n
       Returns 0 on success, 1 on failure; outputs set to NaN on failure. */
    int hawkes_mle_fit(const double *ts, const double *dt, int n,
                       double *out_mu, double *out_alpha, double *out_beta);

    /* ---- Full consistency checker ---- */
    void lobpy_validate_full(
        const long long *timestamps,
        const uint8_t   *event_types,
        const uint8_t   *sides,
        const double    *prices,
        const double    *quantities,
        int              n_rows,
        long long       *out_failed_ts,
        double          *out_failed_bid,
        double          *out_failed_ask,
        int             *out_n_failed
    );
""")

here = os.path.dirname(os.path.abspath(__file__))

# libm is a separate library on Linux/macOS; on Windows it is part of the C runtime.
_libraries = [] if sys.platform == "win32" else ["m"]

ffibuilder.set_source(
    "lobpy._cext._core",
    '#include "_core.h"',
    include_dirs=[here],
    libraries=_libraries,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
