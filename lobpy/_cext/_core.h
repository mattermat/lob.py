#ifndef LOB_CORE_H
#define LOB_CORE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* ================================================================== */
/* Side — one side of the order book (bids or asks)                   */
/* ================================================================== */

#define SIDE_INITIAL_CAP 16

typedef struct {
    double *data;   /* interleaved [price, qty, price, qty, ...] */
    int len;        /* number of price levels */
    int cap;        /* allocated capacity (in levels) */
    int is_bid;     /* 1 = bids (descending), 0 = asks (ascending) */
} Side;

static void side_init(Side *s, int is_bid) {
    s->data = (double *)malloc(SIDE_INITIAL_CAP * 2 * sizeof(double));
    s->len = 0;
    s->cap = SIDE_INITIAL_CAP;
    s->is_bid = is_bid;
}

static void side_free(Side *s) {
    free(s->data);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

static void side_grow(Side *s) {
    int new_cap = s->cap * 2;
    double *nd = (double *)realloc(s->data, (size_t)new_cap * 2 * sizeof(double));
    if (!nd) return;
    s->data = nd;
    s->cap = new_cap;
}

static int side_find(const Side *s, double price) {
    int lo = 0, hi = s->len - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        double mp = s->data[mid * 2];
        if (mp == price) return mid;
        if (s->is_bid) {
            if (mp > price) lo = mid + 1;
            else            hi = mid - 1;
        } else {
            if (mp < price) lo = mid + 1;
            else            hi = mid - 1;
        }
    }
    return -1;
}

static int side_insert_pos(const Side *s, double price) {
    int lo = 0, hi = s->len;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        double mp = s->data[mid * 2];
        if (s->is_bid) {
            if (mp > price) lo = mid + 1;
            else            hi = mid;
        } else {
            if (mp < price) lo = mid + 1;
            else            hi = mid;
        }
    }
    return lo;
}

static void side_insert_at(Side *s, int pos, double price, double qty) {
    if (s->len >= s->cap) side_grow(s);
    if (pos < s->len)
        memmove(&s->data[(pos + 1) * 2], &s->data[pos * 2],
                (size_t)(s->len - pos) * 2 * sizeof(double));
    s->data[pos * 2]     = price;
    s->data[pos * 2 + 1] = qty;
    s->len++;
}

static void side_delete_at(Side *s, int pos) {
    if (pos < s->len - 1)
        memmove(&s->data[pos * 2], &s->data[(pos + 1) * 2],
                (size_t)(s->len - pos - 1) * 2 * sizeof(double));
    s->len--;
}

static void side_update(Side *s, double price, double qty) {
    int idx = side_find(s, price);
    if (idx >= 0) {
        if (qty == 0.0)
            side_delete_at(s, idx);
        else
            s->data[idx * 2 + 1] = qty;
    } else if (qty != 0.0) {
        side_insert_at(s, side_insert_pos(s, price), price, qty);
    }
}

static void side_set(Side *s, const double *prices, const double *qtys, int n) {
    if (n > s->cap) {
        int nc = SIDE_INITIAL_CAP;
        while (nc < n) nc *= 2;
        free(s->data);
        s->data = (double *)malloc((size_t)nc * 2 * sizeof(double));
        s->cap = nc;
    }
    for (int i = 0; i < n; i++) {
        s->data[i * 2]     = prices[i];
        s->data[i * 2 + 1] = qtys[i];
    }
    s->len = n;
}

static void side_clear(Side *s) { s->len = 0; }

static double side_at(const Side *s, double price) {
    int idx = side_find(s, price);
    return idx >= 0 ? s->data[idx * 2 + 1] : 0.0;
}

static void side_sort_desc(double *data, int n) {
    for (int i = 1; i < n; i++) {
        double pk = data[i * 2], qk = data[i * 2 + 1];
        int j = i - 1;
        while (j >= 0 && data[j * 2] < pk) {
            data[(j+1)*2]   = data[j*2];
            data[(j+1)*2+1] = data[j*2+1];
            j--;
        }
        data[(j+1)*2]   = pk;
        data[(j+1)*2+1] = qk;
    }
}

static void side_sort_asc(double *data, int n) {
    for (int i = 1; i < n; i++) {
        double pk = data[i * 2], qk = data[i * 2 + 1];
        int j = i - 1;
        while (j >= 0 && data[j * 2] > pk) {
            data[(j+1)*2]   = data[j*2];
            data[(j+1)*2+1] = data[j*2+1];
            j--;
        }
        data[(j+1)*2]   = pk;
        data[(j+1)*2+1] = qk;
    }
}

static double side_aggq_nlevel(const Side *s, int nlevel) {
    if (nlevel <= 0 || s->len == 0) return 0.0;
    int end = nlevel < s->len ? nlevel : s->len;
    double total = 0.0;
    for (int i = 0; i < end; i++) total += s->data[i * 2 + 1];
    return total;
}

static double side_aggq_price(const Side *s, double price) {
    if (s->len == 0) return 0.0;
    double total = 0.0;
    if (s->is_bid) {
        for (int i = 0; i < s->len; i++) {
            if (s->data[i * 2] >= price) total += s->data[i * 2 + 1];
            else break;
        }
    } else {
        for (int i = 0; i < s->len; i++) {
            if (s->data[i * 2] <= price) total += s->data[i * 2 + 1];
            else break;
        }
    }
    return total;
}

/* Copy src into dst (dst must already be initialised). */
static void side_copy(Side *dst, const Side *src) {
    if (src->len == 0) { dst->len = 0; return; }
    if (src->len > dst->cap) {
        int nc = SIDE_INITIAL_CAP;
        while (nc < src->len) nc *= 2;
        free(dst->data);
        dst->data = (double *)malloc((size_t)nc * 2 * sizeof(double));
        dst->cap = nc;
    }
    memcpy(dst->data, src->data, (size_t)src->len * 2 * sizeof(double));
    dst->len = src->len;
}

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

static LobBook *lob_create(const char *name, double tick_size, long long timestamp) {
    LobBook *lob = (LobBook *)calloc(1, sizeof(LobBook));
    if (!lob) return NULL;
    side_init(&lob->bids, 1);
    side_init(&lob->asks, 0);
    lob->tick_size = tick_size;
    lob->timestamp = timestamp;
    if (name) {
        strncpy(lob->name, name, sizeof(lob->name) - 1);
        lob->name[sizeof(lob->name) - 1] = '\0';
    }
    return lob;
}

static void lob_destroy(LobBook *lob) {
    if (!lob) return;
    side_free(&lob->bids);
    side_free(&lob->asks);
    free(lob);
}

static LobBook *lob_copy(const LobBook *src) {
    LobBook *dst = (LobBook *)calloc(1, sizeof(LobBook));
    if (!dst) return NULL;
    side_init(&dst->bids, 1);
    side_init(&dst->asks, 0);
    side_copy(&dst->bids, &src->bids);
    side_copy(&dst->asks, &src->asks);
    dst->tick_size = src->tick_size;
    dst->timestamp = src->timestamp;
    strncpy(dst->name, src->name, sizeof(dst->name) - 1);
    return dst;
}

static void lob_set_snapshot(LobBook *lob, long long ts,
                              const double *bid_p, const double *bid_q, int n_bids,
                              const double *ask_p, const double *ask_q, int n_asks) {
    if (n_bids > 0 && bid_p && bid_q) {
        side_set(&lob->bids, bid_p, bid_q, n_bids);
        side_sort_desc(lob->bids.data, n_bids);
    } else {
        side_clear(&lob->bids);
    }
    if (n_asks > 0 && ask_p && ask_q) {
        side_set(&lob->asks, ask_p, ask_q, n_asks);
        side_sort_asc(lob->asks.data, n_asks);
    } else {
        side_clear(&lob->asks);
    }
    if (ts != 0) lob->timestamp = ts;
}

static void lob_set_updates(LobBook *lob, long long ts,
                             const uint8_t *sides, const double *prices,
                             const double *qtys, int n) {
    for (int i = 0; i < n; i++) {
        if (sides[i] == 0) side_update(&lob->bids, prices[i], qtys[i]);
        else               side_update(&lob->asks, prices[i], qtys[i]);
    }
    if (ts != 0) lob->timestamp = ts;
}

static void lob_update_level(LobBook *lob, long long ts, int side, double price, double qty) {
    if (side == 0) side_update(&lob->bids, price, qty);
    else           side_update(&lob->asks, price, qty);
    if (ts != 0) lob->timestamp = ts;
}

static void lob_delete_bid(LobBook *lob, long long ts, double price) {
    int idx = side_find(&lob->bids, price);
    if (idx >= 0) side_delete_at(&lob->bids, idx);
    if (ts != 0) lob->timestamp = ts;
}

static void lob_delete_ask(LobBook *lob, long long ts, double price) {
    int idx = side_find(&lob->asks, price);
    if (idx >= 0) side_delete_at(&lob->asks, idx);
    if (ts != 0) lob->timestamp = ts;
}

static double lob_at(const LobBook *lob, int side, double price) {
    return side == 0 ? side_at(&lob->bids, price) : side_at(&lob->asks, price);
}

static void lob_get_bids(const LobBook *lob, const double **out_data, int *out_len) {
    *out_data = lob->bids.data;
    *out_len  = lob->bids.len;
}

static void lob_get_asks(const LobBook *lob, const double **out_data, int *out_len) {
    *out_data = lob->asks.data;
    *out_len  = lob->asks.len;
}

static double lob_spread(const LobBook *lob) {
    if (lob->bids.len == 0 || lob->asks.len == 0) return NAN;
    double bp = lob->bids.data[0], ap = lob->asks.data[0];
    if (bp <= 0 || ap <= 0) return NAN;
    return ap - bp;
}

static double lob_midprice(const LobBook *lob) {
    if (lob->bids.len == 0 || lob->asks.len == 0) return NAN;
    double bp = lob->bids.data[0], ap = lob->asks.data[0];
    if (bp <= 0 || ap <= 0) return NAN;
    return (bp + ap) / 2.0;
}

static double lob_vw_midprice(const LobBook *lob) {
    if (lob->bids.len == 0 || lob->asks.len == 0) return NAN;
    double bp = lob->bids.data[0], bq = lob->bids.data[1];
    double ap = lob->asks.data[0], aq = lob->asks.data[1];
    if (bp <= 0 || ap <= 0 || bq <= 0 || aq <= 0) return NAN;
    return (bp * bq + ap * aq) / (bq + aq);
}

static double lob_spread_tick(const LobBook *lob) {
    double s = lob_spread(lob);
    return isnan(s) ? NAN : s / lob->tick_size;
}

static double lob_spread_rel(const LobBook *lob) {
    if (lob->bids.len == 0) return NAN;
    double mp = lob_midprice(lob);
    if (mp <= 0) return NAN;
    return lob_spread(lob) / mp;
}

static double lob_vi(const LobBook *lob, int nlevels) {
    if (lob->bids.len == 0 && lob->asks.len == 0) return 0.0;
    double bsum = 0.0, asum = 0.0;
    int bn = nlevels < lob->bids.len ? nlevels : lob->bids.len;
    int an = nlevels < lob->asks.len ? nlevels : lob->asks.len;
    for (int i = 0; i < bn; i++) bsum += lob->bids.data[i * 2 + 1];
    for (int i = 0; i < an; i++) asum += lob->asks.data[i * 2 + 1];
    if (bsum + asum == 0.0) return 0.0;
    return (bsum - asum) / (bsum + asum);
}

static double lob_aggq_nlevel(const LobBook *lob, int side, int nlevel) {
    return side == 0 ? side_aggq_nlevel(&lob->bids, nlevel)
                     : side_aggq_nlevel(&lob->asks, nlevel);
}

static double lob_aggq_ticks(const LobBook *lob, int side, int ticks) {
    if (lob->tick_size <= 0) return 0.0;
    double total = 0.0;
    if (side == 0) {
        if (lob->bids.len == 0) return 0.0;
        double min_p = lob->bids.data[0] - ticks * lob->tick_size;
        for (int i = 0; i < lob->bids.len; i++) {
            if (lob->bids.data[i * 2] >= min_p) total += lob->bids.data[i * 2 + 1];
            else break;
        }
    } else {
        if (lob->asks.len == 0) return 0.0;
        double max_p = lob->asks.data[0] + ticks * lob->tick_size;
        for (int i = 0; i < lob->asks.len; i++) {
            if (lob->asks.data[i * 2] <= max_p) total += lob->asks.data[i * 2 + 1];
            else break;
        }
    }
    return total;
}

static double lob_aggq_price(const LobBook *lob, int side, double price) {
    return side == 0 ? side_aggq_price(&lob->bids, price)
                     : side_aggq_price(&lob->asks, price);
}

static double lob_slippage(const LobBook *lob, double volume, int side) {
    if (volume <= 0) return 0.0;
    double mp = lob_midprice(lob);
    if (isnan(mp)) return NAN;
    double remaining = volume, total_cost = 0.0;
    if (side == 0) {
        for (int i = 0; i < lob->bids.len && remaining > 0; i++) {
            double p = lob->bids.data[i*2], q = lob->bids.data[i*2+1];
            double take = remaining < q ? remaining : q;
            total_cost += take * p; remaining -= take;
        }
        if (remaining > 0) return INFINITY;
        return mp - total_cost / volume;
    } else {
        for (int i = 0; i < lob->asks.len && remaining > 0; i++) {
            double p = lob->asks.data[i*2], q = lob->asks.data[i*2+1];
            double take = remaining < q ? remaining : q;
            total_cost += take * p; remaining -= take;
        }
        if (remaining > 0) return INFINITY;
        return total_cost / volume - mp;
    }
}

static int lob_len_in_tick(const LobBook *lob, int side, double price) {
    if (side == 0) {
        if (lob->bids.len == 0) return -1;
        double best = lob->bids.data[0];
        if (best <= 0) return -1;
        return (int)round((best - price) / lob->tick_size);
    } else {
        if (lob->asks.len == 0) return -1;
        double best = lob->asks.data[0];
        if (best <= 0) return -1;
        return (int)round((price - best) / lob->tick_size);
    }
}

static int lob_check(const LobBook *lob) {
    if (lob->bids.len == 0 || lob->asks.len == 0) return 1;
    return lob->bids.data[0] < lob->asks.data[0];
}

static void       lob_set_tick_size(LobBook *lob, double ts) { lob->tick_size = ts; }
static double     lob_get_tick_size(const LobBook *lob)      { return lob->tick_size; }
static long long  lob_timestamp(const LobBook *lob)          { return lob->timestamp; }
static void       lob_set_timestamp(LobBook *lob, long long ts) { lob->timestamp = ts; }
static const char *lob_name(const LobBook *lob)              { return lob->name; }

/* ================================================================== */
/* LOBts — time-series of LobBooks (delta and latest modes)           */
/* Lazy mode is handled entirely in Python using numpy.               */
/* ================================================================== */

#define LOBTS_DELTA   0
#define LOBTS_LATEST  1
#define LOBTS_INITIAL_CAP 16

typedef struct {
    char       name[64];
    double     tick_size;
    int        mode;
    long long *ts_arr;   /* sorted timestamp array (owned) */
    LobBook  **lob_arr;  /* parallel LobBook* array (owned) */
    int        len;
    int        cap;
} LOBts;

static void lobts_grow(LOBts *lt) {
    int nc = lt->cap * 2;
    long long *nt = (long long *)realloc(lt->ts_arr,  (size_t)nc * sizeof(long long));
    LobBook  **nl = (LobBook  **)realloc(lt->lob_arr, (size_t)nc * sizeof(LobBook *));
    if (!nt || !nl) return;
    lt->ts_arr  = nt;
    lt->lob_arr = nl;
    lt->cap = nc;
}

/* Binary search: return index of ts, or -1 if not found. */
static int lobts_find(const LOBts *lt, long long ts) {
    int lo = 0, hi = lt->len - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (lt->ts_arr[mid] == ts) return mid;
        if (lt->ts_arr[mid] < ts) lo = mid + 1;
        else                      hi = mid - 1;
    }
    return -1;
}

/* Return index where ts would be inserted to keep the array sorted. */
static int lobts_insert_pos(const LOBts *lt, long long ts) {
    int lo = 0, hi = lt->len;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (lt->ts_arr[mid] < ts) lo = mid + 1;
        else                      hi = mid;
    }
    return lo;
}

static void lobts_insert_at(LOBts *lt, int pos, long long ts, LobBook *lob) {
    if (lt->len >= lt->cap) lobts_grow(lt);
    if (pos < lt->len) {
        memmove(&lt->ts_arr[pos + 1],  &lt->ts_arr[pos],
                (size_t)(lt->len - pos) * sizeof(long long));
        memmove(&lt->lob_arr[pos + 1], &lt->lob_arr[pos],
                (size_t)(lt->len - pos) * sizeof(LobBook *));
    }
    lt->ts_arr[pos]  = ts;
    lt->lob_arr[pos] = lob;
    lt->len++;
}

static LOBts *lobts_create(const char *name, double tick_size, int mode) {
    LOBts *lt = (LOBts *)calloc(1, sizeof(LOBts));
    if (!lt) return NULL;
    if (name) {
        strncpy(lt->name, name, sizeof(lt->name) - 1);
        lt->name[sizeof(lt->name) - 1] = '\0';
    }
    lt->tick_size = tick_size;
    lt->mode      = mode;
    lt->ts_arr    = (long long *)malloc(LOBTS_INITIAL_CAP * sizeof(long long));
    lt->lob_arr   = (LobBook  **)malloc(LOBTS_INITIAL_CAP * sizeof(LobBook *));
    lt->len = 0;
    lt->cap = LOBTS_INITIAL_CAP;
    return lt;
}

static void lobts_destroy(LOBts *lt) {
    if (!lt) return;
    for (int i = 0; i < lt->len; i++) lob_destroy(lt->lob_arr[i]);
    free(lt->ts_arr);
    free(lt->lob_arr);
    free(lt);
}

static void lobts_clear(LOBts *lt) {
    for (int i = 0; i < lt->len; i++) lob_destroy(lt->lob_arr[i]);
    lt->len = 0;
}

static int       lobts_len(const LOBts *lt)     { return lt->len; }
static long long lobts_last_ts(const LOBts *lt) { return lt->len > 0 ? lt->ts_arr[lt->len - 1] : -1; }
static const char *lobts_get_name(const LOBts *lt) { return lt->name; }

/*
 * Set a full snapshot at ts.
 * Returns  0 on success.
 * Returns -1 if ts already exists and force == 0.
 */
static int lobts_set_snapshot(LOBts *lt, long long ts,
                               const double *bid_p, const double *bid_q, int n_bids,
                               const double *ask_p, const double *ask_q, int n_asks,
                               int force) {
    if (lt->mode == LOBTS_LATEST) lobts_clear(lt);

    int existing = lobts_find(lt, ts);
    if (existing >= 0) {
        if (!force) return -1;
        lob_set_snapshot(lt->lob_arr[existing], ts, bid_p, bid_q, n_bids, ask_p, ask_q, n_asks);
        return 0;
    }

    char lob_name[64];
    snprintf(lob_name, sizeof(lob_name) - 1, "%.55s_t%lld", lt->name, (long long)ts);
    LobBook *lob = lob_create(lob_name, lt->tick_size, ts);
    if (!lob) return -1;
    lob_set_snapshot(lob, ts, bid_p, bid_q, n_bids, ask_p, ask_q, n_asks);

    lobts_insert_at(lt, lobts_insert_pos(lt, ts), ts, lob);
    return 0;
}

/*
 * Apply incremental updates at ts.
 * Copies the last LOB state as a starting point (matching Python behaviour),
 * or creates an empty LOB if the series is empty.
 * If ts already exists, updates it in place.
 */
static void lobts_set_updates(LOBts *lt, long long ts,
                               const uint8_t *sides, const double *prices,
                               const double *qtys, int n) {
    int existing = lobts_find(lt, ts);
    LobBook *working;

    if (existing >= 0) {
        working = lt->lob_arr[existing];
    } else {
        char lob_name[64];
        snprintf(lob_name, sizeof(lob_name) - 1, "%.55s_t%lld", lt->name, (long long)ts);
        if (lt->len > 0) {
            /* Copy the most recent LOB — mirrors Python's keys()[-1] behaviour. */
            working = lob_copy(lt->lob_arr[lt->len - 1]);
            strncpy(working->name, lob_name, sizeof(working->name) - 1);
        } else {
            working = lob_create(lob_name, lt->tick_size, ts);
        }
        lobts_insert_at(lt, lobts_insert_pos(lt, ts), ts, working);
    }

    lob_set_updates(working, ts, sides, prices, qtys, n);
}

/* Get LOB at exact timestamp (borrowed — do NOT destroy). Returns NULL if not found. */
static LobBook *lobts_get(const LOBts *lt, long long ts) {
    int idx = lobts_find(lt, ts);
    return idx >= 0 ? lt->lob_arr[idx] : NULL;
}

/* Index-based access (for iteration). */
static LobBook  *lobts_get_at(const LOBts *lt, int i) { return (i >= 0 && i < lt->len) ? lt->lob_arr[i] : NULL; }
static long long lobts_ts_at (const LOBts *lt, int i) { return (i >= 0 && i < lt->len) ? lt->ts_arr[i]  : -1;   }

/* Expose the raw sorted timestamp array (borrowed). */
static void lobts_get_timestamps(const LOBts *lt, const long long **out_ts, int *out_len) {
    *out_ts  = lt->ts_arr;
    *out_len = lt->len;
}

/*
 * Return a new LOBts slice [start_ts, end_ts] (both inclusive).
 * has_start / has_end: 1 if the bound is active, 0 if unbounded.
 * The new LOBts owns deep copies of all included LobBooks.
 * Caller must call lobts_destroy() on the result.
 */
static LOBts *lobts_get_range(const LOBts *lt, long long start_ts, long long end_ts,
                               int has_start, int has_end) {
    LOBts *result = lobts_create(lt->name, lt->tick_size, lt->mode);
    if (!result) return NULL;
    for (int i = 0; i < lt->len; i++) {
        long long ts = lt->ts_arr[i];
        if (has_start && ts < start_ts) continue;
        if (has_end   && ts > end_ts)   break;
        LobBook *cp = lob_copy(lt->lob_arr[i]);
        if (!cp) continue;
        /* Append in order — no need to search for insert position. */
        if (result->len >= result->cap) lobts_grow(result);
        result->ts_arr [result->len] = ts;
        result->lob_arr[result->len] = cp;
        result->len++;
    }
    return result;
}

/* ================================================================== */
/* Sequential best-price extractor — lazy-mode analytics fast path    */
/* ================================================================== */

/*
 * Binary search: leftmost index i such that arr[i] >= val.
 * Returns n if all elements are < val.
 */
static int bsearch_left_ll(const long long *arr, int n, long long val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else                hi = mid;
    }
    return lo;
}

/*
 * Load a Side from a flat interleaved array [price0,qty0,price1,qty1,...].
 * The caller guarantees the data is already in the correct sort order
 * (descending for bids, ascending for asks).
 */
static void side_load_flat(Side *s, const double *flat, int n_levels) {
    if (n_levels == 0) { s->len = 0; return; }
    if (n_levels > s->cap) {
        int nc = s->cap;
        while (nc < n_levels) nc *= 2;
        free(s->data);
        s->data = (double *)malloc((size_t)nc * 2 * sizeof(double));
        s->cap  = nc;
    }
    memcpy(s->data, flat, (size_t)n_levels * 2 * sizeof(double));
    s->len = n_levels;
}

/*
 * Single forward pass through checkpoints + delta log.
 *
 * For each timestamp in all_ts[0..n_ts), computes the best bid price,
 * best ask price, best bid qty, and best ask qty, writing results into
 * the four pre-allocated out_* arrays (length n_ts).
 *
 * Parameters
 * ----------
 * all_ts        sorted int64 timestamps to emit (length n_ts)
 * ckpt_ts       sorted int64 checkpoint timestamps (length n_ckpts)
 * bid_offs      per-checkpoint start offsets into ckpt_bids (in levels),
 *               length n_ckpts+1 (last entry = total bid levels across all ckpts)
 * ckpt_bids     flat interleaved [price,qty,...] for all bid checkpoints,
 *               pre-sorted descending per checkpoint; total length = bid_offs[n_ckpts]*2
 * ask_offs      same layout for asks, length n_ckpts+1
 * ckpt_asks     flat interleaved [price,qty,...] for all ask checkpoints,
 *               pre-sorted ascending per checkpoint; total length = ask_offs[n_ckpts]*2
 * dl_ts/side/price/qty  delta log arrays (length n_deltas), sorted ascending
 *               by ts; side encoding: 0 = bid, 1 = ask
 * anchor_idx    index of the nearest checkpoint at or before all_ts[0],
 *               or -1 if no such checkpoint exists
 * out_*         caller-allocated float64 output arrays, length n_ts;
 *               NaN is written when the respective side is empty
 */
static void lobts_seq_extract_best(
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
    double          *out_askqs
) {
    Side bids, asks;
    side_init(&bids, 1);
    side_init(&asks, 0);

    /* Load anchor checkpoint state (book state at or before all_ts[0]) */
    if (anchor_idx >= 0) {
        int b0 = bid_offs[anchor_idx], b1 = bid_offs[anchor_idx + 1];
        side_load_flat(&bids, &ckpt_bids[b0 * 2], b1 - b0);
        int a0 = ask_offs[anchor_idx], a1 = ask_offs[anchor_idx + 1];
        side_load_flat(&asks, &ckpt_asks[a0 * 2], a1 - a0);
    }

    /* ckpt_cursor: next checkpoint in ckpt_ts we have not yet consumed */
    int ckpt_cursor = (anchor_idx >= 0) ? anchor_idx + 1 : 0;
    int delta_idx   = 0;

    for (int i = 0; i < n_ts; i++) {
        long long t = all_ts[i];

        /*
         * If the next checkpoint lands exactly at t, reset book state from it
         * and rewind delta_idx to the first delta at t.
         *
         * Invariant: since all_ts is the union of checkpoint timestamps and
         * delta-log timestamps (filtered to the requested range), every
         * checkpoint in range appears in all_ts, so ckpt_ts[ckpt_cursor] is
         * never < t when we reach t — it can only equal t or be larger.
         */
        if (ckpt_cursor < n_ckpts && ckpt_ts[ckpt_cursor] == t) {
            int k  = ckpt_cursor;
            int b0 = bid_offs[k], b1 = bid_offs[k + 1];
            side_load_flat(&bids, &ckpt_bids[b0 * 2], b1 - b0);
            int a0 = ask_offs[k], a1 = ask_offs[k + 1];
            side_load_flat(&asks, &ckpt_asks[a0 * 2], a1 - a0);
            delta_idx = bsearch_left_ll(dl_ts, n_deltas, t);
            ckpt_cursor++;
        }

        /* Apply all delta-log rows at timestamp t */
        while (delta_idx < n_deltas && dl_ts[delta_idx] == t) {
            if (dl_side[delta_idx] == 0)
                side_update(&bids, dl_price[delta_idx], dl_qty[delta_idx]);
            else
                side_update(&asks, dl_price[delta_idx], dl_qty[delta_idx]);
            delta_idx++;
        }

        out_bids[i]  = bids.len > 0 ? bids.data[0] : NAN;
        out_asks[i]  = asks.len > 0 ? asks.data[0] : NAN;
        out_bidqs[i] = bids.len > 0 ? bids.data[1] : NAN;
        out_askqs[i] = asks.len > 0 ? asks.data[1] : NAN;
    }

    side_free(&bids);
    side_free(&asks);
}

/*
 * Sequential full-state extractor for _iter_seq_states.
 *
 * Shares the same checkpoint + delta-log inputs as lobts_seq_extract_best.
 * Outputs the complete bid/ask book at every timestamp in CSR format:
 *
 *   out_ts[i]                          — timestamp i
 *   out_bid_offs[i]..out_bid_offs[i+1] — range in out_bid_data for timestamp i
 *   out_bid_data[off*2 .. (off+len)*2] — interleaved [price,qty,...], descending
 *   (same for ask side, ascending)
 *
 * Two-call protocol (avoids C-side dynamic allocation):
 *
 *   Call 1 — count pass: pass NULL for out_bid_data and out_ask_data.
 *     Fills out_ts, out_bid_offs, out_ask_offs; writes total levels to
 *     *out_total_bid and *out_total_ask.
 *
 *   Call 2 — fill pass: allocate out_bid_data[*out_total_bid * 2] and
 *     out_ask_data[*out_total_ask * 2], then call again with the same inputs.
 *     Fills the data arrays in addition to the offset tables.
 */
static void lobts_iter_seq_states(
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
    long long       *out_ts,        /* length n_ts */
    int             *out_bid_offs,  /* length n_ts+1 */
    int             *out_ask_offs,  /* length n_ts+1 */
    double          *out_bid_data,  /* NULL → count only; else length *out_total_bid * 2 */
    double          *out_ask_data,  /* NULL → count only; else length *out_total_ask * 2 */
    int             *out_total_bid,
    int             *out_total_ask
) {
    Side bids, asks;
    side_init(&bids, 1);
    side_init(&asks, 0);

    if (anchor_idx >= 0) {
        int b0 = bid_offs[anchor_idx], b1 = bid_offs[anchor_idx + 1];
        side_load_flat(&bids, &ckpt_bids[b0 * 2], b1 - b0);
        int a0 = ask_offs[anchor_idx], a1 = ask_offs[anchor_idx + 1];
        side_load_flat(&asks, &ckpt_asks[a0 * 2], a1 - a0);
    }

    int ckpt_cursor = (anchor_idx >= 0) ? anchor_idx + 1 : 0;
    int delta_idx   = 0;
    int bid_cursor  = 0;
    int ask_cursor  = 0;

    out_bid_offs[0] = 0;
    out_ask_offs[0] = 0;

    for (int i = 0; i < n_ts; i++) {
        long long t = all_ts[i];
        out_ts[i] = t;

        if (ckpt_cursor < n_ckpts && ckpt_ts[ckpt_cursor] == t) {
            int k  = ckpt_cursor;
            int b0 = bid_offs[k], b1 = bid_offs[k + 1];
            side_load_flat(&bids, &ckpt_bids[b0 * 2], b1 - b0);
            int a0 = ask_offs[k], a1 = ask_offs[k + 1];
            side_load_flat(&asks, &ckpt_asks[a0 * 2], a1 - a0);
            delta_idx = bsearch_left_ll(dl_ts, n_deltas, t);
            ckpt_cursor++;
        }

        while (delta_idx < n_deltas && dl_ts[delta_idx] == t) {
            if (dl_side[delta_idx] == 0)
                side_update(&bids, dl_price[delta_idx], dl_qty[delta_idx]);
            else
                side_update(&asks, dl_price[delta_idx], dl_qty[delta_idx]);
            delta_idx++;
        }

        out_bid_offs[i + 1] = bid_cursor + bids.len;
        out_ask_offs[i + 1] = ask_cursor + asks.len;

        if (out_bid_data != NULL && bids.len > 0)
            memcpy(&out_bid_data[bid_cursor * 2], bids.data,
                   (size_t)bids.len * 2 * sizeof(double));
        if (out_ask_data != NULL && asks.len > 0)
            memcpy(&out_ask_data[ask_cursor * 2], asks.data,
                   (size_t)asks.len * 2 * sizeof(double));

        bid_cursor += bids.len;
        ask_cursor += asks.len;
    }

    *out_total_bid = bid_cursor;
    *out_total_ask = ask_cursor;

    side_free(&bids);
    side_free(&asks);
}

/* ================================================================== */
/* Guéant rolling intensity estimation                                */
/* ================================================================== */

/* Return bucket index for integer delta d, given sorted int thresholds.
   Bin b covers delta in (thresholds[b-1], thresholds[b]] (bin 0 from 0).
   Returns n_buckets for overflow (d > all thresholds). */
static int gueant_find_bucket(int d, const int *thresholds, int n_buckets) {
    for (int b = 0; b < n_buckets; b++)
        if (d <= thresholds[b]) return b;
    return n_buckets;
}

/* OLS log-linear fit: log(N[b]/T[b]) = intercept + slope * thresholds[b]
   => A = exp(intercept), k = -slope.
   Sets *out_A = *out_k = NAN when fewer than 2 valid (T>0 and N>0) points. */
static void gueant_fit(const int *thresholds, const double *T_agg,
                       const double *N_agg, int n_buckets,
                       double *out_A, double *out_k) {
    int    n_v   = 0;
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int b = 0; b < n_buckets; b++) {
        if (T_agg[b] <= 0.0 || N_agg[b] <= 0.0) continue;
        double x  = (double)thresholds[b];
        double y  = log(N_agg[b] / T_agg[b]);
        sum_x  += x; sum_y  += y;
        sum_xx += x * x; sum_xy += x * y;
        n_v++;
    }
    if (n_v < 2) { *out_A = NAN; *out_k = NAN; return; }
    double denom = (double)n_v * sum_xx - sum_x * sum_x;
    if (denom == 0.0) { *out_A = NAN; *out_k = NAN; return; }
    double slope     = ((double)n_v * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / (double)n_v;
    *out_A = exp(intercept);
    *out_k = -slope;
}

/* Rolling Guéant estimation with fixed buckets.
 *
 * Inputs (CSR LOB state from lobts_iter_seq_states):
 *   lob_ts[0..n_lob-1]          : LOB change timestamps (sorted)
 *   bid_offs[0..n_lob], bid_data : CSR bids (descending price order → data[0]=best bid)
 *   ask_offs[0..n_lob], ask_data : CSR asks (ascending price order  → data[0]=best ask)
 *   end_ts                       : effective end of last LOB interval
 *
 * Trades (sorted by timestamp):
 *   trade_ts, trade_prices, trade_sides[i] ∈ {0=buy-aggressor, 1=sell-aggressor}
 *
 * Parameters:
 *   tick_size, gueant_side (0=ask, 1=bid), window_size (nanoseconds)
 *   thresholds[0..n_buckets-1] : sorted integer delta upper bounds, e.g. [1,3,5,10]
 *
 * Query timestamps (sorted, typically lob_ts ∪ trade_ts):
 *   all_ts[0..n_all_ts-1]
 *
 * Outputs: out_A[n_all_ts], out_k[n_all_ts]
 */
static void gueant_rolling_buckets(
    const long long *lob_ts,       int n_lob,
    const int       *bid_offs,     const double *bid_data,
    const int       *ask_offs,     const double *ask_data,
    long long        end_ts,
    const long long *trade_ts,     const double *trade_prices,
    const uint8_t   *trade_sides,  int n_trades,
    double           tick_size,    int gueant_side,  long long window_size,
    const int       *thresholds,   int n_buckets,
    const long long *all_ts,       int n_all_ts,
    double          *out_A,        double *out_k)
{
    if (n_lob <= 0 || n_all_ts <= 0 || n_buckets <= 0) return;

    int K = n_buckets;

    /* --- Allocate working arrays --- */
    int       *interval_counts = (int       *)calloc((size_t)n_lob * K, sizeof(int));
    long long *interval_dur    = (long long *)malloc ((size_t)n_lob     * sizeof(long long));
    long long *intv_end        = (long long *)malloc ((size_t)n_lob     * sizeof(long long));
    double    *T_cumsum        = (double    *)calloc ((size_t)(n_lob+1) * K, sizeof(double));
    long long *rtrade_ts       = (long long *)malloc ((size_t)(n_trades > 0 ? n_trades : 1) * sizeof(long long));
    int       *rtrade_bucket   = (int       *)malloc ((size_t)(n_trades > 0 ? n_trades : 1) * sizeof(int));
    double    *T_agg           = (double    *)malloc ((size_t)K * sizeof(double));
    double    *N_agg           = (double    *)malloc ((size_t)K * sizeof(double));
    int       *N_window        = (int       *)calloc ((size_t)K, sizeof(int));

    if (!interval_counts || !interval_dur || !intv_end || !T_cumsum ||
        !rtrade_ts || !rtrade_bucket || !T_agg || !N_agg || !N_window)
        goto cleanup;

    /* --- Pre-compute interval end times --- */
    for (int i = 0; i < n_lob; i++)
        intv_end[i] = (i + 1 < n_lob) ? lob_ts[i + 1] : end_ts;

    /* --- Pre-compute per-interval bucket level counts and durations --- */
    for (int i = 0; i < n_lob; i++) {
        long long dur = intv_end[i] - lob_ts[i];
        interval_dur[i] = dur > 0 ? dur : 0;

        if (dur <= 0) continue;

        int n_bids = bid_offs[i + 1] - bid_offs[i];
        int n_asks = ask_offs[i + 1] - ask_offs[i];
        if (n_bids == 0 || n_asks == 0) continue;

        double best_bid = bid_data[(size_t)bid_offs[i] * 2];   /* highest bid */
        double best_ask = ask_data[(size_t)ask_offs[i] * 2];   /* lowest  ask */
        if (best_bid <= 0.0 || best_ask <= 0.0) continue;

        int *ic = &interval_counts[i * K];

        if (gueant_side == 0) {                     /* ask side */
            int off = ask_offs[i];
            for (int j = 0; j < n_asks; j++) {
                double price = ask_data[(size_t)(off + j) * 2];
                int d = (int)round((price - best_bid) / tick_size);
                if (d < 0) continue;
                int b = gueant_find_bucket(d, thresholds, K);
                if (b < K) ic[b]++;
            }
        } else {                                    /* bid side */
            int off = bid_offs[i];
            for (int j = 0; j < n_bids; j++) {
                double price = bid_data[(size_t)(off + j) * 2];
                int d = (int)round((best_ask - price) / tick_size);
                if (d < 0) continue;
                int b = gueant_find_bucket(d, thresholds, K);
                if (b < K) ic[b]++;
            }
        }
    }

    /* --- Build T_cumsum[(n_lob+1) × K] --- */
    for (int i = 0; i < n_lob; i++) {
        for (int b = 0; b < K; b++) {
            T_cumsum[(i + 1) * K + b] = T_cumsum[i * K + b]
                + (double)interval_counts[i * K + b] * (double)interval_dur[i];
        }
    }

    /* --- Pre-compute relevant trade deltas → (rtrade_ts, rtrade_bucket) --- */
    uint8_t target_side = (gueant_side == 0) ? 0 : 1;  /* 0=buy-agg, 1=sell-agg */
    int n_rtrades = 0;
    for (int t = 0; t < n_trades; t++) {
        if (trade_sides[t] != target_side) continue;

        long long tts = trade_ts[t];
        /* Find active LOB state: last index with lob_ts[j] <= tts */
        int li = bsearch_left_ll(lob_ts, n_lob, tts);
        if (li == n_lob || lob_ts[li] > tts) li--;
        if (li < 0) continue;

        int nb = bid_offs[li + 1] - bid_offs[li];
        int na = ask_offs[li + 1] - ask_offs[li];
        if (nb == 0 || na == 0) continue;

        double bb = bid_data[(size_t)bid_offs[li] * 2];
        double ba = ask_data[(size_t)ask_offs[li] * 2];
        if (bb <= 0.0 || ba <= 0.0) continue;

        int d = (gueant_side == 0)
            ? (int)round((trade_prices[t] - bb) / tick_size)
            : (int)round((ba - trade_prices[t]) / tick_size);
        if (d < 0) continue;

        int b = gueant_find_bucket(d, thresholds, K);
        if (b >= K) continue;   /* overflow trade */

        rtrade_ts[n_rtrades]     = tts;
        rtrade_bucket[n_rtrades] = b;
        n_rtrades++;
    }

    /* --- Rolling loop over all_ts --- */
    int tl = 0, tr = 0;   /* sliding window into rtrade_ts: [tl, tr) */

    for (int qi = 0; qi < n_all_ts; qi++) {
        long long ts = all_ts[qi];
        long long lo = ts - window_size;

        /* Advance trade right pointer: add trades with ts <= ts */
        while (tr < n_rtrades && rtrade_ts[tr] <= ts) {
            N_window[rtrade_bucket[tr]]++;
            tr++;
        }
        /* Advance trade left pointer: remove trades with ts < lo */
        while (tl < tr && rtrade_ts[tl] < lo) {
            N_window[rtrade_bucket[tl]]--;
            tl++;
        }

        /* T interval range: [ileft, iright)
           ileft  = first interval whose end   > lo  (has overlap with [lo, ts])
           iright = first interval whose start > ts  (starts after ts) */
        int ileft  = bsearch_left_ll(intv_end, n_lob, lo + 1);
        int iright = bsearch_left_ll(lob_ts,   n_lob, ts + 1);

        if (ileft >= iright) {
            out_A[qi] = NAN;
            out_k[qi] = NAN;
            continue;
        }

        /* T_agg = T_cumsum[iright] - T_cumsum[ileft] (full durations) */
        for (int b = 0; b < K; b++)
            T_agg[b] = T_cumsum[iright * K + b] - T_cumsum[ileft * K + b];

        /* Boundary correction for ileft (left edge may start before lo) */
        {
            int i = ileft;
            long long full_dur = interval_dur[i];
            if (full_dur > 0) {
                long long lo_c = lo > lob_ts[i] ? lo : lob_ts[i];
                long long hi_c = ts < intv_end[i] ? ts : intv_end[i];
                long long clipped = hi_c - lo_c;
                if (clipped != full_dur) {
                    long long diff = clipped - full_dur;
                    for (int b = 0; b < K; b++)
                        T_agg[b] += (double)interval_counts[i * K + b] * (double)diff;
                }
            }
        }

        /* Boundary correction for iright-1 (right edge may extend past ts) */
        if (iright - 1 != ileft) {
            int i = iright - 1;
            long long full_dur = interval_dur[i];
            if (full_dur > 0) {
                long long lo_c = lo > lob_ts[i] ? lo : lob_ts[i];
                long long hi_c = ts < intv_end[i] ? ts : intv_end[i];
                long long clipped = hi_c - lo_c;
                if (clipped != full_dur) {
                    long long diff = clipped - full_dur;
                    for (int b = 0; b < K; b++)
                        T_agg[b] += (double)interval_counts[i * K + b] * (double)diff;
                }
            }
        }

        /* Collect N per bucket */
        for (int b = 0; b < K; b++)
            N_agg[b] = (double)N_window[b];

        gueant_fit(thresholds, T_agg, N_agg, K, &out_A[qi], &out_k[qi]);
    }

cleanup:
    free(interval_counts);
    free(interval_dur);
    free(intv_end);
    free(T_cumsum);
    free(rtrade_ts);
    free(rtrade_bucket);
    free(T_agg);
    free(N_agg);
    free(N_window);
}

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

static Trades *trades_create(void) {
    Trades *tr = (Trades *)calloc(1, sizeof(Trades));
    if (!tr) return NULL;
    tr->data = (TradeEntry *)malloc(TRADES_INITIAL_CAP * sizeof(TradeEntry));
    tr->len  = 0;
    tr->cap  = TRADES_INITIAL_CAP;
    return tr;
}

static void trades_destroy(Trades *tr) {
    if (!tr) return;
    free(tr->data);
    free(tr);
}

static void trades_clear(Trades *tr) { tr->len = 0; }
static int  trades_len(const Trades *tr) { return tr->len; }

static void trades_grow(Trades *tr) {
    int nc = tr->cap * 2;
    TradeEntry *nd = (TradeEntry *)realloc(tr->data, (size_t)nc * sizeof(TradeEntry));
    if (!nd) return;
    tr->data = nd;
    tr->cap  = nc;
}

static void trades_append(Trades *tr, long long ts, int side, double price, double volume) {
    if (tr->len >= tr->cap) trades_grow(tr);
    TradeEntry *e = &tr->data[tr->len++];
    e->timestamp = ts;
    e->side      = (uint8_t)side;
    e->price     = price;
    e->volume    = volume;
}

static void trades_get_at(const Trades *tr, int i,
                           long long *ts, int *side, double *price, double *volume) {
    if (i < 0 || i >= tr->len) { *ts = 0; *side = 0; *price = 0.0; *volume = 0.0; return; }
    const TradeEntry *e = &tr->data[i];
    *ts     = e->timestamp;
    *side   = (int)e->side;
    *price  = e->price;
    *volume = e->volume;
}

static void trades_append_bulk(Trades *tr,
                                const long long *ts, const uint8_t *sides,
                                const double *prices, const double *volumes, int n) {
    for (int i = 0; i < n; i++)
        trades_append(tr, ts[i], (int)sides[i], prices[i], volumes[i]);
}

#endif /* LOB_CORE_H */
