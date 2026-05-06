/*
 * get_features - Extract trading features from raw LOB+trades parquet.
 *
 * Usage:
 *   ./get_features <input.parquet> [options]
 *
 * Reads a parquet file with columns:
 *   timestamp (int64), event_type (string), side (string),
 *   price (double), quantity (double)
 *
 * Outputs a CSV with one row per LOB timestamp:
 *   timestamp, spread, bid, ask, midprice, bidq, askq, vw_midprice, vi,
 *   realized_vol, gueant_A_ask, gueant_k_ask, gueant_A_bid, gueant_k_bid,
 *   ohlc_open, ohlc_high, ohlc_low, ohlc_close, ohlc_volume
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <carquet/carquet.h>
#include "lobcore.h"

/* ── Constants ──────────────────────────────────────────────────────────── */

#define MAX_BOOK_LEVELS  2048
#define OHLC_PERIOD_NS   1000000000LL
#define DEFAULT_WINDOW   60
#define NAN_D            ((double)NAN)
#define SEC_TO_NS        1000000000LL
#define MAX_PHASES       32

/* ── Timing ─────────────────────────────────────────────────────────────── */

typedef struct { const char *name; double elapsed; } Phase;
static Phase g_phases[MAX_PHASES];
static int g_n_phases = 0;
static clock_t g_phase_start;

static void phase_begin(void) { g_phase_start = clock(); }
static void phase_end(const char *name) {
    if (g_n_phases < MAX_PHASES)
        g_phases[g_n_phases++] = (Phase){name, (double)(clock() - g_phase_start) / CLOCKS_PER_SEC};
}
static void phase_skip(void) { /* timed but not logged */ }

/* ── Column registry ───────────────────────────────────────────────────── */

typedef struct { const char *name; const char *kind; int window_s; } Column;
static Column g_cols[64];
static int g_n_cols = 0;

static void col_add(const char *name, const char *kind, int window_s) {
    if (g_n_cols < 64) g_cols[g_n_cols++] = (Column){name, kind, window_s};
}
static void col_immediate(const char *name) { col_add(name, "immediate", 0); }
static void col_rolling(const char *name, int w) { col_add(name, "rolling", w); }

static void print_timing_table(void) {
    /* ── Bullet list: feature groups with timing ───────────────────── */
    printf("\n  timing:\n");
    double total = 0;
    for (int i = 0; i < g_n_phases; i++) {
        printf("    %-45s %7.2fs\n", g_phases[i].name, g_phases[i].elapsed);
        total += g_phases[i].elapsed;
    }
    printf("    %-45s %7.2fs\n", "total", total);

    /* ── Table: one row per output column ──────────────────────────── */
    printf("\n╔══════════════════════════════════════════════╤══════════╤═══════════╗\n");
    printf("║ Column                                       │ Type     │ Window    ║\n");
    printf("╠══════════════════════════════════════════════╪══════════╪═══════════╣\n");
    for (int i = 0; i < g_n_cols; i++) {
        char win[10] = "—";
        if (g_cols[i].window_s > 0) snprintf(win, sizeof(win), "%ds", g_cols[i].window_s);
        printf("║ %-44s │ %-8s │ %-9s ║\n", g_cols[i].name, g_cols[i].kind, win);
    }
    printf("╚══════════════════════════════════════════════╧══════════╧═══════════╝\n");
}

/* ── Event types (matching the parquet schema) ──────────────────────────── */

typedef enum { EVT_BOOK_LEVEL = 0, EVT_BOOK_UPDATE = 1, EVT_TRADE = 2 } EventType;

/* ── Parsed row ─────────────────────────────────────────────────────────── */

typedef struct {
    long long  timestamp;
    uint8_t    event_type;  /* 0=book_level, 1=book_update, 2=trade */
    uint8_t    side;        /* 0=bid, 1=ask  (for trades: 0=buy-agg, 1=sell-agg) */
    double     price;
    double     quantity;
} Row;

static int g_num_rows = 0;
static Row *g_rows = NULL;

/* ── Comparison helpers ─────────────────────────────────────────────────── */

static int row_cmp_by_ts(const void *a, const void *b) {
    long long ta = ((const Row *)a)->timestamp;
    long long tb = ((const Row *)b)->timestamp;
    if (ta < tb) return -1;
    if (ta > tb) return 1;
    /* book_level before book_update before trade within same timestamp */
    int ea = ((const Row *)a)->event_type;
    int eb = ((const Row *)b)->event_type;
    return ea - eb;
}

static int ll_cmp(const void *a, const void *b) {
    long long va = *(const long long *)a;
    long long vb = *(const long long *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* ── Read parquet file into g_rows ─────────────────────────────────────── */

/* Read all values from a numeric column. */
static int read_column_data(carquet_reader_t *reader, int col_idx,
                             void *out, size_t elem_size) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    int64_t n_rows = carquet_reader_num_rows(reader);
    int32_t n_rg = carquet_reader_num_row_groups(reader);

    size_t offset = 0;
    for (int32_t rg = 0; rg < n_rg; rg++) {
        carquet_column_reader_t *col = carquet_reader_get_column(reader, rg, col_idx, &err);
        if (!col) { fprintf(stderr, "read column %d rg %d: %s\n", col_idx, rg, err.message); return -1; }

        /* Read entire row group in one batch */
        int64_t remaining = carquet_column_remaining(col);
        int64_t n = carquet_column_read_batch(
            col, (char *)out + offset * elem_size, remaining, NULL, NULL);
        carquet_column_reader_free(col);
        if (n < 0) { fprintf(stderr, "error reading col %d\n", col_idx); return -1; }
        offset += n;
    }
    return 0;
}

/* Read a string (BYTE_ARRAY) column into flat char buffer + offset array.
   Caller must free *out_data and *out_offs. */
static int read_string_column(carquet_reader_t *reader, int col_idx,
                               uint32_t **out_offs, char **out_data,
                               int64_t *out_data_len) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    int64_t n_rows = carquet_reader_num_rows(reader);
    int32_t n_rg = carquet_reader_num_row_groups(reader);

    /* First pass: compute total data size */
    int64_t total_data = 0;
    for (int32_t rg = 0; rg < n_rg; rg++) {
        carquet_column_reader_t *col = carquet_reader_get_column(reader, rg, col_idx, &err);
        if (!col) return -1;
        int64_t n_left = carquet_column_remaining(col);
        /* Read all byte arrays */
        carquet_byte_array_t *bas = (carquet_byte_array_t *)malloc(
            n_left * sizeof(carquet_byte_array_t));
        int64_t n = carquet_column_read_batch(col, bas, n_left, NULL, NULL);
        if (n < 0) { free(bas); carquet_column_reader_free(col); return -1; }
        for (int64_t i = 0; i < n; i++) total_data += bas[i].length;
        free(bas);
        carquet_column_reader_free(col);
    }

    /* Allocate output buffers */
    *out_data = (char *)malloc(total_data);
    *out_offs = (uint32_t *)malloc((n_rows + 1) * sizeof(uint32_t));
    if (!*out_data || !*out_offs) { free(*out_data); free(*out_offs); return -1; }

    (*out_offs)[0] = 0;
    size_t data_cursor = 0;
    int64_t row_cursor = 0;

    /* Second pass: copy data */
    for (int32_t rg = 0; rg < n_rg; rg++) {
        carquet_column_reader_t *col = carquet_reader_get_column(reader, rg, col_idx, &err);
        if (!col) { free(*out_data); free(*out_offs); return -1; }
        int64_t n_left = carquet_column_remaining(col);
        carquet_byte_array_t *bas = (carquet_byte_array_t *)malloc(
            n_left * sizeof(carquet_byte_array_t));
        int64_t n = carquet_column_read_batch(col, bas, n_left, NULL, NULL);
        if (n < 0) { free(bas); carquet_column_reader_free(col); return -1; }
        for (int64_t i = 0; i < n; i++) {
            if (bas[i].data && bas[i].length > 0)
                memcpy(*out_data + data_cursor, bas[i].data, bas[i].length);
            data_cursor += bas[i].length;
            (*out_offs)[row_cursor + i + 1] = (uint32_t)data_cursor;
        }
        free(bas);
        row_cursor += n;
        carquet_column_reader_free(col);
    }

    *out_data_len = total_data;
    return 0;
}

static int parse_parquet(const char *path) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t *reader = carquet_reader_open(path, NULL, &err);
    if (!reader) {
        fprintf(stderr, "Error opening %s: %s\n", path, err.message);
        return -1;
    }

    int64_t n_rows = carquet_reader_num_rows(reader);
    int32_t n_cols = carquet_reader_num_columns(reader);

    g_num_rows = (int)n_rows;
    g_rows = (Row *)malloc(n_rows * sizeof(Row));
    if (!g_rows) {
        fprintf(stderr, "Out of memory\n");
        carquet_reader_close(reader);
        return -1;
    }

    /* Column names are case-sensitive; try to find standard columns */
    int32_t ts_col = -1, et_col = -1, side_col = -1, price_col = -1, qty_col = -1;

    const carquet_schema_t *schema = carquet_reader_schema(reader);
    for (int32_t i = 0; i < n_cols; i++) {
        const char *name = carquet_schema_column_name(schema, i);
        if (!name) continue;
        if (strcmp(name, "timestamp") == 0) ts_col = i;
        else if (strcmp(name, "event_type") == 0) et_col = i;
        else if (strcmp(name, "side") == 0) side_col = i;
        else if (strcmp(name, "price") == 0) price_col = i;
        else if (strcmp(name, "quantity") == 0) qty_col = i;
    }

    if (ts_col < 0 || et_col < 0 || side_col < 0 || price_col < 0 || qty_col < 0) {
        fprintf(stderr, "Missing required columns (need timestamp,event_type,side,price,quantity)\n");
        carquet_reader_close(reader);
        return -1;
    }

    /* Allocate temp arrays */
    long long *timestamps = (long long *)malloc(n_rows * sizeof(long long));
    double    *prices     = (double *)    malloc(n_rows * sizeof(double));
    double    *quantities = (double *)    malloc(n_rows * sizeof(double));

    if (!timestamps || !prices || !quantities) {
        fprintf(stderr, "Out of memory\n");
        free(timestamps); free(prices); free(quantities);
        carquet_reader_close(reader);
        return -1;
    }

    /* Read numeric columns */
    if (read_column_data(reader, ts_col, timestamps, sizeof(long long)) < 0) goto fail;
    if (read_column_data(reader, price_col, prices, sizeof(double)) < 0) goto fail;
    if (read_column_data(reader, qty_col, quantities, sizeof(double)) < 0) goto fail;

    /* Read string columns */
    uint32_t *et_offs = NULL, *side_offs = NULL;
    char *et_data = NULL, *side_data = NULL;
    int64_t et_data_len = 0, side_data_len = 0;

    if (read_string_column(reader, et_col, &et_offs, &et_data, &et_data_len) < 0) goto fail;
    if (read_string_column(reader, side_col, &side_offs, &side_data, &side_data_len) < 0) {
        free(et_offs); free(et_data);
        goto fail;
    }

    /* Parse into g_rows */
    for (int64_t i = 0; i < n_rows; i++) {
        Row *r = &g_rows[i];
        r->timestamp = timestamps[i];
        r->price     = prices[i];
        r->quantity  = quantities[i];

        /* event_type string */
        uint32_t et_start = et_offs[i];
        uint32_t et_len   = et_offs[i + 1] - et_start;
        char *et_str = et_data + et_start;

        if (et_len == 10 && memcmp(et_str, "book_level", 10) == 0)
            r->event_type = EVT_BOOK_LEVEL;
        else if (et_len == 11 && memcmp(et_str, "book_update", 11) == 0)
            r->event_type = EVT_BOOK_UPDATE;
        else if (et_len == 5 && memcmp(et_str, "trade", 5) == 0)
            r->event_type = EVT_TRADE;
        else {
            fprintf(stderr, "Unknown event_type at row %lld: '%.*s'\n",
                    (long long)i, (int)et_len, et_str);
            r->event_type = EVT_TRADE; /* skip unknown */
        }

        /* side string */
        uint32_t si_start = side_offs[i];
        uint32_t si_len   = side_offs[i + 1] - si_start;
        char *si_str = side_data + si_start;

        if (si_len == 3 && memcmp(si_str, "bid", 3) == 0)
            r->side = 0;
        else if (si_len == 3 && memcmp(si_str, "ask", 3) == 0)
            r->side = 1;
        else if (si_len == 3 && memcmp(si_str, "buy", 3) == 0)
            r->side = 0;  /* buy-aggressor */
        else if (si_len == 4 && memcmp(si_str, "sell", 4) == 0)
            r->side = 1;  /* sell-aggressor */
        else
            r->side = 2;  /* unknown */
    }

    free(et_offs); free(et_data);
    free(side_offs); free(side_data);

    carquet_reader_close(reader);
    free(timestamps); free(prices); free(quantities);

    /* Sort by timestamp (then event_type within same ts) */
    qsort(g_rows, g_num_rows, sizeof(Row), row_cmp_by_ts);
    return 0;

fail:
    free(timestamps); free(prices); free(quantities);
    free(et_offs); free(et_data);
    free(side_offs); free(side_data);
    carquet_reader_close(reader);
    return -1;
}

/* ── Collect ALL unique timestamps (LOB + trades) ───────────────────────── */

static long long *collect_all_timestamps(int *out_n) {
    /* Two passes: first count, then fill */
    int n = 0;
    long long last = 0;
    for (int i = 0; i < g_num_rows; i++) {
        long long ts = g_rows[i].timestamp;
        if (n == 0 || ts != last) { n++; last = ts; }
    }
    long long *arr = (long long *)malloc((n > 0 ? n : 1) * sizeof(long long));
    if (!arr) { *out_n = 0; return NULL; }
    n = 0;
    for (int i = 0; i < g_num_rows; i++) {
        long long ts = g_rows[i].timestamp;
        if (n == 0 || ts != arr[n - 1]) arr[n++] = ts;
    }
    *out_n = n;
    return arr;
}

/* ── Collect trades ─────────────────────────────────────────────────────── */

static void collect_trades(long long **out_ts, double **out_prices,
                            uint8_t **out_sides, double **out_vols, int *out_n) {
    int n = 0;
    for (int i = 0; i < g_num_rows; i++) {
        if (g_rows[i].event_type != EVT_TRADE) continue;
        if (g_rows[i].side > 1) continue;
        n++;
    }
    int sz = n > 0 ? n : 1;
    long long *ts  = (long long *)malloc(sz * sizeof(long long));
    double    *pr  = (double *)   malloc(sz * sizeof(double));
    double    *vl  = (double *)   malloc(sz * sizeof(double));
    uint8_t   *sd  = (uint8_t *)  malloc(sz * sizeof(uint8_t));
    n = 0;
    for (int i = 0; i < g_num_rows; i++) {
        if (g_rows[i].event_type != EVT_TRADE) continue;
        if (g_rows[i].side > 1) continue;
        ts[n] = g_rows[i].timestamp;
        pr[n] = g_rows[i].price;
        vl[n] = g_rows[i].quantity;
        sd[n] = g_rows[i].side;
        n++;
    }
    *out_ts = ts; *out_prices = pr; *out_sides = sd; *out_vols = vl;
    *out_n = n;
}

/* ── Checkpoint + delta data (built once, reused) ───────────────────────── */

typedef struct {
    int        n_ckpts;
    long long *ckpt_ts;
    int       *bid_offs;
    int       *ask_offs;
    double    *ckpt_bids;
    double    *ckpt_asks;
    int        total_bid;
    int        total_ask;

    int        n_deltas;
    long long *dl_ts;
    uint8_t   *dl_side;
    double    *dl_price;
    double    *dl_qty;

    int        anchor_idx;
} CkptDelta;

static void ckpt_delta_free(CkptDelta *cd) {
    free(cd->ckpt_ts);
    free(cd->bid_offs); free(cd->ask_offs);
    free(cd->ckpt_bids); free(cd->ckpt_asks);
    free(cd->dl_ts); free(cd->dl_side); free(cd->dl_price); free(cd->dl_qty);
}

static int build_ckpt_delta(CkptDelta *cd, const long long *all_ts, int n_ts,
                             int start_i, int end_i) {
    memset(cd, 0, sizeof(*cd));
    cd->anchor_idx = -1;

    /* Count unique book_level timestamps using a temp last-seen ts */
    long long last_bl_ts = 0;
    for (int i = start_i; i < end_i; i++) {
        if (g_rows[i].event_type != EVT_BOOK_LEVEL) continue;
        long long ts = g_rows[i].timestamp;
        if (cd->n_ckpts == 0 || ts != last_bl_ts) {
            cd->n_ckpts++;
            last_bl_ts = ts;
        }
    }
    cd->ckpt_ts = (long long *)malloc((cd->n_ckpts > 0 ? cd->n_ckpts : 1) * sizeof(long long));
    if (!cd->ckpt_ts) goto fail;
    cd->n_ckpts = 0;
    for (int i = start_i; i < end_i; i++) {
        if (g_rows[i].event_type != EVT_BOOK_LEVEL) continue;
        long long ts = g_rows[i].timestamp;
        if (cd->n_ckpts == 0 || cd->ckpt_ts[cd->n_ckpts - 1] != ts)
            cd->ckpt_ts[cd->n_ckpts++] = ts;
    }

    /* Build flat checkpoint bid/ask arrays */
    cd->bid_offs = (int *)malloc((cd->n_ckpts + 1) * sizeof(int));
    cd->ask_offs = (int *)malloc((cd->n_ckpts + 1) * sizeof(int));
    cd->bid_offs[0] = cd->ask_offs[0] = 0;

    int row_idx = start_i;
    for (int k = 0; k < cd->n_ckpts; k++) {
        long long ts = cd->ckpt_ts[k];
        int n_bid = 0, n_ask = 0;
        while (row_idx < end_i && g_rows[row_idx].timestamp == ts
               && g_rows[row_idx].event_type == EVT_BOOK_LEVEL) {
            if (g_rows[row_idx].side == 0) n_bid++;
            else if (g_rows[row_idx].side == 1) n_ask++;
            row_idx++;
        }
        cd->bid_offs[k + 1] = cd->bid_offs[k] + n_bid;
        cd->ask_offs[k + 1] = cd->ask_offs[k] + n_ask;
    }

    cd->total_bid = cd->bid_offs[cd->n_ckpts];
    cd->total_ask = cd->ask_offs[cd->n_ckpts];
    cd->ckpt_bids = (double *)malloc((cd->total_bid > 0 ? cd->total_bid : 1) * 2 * sizeof(double));
    cd->ckpt_asks = (double *)malloc((cd->total_ask > 0 ? cd->total_ask : 1) * 2 * sizeof(double));

    row_idx = start_i;
    for (int k = 0; k < cd->n_ckpts; k++) {
        long long ts = cd->ckpt_ts[k];
        int b_pos = cd->bid_offs[k], a_pos = cd->ask_offs[k];
        while (row_idx < end_i && g_rows[row_idx].timestamp == ts
               && g_rows[row_idx].event_type == EVT_BOOK_LEVEL) {
            Row *r = &g_rows[row_idx];
            if (r->side == 0 && b_pos < cd->bid_offs[k + 1]) {
                cd->ckpt_bids[b_pos * 2]     = r->price;
                cd->ckpt_bids[b_pos * 2 + 1] = r->quantity;
                b_pos++;
            } else if (r->side == 1 && a_pos < cd->ask_offs[k + 1]) {
                cd->ckpt_asks[a_pos * 2]     = r->price;
                cd->ckpt_asks[a_pos * 2 + 1] = r->quantity;
                a_pos++;
            }
            row_idx++;
        }
        side_sort_desc(&cd->ckpt_bids[cd->bid_offs[k] * 2], cd->bid_offs[k + 1] - cd->bid_offs[k]);
        side_sort_asc(&cd->ckpt_asks[cd->ask_offs[k] * 2], cd->ask_offs[k + 1] - cd->ask_offs[k]);
    }

    /* Build delta log from book_update rows */
    for (int i = start_i; i < end_i; i++)
        if (g_rows[i].event_type == EVT_BOOK_UPDATE) cd->n_deltas++;

    int nd = cd->n_deltas > 0 ? cd->n_deltas : 1;
    cd->dl_ts    = (long long *)malloc(nd * sizeof(long long));
    cd->dl_side  = (uint8_t *)  malloc(nd * sizeof(uint8_t));
    cd->dl_price = (double *)   malloc(nd * sizeof(double));
    cd->dl_qty   = (double *)   malloc(nd * sizeof(double));

    cd->n_deltas = 0;
    for (int i = start_i; i < end_i; i++) {
        if (g_rows[i].event_type != EVT_BOOK_UPDATE) continue;
        cd->dl_ts[cd->n_deltas]    = g_rows[i].timestamp;
        cd->dl_side[cd->n_deltas]  = g_rows[i].side;
        cd->dl_price[cd->n_deltas] = g_rows[i].price;
        cd->dl_qty[cd->n_deltas]   = g_rows[i].quantity;
        cd->n_deltas++;
    }

    /* Find anchor checkpoint index for all_ts[0] */
    if (cd->n_ckpts > 0) {
        int lo = 0, hi = cd->n_ckpts;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (cd->ckpt_ts[mid] <= all_ts[0]) { cd->anchor_idx = mid; lo = mid + 1; }
            else hi = mid;
        }
    }
    return 0;

fail:
    ckpt_delta_free(cd);
    return -1;
}

/* ── Compute realized volatility ────────────────────────────────────────── */

static double compute_realized_vol_slice(const double *prices, int n) {
    if (n < 2) return NAN_D;
    double sum_sq = 0.0;
    for (int i = 1; i < n; i++) {
        if (prices[i - 1] <= 0 || prices[i] <= 0) continue;
        double lr = log(prices[i] / prices[i - 1]);
        sum_sq += lr * lr;
    }
    return sqrt(sum_sq);
}

static void compute_realized_vol_rolling(
    const long long *all_ts, int n_ts,
    const long long *trade_ts, const double *trade_prices, int n_trades,
    long long window_ns,
    double *out_rv
) {
    if (n_trades == 0) {
        for (int i = 0; i < n_ts; i++) out_rv[i] = NAN_D;
        return;
    }

    /* Two-pointer sliding window over trades */
    int left = 0;
    for (int i = 0; i < n_ts; i++) {
        long long ts = all_ts[i];
        long long lo = ts - window_ns;

        /* Advance left pointer */
        while (left < n_trades && trade_ts[left] < lo) left++;

        /* Count trades in [lo, ts] */
        int right = left;
        while (right < n_trades && trade_ts[right] <= ts) right++;

        int n = right - left;
        if (n < 2) {
            out_rv[i] = NAN_D;
        } else {
            out_rv[i] = compute_realized_vol_slice(&trade_prices[left], n);
        }

        /* Move right pointer along - will be set correctly next iteration */
    }
}

/* ── Compute OHLC bars ──────────────────────────────────────────────────── */

static void compute_ohlc(
    const long long *trade_ts, const double *trade_prices,
    const double *trade_volumes, int n_trades,
    long long period_ns,
    long long *out_ohlc_ts, double *out_open, double *out_high,
    double *out_low, double *out_close, double *out_volume,
    int *out_n_bars, int max_bars
) {
    *out_n_bars = 0;
    if (n_trades == 0) return;

    /* Bucket trades by (timestamp / period_ns) */
    /* First pass: find the bar range */
    long long first_bar = trade_ts[0] / period_ns;
    long long last_bar  = trade_ts[n_trades - 1] / period_ns;

    for (long long bar = first_bar; bar <= last_bar; bar++) {
        long long bar_start = bar * period_ns;
        long long bar_end   = bar_start + period_ns;

        int first = -1, last = -1;
        double high_v = -1e100, low_v = 1e100;
        double vol = 0.0;
        int count = 0;

        for (int j = 0; j < n_trades; j++) {
            if (trade_ts[j] < bar_start) continue;
            if (trade_ts[j] >= bar_end) break;
            if (first < 0) first = j;
            last = j;
            double p = trade_prices[j];
            if (p > high_v) high_v = p;
            if (p < low_v) low_v = p;
            vol += trade_volumes[j];
            count++;
        }

        if (count > 0 && *out_n_bars < max_bars) {
            out_ohlc_ts[*out_n_bars] = bar_start;
            out_open[*out_n_bars]    = trade_prices[first];
            out_high[*out_n_bars]    = high_v;
            out_low[*out_n_bars]     = low_v;
            out_close[*out_n_bars]   = trade_prices[last];
            out_volume[*out_n_bars]  = vol;
            (*out_n_bars)++;
        }
    }
}

/* ── asof-join OHLC bars onto LOB timestamps ────────────────────────────── */

static void asof_join_ohlc(
    const long long *lob_ts, int n_lob,
    const long long *ohlc_ts, const double *ohlc_open,
    const double *ohlc_high, const double *ohlc_low,
    const double *ohlc_close, const double *ohlc_volume,
    int n_ohlc,
    double *out_open, double *out_high, double *out_low,
    double *out_close, double *out_volume
) {
    int bar_idx = 0;
    for (int i = 0; i < n_lob; i++) {
        long long ts = lob_ts[i];

        /* Find last bar with bar_start <= ts */
        while (bar_idx + 1 < n_ohlc && ohlc_ts[bar_idx + 1] <= ts)
            bar_idx++;

        if (bar_idx < n_ohlc && ohlc_ts[bar_idx] <= ts) {
            out_open[i]   = ohlc_open[bar_idx];
            out_high[i]   = ohlc_high[bar_idx];
            out_low[i]    = ohlc_low[bar_idx];
            out_close[i]  = ohlc_close[bar_idx];
            out_volume[i] = ohlc_volume[bar_idx];
        } else {
            out_open[i]   = NAN_D;
            out_high[i]   = NAN_D;
            out_low[i]    = NAN_D;
            out_close[i]  = NAN_D;
            out_volume[i] = 0.0;
        }
    }
}

/* ── Compute order/ trade rates (short window rolling) ───────────────────── */

typedef struct {
    long long ts;      /* end timestamp of this transition */
    double duration;   /* seconds from previous LOB timestamp */
    double bid_arr, bid_can, ask_arr, ask_can;  /* volumes */
    int    n_trades;   /* trades in this transition interval */
    double trd_vol;    /* total trade volume */
    int    n_buy, n_sell;  /* buy/sell aggressor trades */
} OTrans;

static void compute_order_rates_rolling(
    const long long *all_ts, int n_ts,
    const int *csr_bid_offs, const double *csr_bid_data,
    const int *csr_ask_offs, const double *csr_ask_data,
    const long long *trade_ts, const uint8_t *trade_sides, int n_trades,
    long long window_ns,
    double *out_ord_arr, double *out_ord_can,
    double *out_bid_arr, double *out_ask_arr,
    double *out_bid_can, double *out_ask_can,
    double *out_trd_freq, double *out_buy_freq, double *out_sell_freq)
{
    double nan = NAN_D;
    for (int i = 0; i < n_ts; i++) {
        out_ord_arr[i] = out_ord_can[i] = nan;
        out_bid_arr[i] = out_ask_arr[i] = nan;
        out_bid_can[i] = out_ask_can[i] = nan;
        out_trd_freq[i] = out_buy_freq[i] = out_sell_freq[i] = nan;
    }
    if (n_ts < 2) return;

    /* Build per-transition records from CSR state diffs */
    int n_ot = n_ts;
    OTrans *ot = (OTrans *)calloc(n_ot, sizeof(OTrans));
    if (!ot) return;

    int ti = 0;  /* trade index */
    for (int i = 1; i < n_ts; i++) {
        int pb0 = csr_bid_offs[i-1], pb1 = csr_bid_offs[i];
        int pa0 = csr_ask_offs[i-1], pa1 = csr_ask_offs[i];
        int cb0 = csr_bid_offs[i],   cb1 = csr_bid_offs[i+1];
        int ca0 = csr_ask_offs[i],   ca1 = csr_ask_offs[i+1];

        double bid_arr = 0, bid_can = 0, ask_arr = 0, ask_can = 0;

        /* Bid merge */
        int pi = pb0, ci = cb0;
        while (pi < pb1 && ci < cb1) {
            double pp = csr_bid_data[pi*2], pq = csr_bid_data[pi*2+1];
            double cp = csr_bid_data[ci*2], cq = csr_bid_data[ci*2+1];
            if (cp > pp)      { bid_arr += cq; ci++; }
            else if (cp < pp) { bid_can += pq; pi++; }
            else              { if (cq > pq) bid_arr += cq - pq;
                                else if (cq < pq) bid_can += pq - cq;
                                pi++; ci++; }
        }
        while (ci < cb1) { bid_arr += csr_bid_data[ci*2+1]; ci++; }
        while (pi < pb1) { bid_can += csr_bid_data[pi*2+1]; pi++; }

        /* Ask merge */
        pi = pa0; ci = ca0;
        while (pi < pa1 && ci < ca1) {
            double pp = csr_ask_data[pi*2], pq = csr_ask_data[pi*2+1];
            double cp = csr_ask_data[ci*2], cq = csr_ask_data[ci*2+1];
            if (cp < pp)      { ask_arr += cq; ci++; }
            else if (cp > pp) { ask_can += pq; pi++; }
            else              { if (cq > pq) ask_arr += cq - pq;
                                else if (cq < pq) ask_can += pq - cq;
                                pi++; ci++; }
        }
        while (ci < ca1) { ask_arr += csr_ask_data[ci*2+1]; ci++; }
        while (pi < pa1) { ask_can += csr_ask_data[pi*2+1]; pi++; }

        long long dur_ns = all_ts[i] - all_ts[i-1];
        ot[i].ts       = all_ts[i];
        ot[i].duration = (double)dur_ns / SEC_TO_NS;
        ot[i].bid_arr  = bid_arr;  ot[i].bid_can = bid_can;
        ot[i].ask_arr  = ask_arr;  ot[i].ask_can = ask_can;

        /* Trades in this interval */
        while (ti < n_trades && trade_ts[ti] <= all_ts[i-1]) ti++;
        while (ti < n_trades && trade_ts[ti] <= all_ts[i]) {
            ot[i].n_trades++;
            ot[i].trd_vol += 1;  /* counting trades, not volume */
            if (trade_sides[ti] == 0) ot[i].n_buy++;
            else                      ot[i].n_sell++;
            ti++;
        }
    }

    /* Sliding window over transitions - incremental two-pointer */
    long long w_ns = window_ns;
    int left = 1;
    double sum_dur = 0, sum_ba = 0, sum_bc = 0, sum_aa = 0, sum_ac = 0;
    double sum_trd = 0, sum_buy = 0, sum_sell = 0;
    int next = 1;  /* next transition to add */

    for (int qi = 0; qi < n_ts; qi++) {
        long long ts = all_ts[qi];
        long long lo = ts - w_ns;

        /* Remove transitions whose end time < lo */
        while (left < n_ts && ot[left].ts <= lo) {
            sum_dur -= ot[left].duration;
            sum_ba  -= ot[left].bid_arr;  sum_bc -= ot[left].bid_can;
            sum_aa  -= ot[left].ask_arr;  sum_ac -= ot[left].ask_can;
            sum_trd -= ot[left].n_trades;
            sum_buy -= ot[left].n_buy;    sum_sell -= ot[left].n_sell;
            left++;
        }

        /* Add transitions up to ts */
        while (next < n_ts && ot[next].ts <= ts) {
            if (all_ts[next-1] < lo) {
                /* Left-clipped: scale proportionally */
                long long overlap = ot[next].ts - lo;
                double frac = (double)overlap / (ot[next].ts - all_ts[next-1]);
                if (frac > 0 && frac <= 1) {
                    sum_dur += ot[next].duration * frac;
                    sum_ba  += ot[next].bid_arr * frac; sum_bc += ot[next].bid_can * frac;
                    sum_aa  += ot[next].ask_arr * frac; sum_ac += ot[next].ask_can * frac;
                    sum_trd += ot[next].n_trades * frac;
                    sum_buy += ot[next].n_buy   * frac;
                    sum_sell+= ot[next].n_sell  * frac;
                }
            } else {
                sum_dur += ot[next].duration;
                sum_ba  += ot[next].bid_arr;  sum_bc += ot[next].bid_can;
                sum_aa  += ot[next].ask_arr;  sum_ac += ot[next].ask_can;
                sum_trd += ot[next].n_trades;
                sum_buy += ot[next].n_buy;    sum_sell += ot[next].n_sell;
            }
            next++;
        }

        if (sum_dur > 0) {
            out_ord_arr[qi]  = (sum_ba + sum_aa) / sum_dur;
            out_ord_can[qi]  = (sum_bc + sum_ac) / sum_dur;
            out_bid_arr[qi]  = sum_ba / sum_dur;
            out_ask_arr[qi]  = sum_aa / sum_dur;
            out_bid_can[qi]  = sum_bc / sum_dur;
            out_ask_can[qi]  = sum_ac / sum_dur;
            out_trd_freq[qi] = sum_trd / sum_dur;
            out_buy_freq[qi] = sum_buy / sum_dur;
            out_sell_freq[qi]= sum_sell / sum_dur;
        }
    }
    free(ot);
}

/* ── Compute VPIN ───────────────────────────────────────────────────────── */

static void compute_vpin_rolling(
    const long long *all_ts, int n_ts,
    const long long *trade_ts, const double *trade_prices,
    const uint8_t *trade_sides, const double *trade_vols, int n_trades,
    long long window_ns,
    double *out_vpin
) {
    if (n_trades == 0) {
        for (int i = 0; i < n_ts; i++) out_vpin[i] = NAN_D;
        return;
    }

    double total_vol = 0;
    for (int i = 0; i < n_trades; i++) total_vol += trade_vols[i];
    double bucket_size = total_vol / 50.0;
    if (bucket_size <= 0) {
        for (int i = 0; i < n_ts; i++) out_vpin[i] = NAN_D;
        return;
    }

    /* Two-pointer sliding window */
    int left = 0;
    for (int qi = 0; qi < n_ts; qi++) {
        long long ts = all_ts[qi];
        long long lo = ts - window_ns;
        while (left < n_trades && trade_ts[left] < lo) left++;

        /* Collect and bucket trades in [lo, ts] */
        double buy_vol = 0, sell_vol = 0, remaining = bucket_size;
        int n_buckets = 0;
        double total_imbalance = 0;

        int j = left;
        while (j < n_trades && trade_ts[j] <= ts) {
            double vol = trade_vols[j];
            int side = trade_sides[j];
            while (vol > 0) {
                double fill = (vol < remaining) ? vol : remaining;
                if (side == 0) buy_vol  += fill;
                else           sell_vol += fill;
                vol       -= fill;
                remaining -= fill;
                if (remaining < 1e-12) {
                    total_imbalance += fabs(buy_vol - sell_vol);
                    buy_vol = sell_vol = 0;
                    remaining = bucket_size;
                    n_buckets++;
                }
            }
            j++;
        }
        out_vpin[qi] = (n_buckets > 0) ? total_imbalance / (n_buckets * bucket_size) : NAN_D;
    }
}

/* ── Compute Kyle's lambda ──────────────────────────────────────────────── */

static double ols_slope(const double *x, const double *y, int n) {
    if (n < 2) return NAN_D;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx  += x[i]; sy  += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    double denom = n * sxx - sx * sx;
    if (denom == 0) return NAN_D;
    return (n * sxy - sx * sy) / denom;
}

static void compute_kyle_lambda_rolling(
    const long long *all_ts, int n_ts,
    const long long *trade_ts, const double *trade_prices,
    const uint8_t *trade_sides, const double *trade_vols, int n_trades,
    const double *mid_arr,
    long long window_ns,
    double *out_kl
) {
    for (int qi = 0; qi < n_ts; qi++) out_kl[qi] = NAN_D;
    if (n_trades < 2 || n_ts < 2) return;

    /* Precompute per-interval signed volume and Δmidprice */
    double *sv_arr = (double *)calloc(n_ts, sizeof(double));
    double *dp_arr = (double *)calloc(n_ts, sizeof(double));
    if (!sv_arr || !dp_arr) { free(sv_arr); free(dp_arr); return; }

    int ti = 0;
    for (int i = 1; i < n_ts; i++) {
        double m1 = mid_arr[i-1], m2 = mid_arr[i];
        dp_arr[i] = (!isnan(m1) && !isnan(m2)) ? (m2 - m1) : NAN_D;
        while (ti < n_trades && trade_ts[ti] <= all_ts[i-1]) ti++;
        double sv = 0;
        int tk = ti;
        while (tk < n_trades && trade_ts[tk] <= all_ts[i]) {
            sv += (trade_sides[tk] == 0) ? trade_vols[tk] : -trade_vols[tk];
            tk++;
        }
        sv_arr[i] = sv;
    }

    /* Sliding window over intervals (add/remove incrementally) */
    int left = 1, added_to = 0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;

    for (int qi = 0; qi < n_ts; qi++) {
        long long ts = all_ts[qi];
        long long lo = ts - window_ns;

        /* Remove intervals that fell out */
        while (left < n_ts && all_ts[left] <= lo) {
            if (n > 0 && sv_arr[left] != 0 && !isnan(dp_arr[left])) {
                double x = sv_arr[left], y = dp_arr[left];
                sx -= x; sy -= y; sxx -= x*x; sxy -= x*y; n--;
            }
            if (left > added_to) added_to = left - 1;
            left++;
        }

        /* Add new intervals up to ts */
        int start = (added_to >= left) ? added_to + 1 : left;
        for (int i = start; i < n_ts && all_ts[i] <= ts; i++) {
            if (sv_arr[i] != 0 && !isnan(dp_arr[i])) {
                double x = sv_arr[i], y = dp_arr[i];
                sx += x; sy += y; sxx += x*x; sxy += x*y; n++;
            }
            added_to = i;
        }

        if (n >= 2) {
            double denom = n * sxx - sx * sx;
            out_kl[qi] = (denom != 0) ? (n * sxy - sx * sy) / denom : NAN_D;
        }
    }
    free(sv_arr); free(dp_arr);
}

/* ── Compute order flow imbalance (ofi) per timestamp ────────────────────── */

static void compute_ofi(
    const long long *all_ts, int n_ts,
    const int *csr_bid_offs, const double *csr_bid_data,
    const int *csr_ask_offs, const double *csr_ask_data,
    double *out_ofi
) {
    out_ofi[0] = 0;
    if (n_ts < 2) return;

    for (int i = 1; i < n_ts; i++) {
        double bid_arr = 0, ask_arr = 0;
        int pb0 = csr_bid_offs[i-1], pb1 = csr_bid_offs[i];
        int pa0 = csr_ask_offs[i-1], pa1 = csr_ask_offs[i];
        int cb0 = csr_bid_offs[i],   cb1 = csr_bid_offs[i+1];
        int ca0 = csr_ask_offs[i],   ca1 = csr_ask_offs[i+1];

        /* Bid side: compare current (i) vs previous (i-1) */
        int pi = pb0, ci = cb0;
        while (pi < pb1 && ci < cb1) {
            double pp = csr_bid_data[pi*2], pq = csr_bid_data[pi*2+1];
            double cp = csr_bid_data[ci*2], cq = csr_bid_data[ci*2+1];
            if (cp > pp) {        /* new level in current only */
                bid_arr += cq; ci++;
            } else if (cp < pp) { /* level removed */
                pi++;
            } else {              /* same level */
                if (cq > pq) bid_arr += cq - pq;
                pi++; ci++;
            }
        }
        while (ci < cb1) { bid_arr += csr_bid_data[ci*2+1]; ci++; }

        /* Ask side: ascending order, same merge logic */
        pi = pa0; ci = ca0;
        while (pi < pa1 && ci < ca1) {
            double pp = csr_ask_data[pi*2], pq = csr_ask_data[pi*2+1];
            double cp = csr_ask_data[ci*2], cq = csr_ask_data[ci*2+1];
            if (cp < pp) {
                ask_arr += cq; ci++;
            } else if (cp > pp) {
                pi++;
            } else {
                if (cq > pq) ask_arr += cq - pq;
                pi++; ci++;
            }
        }
        while (ci < ca1) { ask_arr += csr_ask_data[ci*2+1]; ci++; }

        out_ofi[i] = bid_arr - ask_arr;
    }
}

/* ── Compute Hawkes ─────────────────────────────────────────────────────── */

static void compute_hawkes_rolling(
    const long long *all_ts, int n_ts,
    const long long *trade_ts, int n_trades,
    long long window_ns,
    double *out_mu, double *out_alpha, double *out_beta, double *out_br
) {
    double nan = NAN_D;
    for (int i = 0; i < n_ts; i++)
        out_mu[i] = out_alpha[i] = out_beta[i] = out_br[i] = nan;

    if (n_trades < 3) return;

    double *ts_sec = (double *)malloc(n_trades * sizeof(double));
    double *dt     = (double *)malloc(n_trades * sizeof(double));
    if (!ts_sec || !dt) { free(ts_sec); free(dt); return; }

    int left = 0;
    for (int qi = 0; qi < n_ts; qi++) {
        long long ts = all_ts[qi];
        long long lo = ts - window_ns;
        while (left < n_trades && trade_ts[left] < lo) left++;

        /* Collect trades in window, convert to seconds */
        int nw = 0;
        for (int j = left; j < n_trades && trade_ts[j] <= ts; j++) {
            ts_sec[nw] = (double)(trade_ts[j] - trade_ts[left]) / SEC_TO_NS;
            dt[nw] = (nw > 0) ? ts_sec[nw] - ts_sec[nw-1] : 0;
            nw++;
        }
        if (nw >= 3) {
            double mu, alpha, beta;
            if (hawkes_mle_fit(ts_sec, dt, nw, &mu, &alpha, &beta) == 0) {
                out_mu[qi]    = mu;
                out_alpha[qi] = alpha;
                out_beta[qi]  = beta;
                out_br[qi]    = (beta > 0) ? alpha / beta : nan;
            }
        }
    }
    free(ts_sec); free(dt);
}

/* ── Output ─────────────────────────────────────────────────────────────── */

static void write_parquet(const char *path,
                           const long long *timestamps,
                           double *spread, double *bid_arr, double *ask_arr,
                           double *midprice_arr, double *bidq_arr, double *askq_arr,
                           double *vw_mid_arr, double *vi_arr, double *ofi_arr,
                           double *rv_arr,
                           double *gueant_A_ask, double *gueant_k_ask,
                           double *gueant_A_bid, double *gueant_k_bid,
                           double *gA_ask_5, double *gk_ask_5,
                           double *gA_bid_5, double *gk_bid_5,
                           double *hw_mu, double *hw_al, double *hw_be, double *hw_br,
                           double *hw_mu5, double *hw_al5, double *hw_be5, double *hw_br5,
                           double *vpin_arr, double *kl_arr,
                           double *ord_arr, double *ord_can,
                           double *bid_arr_r, double *ask_arr_r,
                           double *bid_can_r, double *ask_can_r,
                           double *trd_freq, double *buy_freq, double *sell_freq,
                           double *ohlc_o, double *ohlc_h, double *ohlc_l,
                           double *ohlc_c, double *ohlc_v,
                           int n) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t *schema = carquet_schema_create(&err);
    if (!schema) { fprintf(stderr, "schema: %s\n", err.message); return; }

    #define ADD(c, name) carquet_schema_add_column(schema, name, CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_OPTIONAL, 0, 0)
    carquet_schema_add_column(schema, "timestamp",              CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    ADD( 1, "spread");              ADD( 2, "bid");
    ADD( 3, "ask");                 ADD( 4, "midprice");
    ADD( 5, "bidq");                ADD( 6, "askq");
    ADD( 7, "vw_midprice");         ADD( 8, "vi");
    ADD( 9, "ofi");                 ADD(10, "realized_vol");
    ADD(11, "gueant_A_ask");        ADD(12, "gueant_k_ask");
    ADD(13, "gueant_A_bid");        ADD(14, "gueant_k_bid");
    ADD(15, "gueant_A_ask_5s");     ADD(16, "gueant_k_ask_5s");
    ADD(17, "gueant_A_bid_5s");     ADD(18, "gueant_k_bid_5s");
    ADD(19, "hawkes_mu");           ADD(20, "hawkes_alpha");
    ADD(21, "hawkes_beta");         ADD(22, "hawkes_branching_ratio");
    ADD(23, "hawkes_mu_5s");        ADD(24, "hawkes_alpha_5s");
    ADD(25, "hawkes_beta_5s");      ADD(26, "hawkes_branching_ratio_5s");
    ADD(27, "vpin");                ADD(28, "kyle_lambda");
    ADD(29, "order_arrival_rate_5s"); ADD(30, "order_cancel_rate_5s");
    ADD(31, "bid_order_arrival_rate_5s"); ADD(32, "ask_order_arrival_rate_5s");
    ADD(33, "bid_order_cancel_rate_5s");  ADD(34, "ask_order_cancel_rate_5s");
    ADD(35, "trade_freq_5s");        ADD(36, "ask_trade_freq_5s");
    ADD(37, "bid_trade_freq_5s");
    ADD(38, "ohlc_open");           ADD(39, "ohlc_high");
    ADD(40, "ohlc_low");            ADD(41, "ohlc_close");
    ADD(42, "ohlc_volume");
    #undef ADD

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_SNAPPY;
    carquet_writer_t *w = carquet_writer_create(path, schema, &opts, &err);
    if (!w) { fprintf(stderr, "writer: %s\n", err.message); carquet_schema_free(schema); return; }

    int16_t *def_levels = (int16_t *)malloc(n * sizeof(int16_t));
    carquet_writer_write_batch(w, 0, timestamps, n, NULL, NULL);

    double *cols[] = {spread, bid_arr, ask_arr, midprice_arr, bidq_arr, askq_arr,
                      vw_mid_arr, vi_arr, ofi_arr, rv_arr,
                      gueant_A_ask, gueant_k_ask, gueant_A_bid, gueant_k_bid,
                      gA_ask_5, gk_ask_5, gA_bid_5, gk_bid_5,
                      hw_mu, hw_al, hw_be, hw_br,
                      hw_mu5, hw_al5, hw_be5, hw_br5,
                      vpin_arr, kl_arr,
                      ord_arr, ord_can, bid_arr_r, ask_arr_r,
                      bid_can_r, ask_can_r, trd_freq, buy_freq, sell_freq,
                      ohlc_o, ohlc_h, ohlc_l, ohlc_c, ohlc_v};
    int n_cols = sizeof(cols) / sizeof(cols[0]);
    for (int ci = 0; ci < n_cols; ci++) {
        int sparse_n = 0;
        for (int i = 0; i < n; i++) if (!isnan(cols[ci][i])) sparse_n++;
        double *sparse = (double *)malloc((sparse_n > 0 ? sparse_n : 1) * sizeof(double));
        sparse_n = 0;
        for (int i = 0; i < n; i++) {
            if (!isnan(cols[ci][i])) { sparse[sparse_n++] = cols[ci][i]; def_levels[i] = 1; }
            else def_levels[i] = 0;
        }
        carquet_writer_write_batch(w, ci + 1, sparse, n, sparse_n > 0 ? def_levels : NULL, NULL);
        free(sparse);
    }
    free(def_levels);
    carquet_writer_close(w);
    carquet_schema_free(schema);
}

static void write_csv(const char *path,
                       const long long *timestamps,
                       double *spread, double *bid_arr, double *ask_arr,
                       double *midprice_arr, double *bidq_arr, double *askq_arr,
                       double *vw_mid_arr, double *vi_arr, double *ofi_arr,
                       double *rv_arr,
                       double *gueant_A_ask, double *gueant_k_ask,
                       double *gueant_A_bid, double *gueant_k_bid,
                       double *gA_ask_5, double *gk_ask_5,
                       double *gA_bid_5, double *gk_bid_5,
                       double *hw_mu, double *hw_al, double *hw_be, double *hw_br,
                       double *hw_mu5, double *hw_al5, double *hw_be5, double *hw_br5,
                       double *vpin_arr, double *kl_arr,
                       double *ord_arr, double *ord_can,
                       double *bid_arr_r, double *ask_arr_r,
                       double *bid_can_r, double *ask_can_r,
                       double *trd_freq, double *buy_freq, double *sell_freq,
                       double *ohlc_o, double *ohlc_h, double *ohlc_l,
                       double *ohlc_c, double *ohlc_v,
                       int n) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "timestamp,spread,bid,ask,midprice,bidq,askq,vw_midprice,vi,ofi,"
            "realized_vol,"
            "gueant_A_ask,gueant_k_ask,gueant_A_bid,gueant_k_bid,"
            "gueant_A_ask_5s,gueant_k_ask_5s,gueant_A_bid_5s,gueant_k_bid_5s,"
            "hawkes_mu,hawkes_alpha,hawkes_beta,hawkes_branching_ratio,"
            "hawkes_mu_5s,hawkes_alpha_5s,hawkes_beta_5s,hawkes_branching_ratio_5s,"
            "vpin,kyle_lambda,"
            "order_arrival_rate_5s,order_cancel_rate_5s,"
            "bid_order_arrival_rate_5s,ask_order_arrival_rate_5s,"
            "bid_order_cancel_rate_5s,ask_order_cancel_rate_5s,"
            "trade_freq_5s,ask_trade_freq_5s,bid_trade_freq_5s,"
            "ohlc_open,ohlc_high,ohlc_low,ohlc_close,ohlc_volume\n");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%lld", (long long)timestamps[i]);
        #define E(v) do { fputc(',', f); if (!isnan(v)) fprintf(f, "%.8g", v); } while(0)
        E(spread[i]); E(bid_arr[i]); E(ask_arr[i]); E(midprice_arr[i]);
        E(bidq_arr[i]); E(askq_arr[i]); E(vw_mid_arr[i]); E(vi_arr[i]);
        E(ofi_arr[i]); E(rv_arr[i]);
        E(gueant_A_ask[i]); E(gueant_k_ask[i]); E(gueant_A_bid[i]); E(gueant_k_bid[i]);
        E(gA_ask_5[i]); E(gk_ask_5[i]); E(gA_bid_5[i]); E(gk_bid_5[i]);
        E(hw_mu[i]); E(hw_al[i]); E(hw_be[i]); E(hw_br[i]);
        E(hw_mu5[i]); E(hw_al5[i]); E(hw_be5[i]); E(hw_br5[i]);
        E(vpin_arr[i]); E(kl_arr[i]);
        E(ord_arr[i]); E(ord_can[i]);
        E(bid_arr_r[i]); E(ask_arr_r[i]);
        E(bid_can_r[i]); E(ask_can_r[i]);
        E(trd_freq[i]); E(buy_freq[i]); E(sell_freq[i]);
        E(ohlc_o[i]); E(ohlc_h[i]); E(ohlc_l[i]); E(ohlc_c[i]); E(ohlc_v[i]);
        #undef E
        fputc('\n', f);
    }
    fclose(f);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);  /* line-buffered output */

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.parquet> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -o <output>        Output file [default: <input>_features.parquet]\n");
        fprintf(stderr, "  -w <seconds>       Long rolling window in seconds [default: 60]\n");
        fprintf(stderr, "  --csv              Output CSV instead of parquet\n");
        fprintf(stderr, "  --no-hawkes        Skip Hawkes estimation (slow on large files)\n");
        fprintf(stderr, "  --no-gueant        Skip Gueant estimation\n");
        fprintf(stderr, "  --no-ohlc          Skip OHLC computation\n");
        return 1;
    }

    const char *input_path  = argv[1];
    char output_path[1024];
    snprintf(output_path, sizeof(output_path), "%s_features.parquet", input_path);
    int do_gueant = 1;
    int do_ohlc   = 1;
    int do_hawkes = 1;
    int use_csv   = 0;
    long long window_sec = DEFAULT_WINDOW;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            snprintf(output_path, sizeof(output_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            window_sec = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0) {
            use_csv = 1;
        } else if (strcmp(argv[i], "--no-gueant") == 0) {
            do_gueant = 0;
        } else if (strcmp(argv[i], "--no-hawkes") == 0) {
            do_hawkes = 0;
        } else if (strcmp(argv[i], "--no-ohlc") == 0) {
            do_ohlc = 0;
        }
    }

    long long window_ns = window_sec * SEC_TO_NS;

    printf("\n");
    printf("  input:    %s\n", input_path);
    printf("  output:   %s\n", output_path);
    printf("  windows:  %llds / 5s\n", (long long)window_sec);
    printf("  features: gueant=%s  hawkes=%s  ohlc=%s  format=%s\n",
           do_gueant ? "on" : "off", do_hawkes ? "on" : "off",
           do_ohlc ? "on" : "off", use_csv ? "csv" : "parquet");
    printf("\n");

    /* ── Parse input ───────────────────────────────────────────────── */
    phase_begin();
    if (parse_parquet(input_path) < 0) return 1;
    phase_skip();

    /* Register all output columns (before computation, for the table) */
    col_immediate("timestamp");

    /* ── Collect ALL timestamps ────────────────────────────────────── */
    int n_ts = 0;
    long long *lob_ts = collect_all_timestamps(&n_ts);
    if (!lob_ts) { fprintf(stderr, "Out of memory\n"); return 1; }

    /* ── Allocate feature output arrays ────────────────────────────── */
    double *spread       = (double *)malloc(n_ts * sizeof(double));
    double *bid_arr      = (double *)malloc(n_ts * sizeof(double));
    double *ask_arr      = (double *)malloc(n_ts * sizeof(double));
    double *midprice_arr = (double *)malloc(n_ts * sizeof(double));
    double *bidq_arr     = (double *)malloc(n_ts * sizeof(double));
    double *askq_arr     = (double *)malloc(n_ts * sizeof(double));
    double *vw_mid_arr   = (double *)malloc(n_ts * sizeof(double));
    double *vi_arr       = (double *)malloc(n_ts * sizeof(double));
    double *rv_arr       = (double *)malloc(n_ts * sizeof(double));

    double *gueant_A_ask = (double *)malloc(n_ts * sizeof(double));
    double *gueant_k_ask = (double *)malloc(n_ts * sizeof(double));
    double *gueant_A_bid = (double *)malloc(n_ts * sizeof(double));
    double *gueant_k_bid = (double *)malloc(n_ts * sizeof(double));

    double *ohlc_o = (double *)malloc(n_ts * sizeof(double));
    double *ohlc_h = (double *)malloc(n_ts * sizeof(double));
    double *ohlc_l = (double *)malloc(n_ts * sizeof(double));
    double *ohlc_c = (double *)malloc(n_ts * sizeof(double));
    double *ohlc_v = (double *)malloc(n_ts * sizeof(double));

    /* Short-window gueant */
    double *gA_ask_5  = (double *)malloc(n_ts * sizeof(double));
    double *gk_ask_5  = (double *)malloc(n_ts * sizeof(double));
    double *gA_bid_5  = (double *)malloc(n_ts * sizeof(double));
    double *gk_bid_5  = (double *)malloc(n_ts * sizeof(double));

    /* Hawkes 60s + 5s */
    double *hw_mu  = (double *)malloc(n_ts * sizeof(double));
    double *hw_al  = (double *)malloc(n_ts * sizeof(double));
    double *hw_be  = (double *)malloc(n_ts * sizeof(double));
    double *hw_br  = (double *)malloc(n_ts * sizeof(double));
    double *hw_mu5 = (double *)malloc(n_ts * sizeof(double));
    double *hw_al5 = (double *)malloc(n_ts * sizeof(double));
    double *hw_be5 = (double *)malloc(n_ts * sizeof(double));
    double *hw_br5 = (double *)malloc(n_ts * sizeof(double));

    /* VPIN, ofi, kyle_lambda */
    double *vpin_arr = (double *)malloc(n_ts * sizeof(double));
    double *ofi_arr  = (double *)malloc(n_ts * sizeof(double));
    double *kl_arr   = (double *)malloc(n_ts * sizeof(double));

    /* Short-window order/trade rates */
    double *ord_arr  = (double *)malloc(n_ts * sizeof(double));
    double *ord_can  = (double *)malloc(n_ts * sizeof(double));
    double *bid_arr_r = (double *)malloc(n_ts * sizeof(double));
    double *ask_arr_r = (double *)malloc(n_ts * sizeof(double));
    double *bid_can_r = (double *)malloc(n_ts * sizeof(double));
    double *ask_can_r = (double *)malloc(n_ts * sizeof(double));
    double *trd_freq  = (double *)malloc(n_ts * sizeof(double));
    double *buy_freq  = (double *)malloc(n_ts * sizeof(double));
    double *sell_freq = (double *)malloc(n_ts * sizeof(double));

    /* ── Build checkpoint/delta data ──────────────────────────────── */
    CkptDelta cd;
    if (build_ckpt_delta(&cd, lob_ts, n_ts, 0, g_num_rows) < 0) {
        fprintf(stderr, "Failed to build checkpoints\n");
        return 1;
    }

    /* ── Compute LOB snapshot features ─────────────────────────────── */
    phase_begin();
    lobts_seq_extract_best(
        lob_ts, n_ts,
        cd.n_ckpts > 0 ? cd.ckpt_ts : NULL, cd.n_ckpts,
        cd.n_ckpts > 0 ? cd.bid_offs : NULL,
        cd.n_ckpts > 0 && cd.total_bid > 0 ? cd.ckpt_bids : NULL,
        cd.n_ckpts > 0 ? cd.ask_offs : NULL,
        cd.n_ckpts > 0 && cd.total_ask > 0 ? cd.ckpt_asks : NULL,
        cd.n_deltas > 0 ? cd.dl_ts : NULL,
        cd.n_deltas > 0 ? cd.dl_side : NULL,
        cd.n_deltas > 0 ? cd.dl_price : NULL,
        cd.n_deltas > 0 ? cd.dl_qty : NULL,
        cd.n_deltas,
        cd.anchor_idx,
        bid_arr, ask_arr, bidq_arr, askq_arr);

    /* Compute derived features */
    for (int i = 0; i < n_ts; i++) {
        double b = bid_arr[i], a = ask_arr[i], bq = bidq_arr[i], aq = askq_arr[i];
        if (!isnan(b) && !isnan(a)) {
            spread[i]     = a - b;
            midprice_arr[i] = (b + a) / 2.0;
            vw_mid_arr[i] = (!isnan(bq) && !isnan(aq) && (bq + aq) > 0)
                            ? (b * bq + a * aq) / (bq + aq) : NAN_D;
            vi_arr[i] = (!isnan(bq) && !isnan(aq) && (bq + aq) > 0)
                        ? (bq - aq) / (bq + aq) : 0.0;
        } else {
            spread[i]     = NAN_D;
            midprice_arr[i] = NAN_D;
            vw_mid_arr[i] = NAN_D;
            vi_arr[i]     = 0.0;
        }
    }
    phase_end("LOB snapshot");
    col_immediate("spread"); col_immediate("bid"); col_immediate("ask"); col_immediate("midprice"); col_immediate("bidq"); col_immediate("askq"); col_immediate("vw_midprice"); col_immediate("vi");

    /* ── Collect trades ────────────────────────────────────────────── */
    long long *trade_ts     = NULL;
    double    *trade_prices = NULL;
    double    *trade_vols   = NULL;
    uint8_t   *trade_sides  = NULL;
    int n_trades = 0;
    collect_trades(&trade_ts, &trade_prices, &trade_sides, &trade_vols, &n_trades);

    /* ── Compute realized volatility ───────────────────────────────── */
    phase_begin();
    compute_realized_vol_rolling(lob_ts, n_ts, trade_ts, trade_prices,
                                  n_trades, window_ns, rv_arr);
    phase_end("realized_vol"); col_rolling("realized_vol", (int)window_sec);

    /* ── Get full LOB CSR state (needed by Gueant, ofi, order rates) ── */
    int *csr_bid_offs = NULL, *csr_ask_offs = NULL;
    double *csr_bid_data = NULL, *csr_ask_data = NULL;
    int csr_total_bid = 0, csr_total_ask = 0;

    {
        csr_bid_offs = (int *)malloc((n_ts + 1) * sizeof(int));
        csr_ask_offs = (int *)malloc((n_ts + 1) * sizeof(int));
        long long *csr_ts = (long long *)malloc(n_ts * sizeof(long long));

        lobts_iter_seq_states(
            lob_ts, n_ts,
            cd.n_ckpts > 0 ? cd.ckpt_ts : NULL, cd.n_ckpts,
            cd.n_ckpts > 0 ? cd.bid_offs : NULL,
            cd.n_ckpts > 0 && cd.total_bid > 0 ? cd.ckpt_bids : NULL,
            cd.n_ckpts > 0 ? cd.ask_offs : NULL,
            cd.n_ckpts > 0 && cd.total_ask > 0 ? cd.ckpt_asks : NULL,
            cd.n_deltas > 0 ? cd.dl_ts : NULL,
            cd.n_deltas > 0 ? cd.dl_side : NULL,
            cd.n_deltas > 0 ? cd.dl_price : NULL,
            cd.n_deltas > 0 ? cd.dl_qty : NULL,
            cd.n_deltas, cd.anchor_idx,
            csr_ts, csr_bid_offs, csr_ask_offs,
            NULL, NULL, &csr_total_bid, &csr_total_ask);
        free(csr_ts);

        csr_bid_data = (double *)malloc((csr_total_bid > 0 ? csr_total_bid : 1) * 2 * sizeof(double));
        csr_ask_data = (double *)malloc((csr_total_ask > 0 ? csr_total_ask : 1) * 2 * sizeof(double));

        lobts_iter_seq_states(
            lob_ts, n_ts,
            cd.n_ckpts > 0 ? cd.ckpt_ts : NULL, cd.n_ckpts,
            cd.n_ckpts > 0 ? cd.bid_offs : NULL,
            cd.n_ckpts > 0 && cd.total_bid > 0 ? cd.ckpt_bids : NULL,
            cd.n_ckpts > 0 ? cd.ask_offs : NULL,
            cd.n_ckpts > 0 && cd.total_ask > 0 ? cd.ckpt_asks : NULL,
            cd.n_deltas > 0 ? cd.dl_ts : NULL,
            cd.n_deltas > 0 ? cd.dl_side : NULL,
            cd.n_deltas > 0 ? cd.dl_price : NULL,
            cd.n_deltas > 0 ? cd.dl_qty : NULL,
            cd.n_deltas, cd.anchor_idx,
            lob_ts, csr_bid_offs, csr_ask_offs,
            csr_total_bid > 0 ? csr_bid_data : NULL,
            csr_total_ask > 0 ? csr_ask_data : NULL,
            &csr_total_bid, &csr_total_ask);
    }

    /* ── Free checkpoint/delta (no longer needed) ──────────────────── */
    ckpt_delta_free(&cd);

    /* ── Gueant ──────────────────────────────────────────────────── */
    if (do_gueant && n_ts > 0 && n_trades > 0) {
        int thresholds[] = {1, 3, 5, 10};
        int n_buckets = 4;
        long long end_ts = lob_ts[n_ts - 1];
        if (trade_ts[n_trades - 1] > end_ts) end_ts = trade_ts[n_trades - 1];

        phase_begin();
        gueant_rolling_buckets(lob_ts, n_ts, csr_bid_offs, csr_bid_data,
            csr_ask_offs, csr_ask_data, end_ts,
            trade_ts, trade_prices, trade_sides, n_trades,
            0.1, 0, window_ns, thresholds, n_buckets, lob_ts, n_ts, gueant_A_ask, gueant_k_ask);
        col_rolling("gueant_A_ask", (int)window_sec); col_rolling("gueant_k_ask", (int)window_sec);
        gueant_rolling_buckets(lob_ts, n_ts, csr_bid_offs, csr_bid_data,
            csr_ask_offs, csr_ask_data, end_ts,
            trade_ts, trade_prices, trade_sides, n_trades,
            0.1, 1, window_ns, thresholds, n_buckets, lob_ts, n_ts, gueant_A_bid, gueant_k_bid);
        phase_end("gueant 60s (ask+bid)"); col_rolling("gueant_A_bid", (int)window_sec); col_rolling("gueant_k_bid", (int)window_sec);

        phase_begin();
        gueant_rolling_buckets(lob_ts, n_ts, csr_bid_offs, csr_bid_data,
            csr_ask_offs, csr_ask_data, end_ts,
            trade_ts, trade_prices, trade_sides, n_trades,
            0.1, 0, 5 * SEC_TO_NS, thresholds, n_buckets, lob_ts, n_ts, gA_ask_5, gk_ask_5);
        col_rolling("gueant_A_ask_5s", 5); col_rolling("gueant_k_ask_5s", 5);
        gueant_rolling_buckets(lob_ts, n_ts, csr_bid_offs, csr_bid_data,
            csr_ask_offs, csr_ask_data, end_ts,
            trade_ts, trade_prices, trade_sides, n_trades,
            0.1, 1, 5 * SEC_TO_NS, thresholds, n_buckets, lob_ts, n_ts, gA_bid_5, gk_bid_5);
        phase_end("gueant 5s (ask+bid)"); col_rolling("gueant_A_bid_5s", 5); col_rolling("gueant_k_bid_5s", 5);
    } else {
        for (int i = 0; i < n_ts; i++)
            gueant_A_ask[i] = gueant_k_ask[i] = gueant_A_bid[i] = gueant_k_bid[i] =
            gA_ask_5[i] = gk_ask_5[i] = gA_bid_5[i] = gk_bid_5[i] = NAN_D;
    }

    /* ── OFI + order/trade rates ──────────────────────────────────── */
    phase_begin();
    compute_ofi(lob_ts, n_ts, csr_bid_offs, csr_bid_data, csr_ask_offs, csr_ask_data, ofi_arr);
    phase_end("ofi"); col_immediate("ofi");

    phase_begin();
    compute_order_rates_rolling(lob_ts, n_ts, csr_bid_offs, csr_bid_data,
        csr_ask_offs, csr_ask_data, trade_ts, trade_sides, n_trades, 5 * SEC_TO_NS,
        ord_arr, ord_can, bid_arr_r, ask_arr_r, bid_can_r, ask_can_r,
        trd_freq, buy_freq, sell_freq);
    phase_end("order/trade rates");
    col_rolling("order_arrival_rate_5s", 5);
    col_rolling("order_cancel_rate_5s", 5);
    col_rolling("bid_order_arrival_rate_5s", 5);
    col_rolling("ask_order_arrival_rate_5s", 5);
    col_rolling("bid_order_cancel_rate_5s", 5);
    col_rolling("ask_order_cancel_rate_5s", 5);
    col_rolling("trade_freq_5s", 5);
    col_rolling("ask_trade_freq_5s", 5);
    col_rolling("bid_trade_freq_5s", 5);

    free(csr_bid_offs); free(csr_ask_offs); free(csr_bid_data); free(csr_ask_data);

    /* ── Hawkes ───────────────────────────────────────────────────── */
    col_rolling("hawkes_mu", (int)window_sec);
    col_rolling("hawkes_alpha", (int)window_sec);
    col_rolling("hawkes_beta", (int)window_sec);
    col_rolling("hawkes_branching_ratio", (int)window_sec);
    col_rolling("hawkes_mu_5s", 5);
    col_rolling("hawkes_alpha_5s", 5);
    col_rolling("hawkes_beta_5s", 5);
    col_rolling("hawkes_branching_ratio_5s", 5);
    if (do_hawkes) {
        phase_begin();
        compute_hawkes_rolling(lob_ts, n_ts, trade_ts, n_trades, window_ns,
                               hw_mu, hw_al, hw_be, hw_br);
        phase_end("hawkes 60s");
        phase_begin();
        compute_hawkes_rolling(lob_ts, n_ts, trade_ts, n_trades, 5 * SEC_TO_NS,
                               hw_mu5, hw_al5, hw_be5, hw_br5);
        phase_end("hawkes 5s");
    } else {
        for (int i = 0; i < n_ts; i++)
            hw_mu[i] = hw_al[i] = hw_be[i] = hw_br[i] =
            hw_mu5[i] = hw_al5[i] = hw_be5[i] = hw_br5[i] = NAN_D;
    }

    /* ── VPIN + Kyle's lambda ─────────────────────────────────────── */
    phase_begin();
    compute_vpin_rolling(lob_ts, n_ts, trade_ts, trade_prices, trade_sides,
                         trade_vols, n_trades, window_ns, vpin_arr);
    phase_end("vpin"); col_rolling("vpin", (int)window_sec);

    phase_begin();
    compute_kyle_lambda_rolling(lob_ts, n_ts, trade_ts, trade_prices, trade_sides,
                                trade_vols, n_trades, midprice_arr, window_ns, kl_arr);
    phase_end("kyle_lambda"); col_rolling("kyle_lambda", (int)window_sec);

    /* ── OHLC ─────────────────────────────────────────────────────── */
    if (do_ohlc) {
        phase_begin();
        long long *bar_ts = (long long *)malloc(n_ts * sizeof(long long));
        double *bar_o = (double *)malloc(n_ts * sizeof(double));
        double *bar_h = (double *)malloc(n_ts * sizeof(double));
        double *bar_l = (double *)malloc(n_ts * sizeof(double));
        double *bar_c = (double *)malloc(n_ts * sizeof(double));
        double *bar_v = (double *)malloc(n_ts * sizeof(double));
        int n_bars = 0;
        compute_ohlc(trade_ts, trade_prices, trade_vols, n_trades,
                     OHLC_PERIOD_NS, bar_ts, bar_o, bar_h, bar_l, bar_c, bar_v, &n_bars, n_ts);
        asof_join_ohlc(lob_ts, n_ts, bar_ts, bar_o, bar_h, bar_l, bar_c, bar_v,
                       n_bars, ohlc_o, ohlc_h, ohlc_l, ohlc_c, ohlc_v);
        free(bar_ts); free(bar_o); free(bar_h); free(bar_l); free(bar_c); free(bar_v);
        phase_end("ohlc"); col_rolling("ohlc_open", 1); col_rolling("ohlc_high", 1); col_rolling("ohlc_low", 1); col_rolling("ohlc_close", 1); col_rolling("ohlc_volume", 1);
    } else {
        for (int i = 0; i < n_ts; i++)
            ohlc_o[i] = ohlc_h[i] = ohlc_l[i] = ohlc_c[i] = NAN_D, ohlc_v[i] = 0;
    }

    /* ── Write output ─────────────────────────────────────────────── */
    phase_begin();
    if (use_csv) {
        write_csv(output_path, lob_ts,
                  spread, bid_arr, ask_arr, midprice_arr, bidq_arr, askq_arr,
                  vw_mid_arr, vi_arr, ofi_arr, rv_arr,
                  gueant_A_ask, gueant_k_ask, gueant_A_bid, gueant_k_bid,
                  gA_ask_5, gk_ask_5, gA_bid_5, gk_bid_5,
                  hw_mu, hw_al, hw_be, hw_br,
                  hw_mu5, hw_al5, hw_be5, hw_br5,
                  vpin_arr, kl_arr,
                  ord_arr, ord_can, bid_arr_r, ask_arr_r,
                  bid_can_r, ask_can_r, trd_freq, buy_freq, sell_freq,
                  ohlc_o, ohlc_h, ohlc_l, ohlc_c, ohlc_v, n_ts);
    } else {
        write_parquet(output_path, lob_ts,
                      spread, bid_arr, ask_arr, midprice_arr, bidq_arr, askq_arr,
                      vw_mid_arr, vi_arr, ofi_arr, rv_arr,
                      gueant_A_ask, gueant_k_ask, gueant_A_bid, gueant_k_bid,
                      gA_ask_5, gk_ask_5, gA_bid_5, gk_bid_5,
                      hw_mu, hw_al, hw_be, hw_br,
                      hw_mu5, hw_al5, hw_be5, hw_br5,
                      vpin_arr, kl_arr,
                      ord_arr, ord_can, bid_arr_r, ask_arr_r,
                      bid_can_r, ask_can_r, trd_freq, buy_freq, sell_freq,
                      ohlc_o, ohlc_h, ohlc_l, ohlc_c, ohlc_v, n_ts);
    }
    phase_skip();

    printf("\n  output: %s  (%d rows \303\227 43 cols)\n", output_path, n_ts);
    print_timing_table();

    /* Cleanup */
    free(lob_ts);
    free(spread); free(bid_arr); free(ask_arr); free(midprice_arr);
    free(bidq_arr); free(askq_arr); free(vw_mid_arr); free(vi_arr);
    free(rv_arr); free(ofi_arr);
    free(gueant_A_ask); free(gueant_k_ask);
    free(gueant_A_bid); free(gueant_k_bid);
    free(gA_ask_5); free(gk_ask_5); free(gA_bid_5); free(gk_bid_5);
    free(hw_mu); free(hw_al); free(hw_be); free(hw_br);
    free(hw_mu5); free(hw_al5); free(hw_be5); free(hw_br5);
    free(vpin_arr); free(kl_arr);
    free(ord_arr); free(ord_can); free(bid_arr_r); free(ask_arr_r);
    free(bid_can_r); free(ask_can_r); free(trd_freq); free(buy_freq); free(sell_freq);
    free(ohlc_o); free(ohlc_h); free(ohlc_l); free(ohlc_c); free(ohlc_v);
    free(trade_ts); free(trade_prices); free(trade_vols); free(trade_sides);
    free(g_rows);

    return 0;
}
