/* wubu_gru.c — self-contained GRU layer forward (C11).
 *
 * Single-layer PyTorch-compatible GRU (batch_first), optional bidir.
 *
 *   r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
 *   z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
 *   n_t = tanh(W_in x_t + b_in + r_t .* (W_hn h_{t-1} + b_hn))
 *   h_t = (1 - z_t) .* n_t + z_t .* h_{t-1}
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_gru.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float *w_ih;   /* 3h x input */
    float *w_hh;   /* 3h x h */
    float *b_ih;   /* 3h */
    float *b_hh;   /* 3h */
    int input_size;
    int hidden;
} GruDir;

struct WuBuGru {
    GruDir fwd, bwd;
    int input_size;
    int hidden;
    int bidirectional;
};

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

WuBuGru *wubu_gru_create(int input_size, int hidden_size, int bidirectional) {
    if (input_size < 1 || hidden_size < 1) return NULL;
    WuBuGru *g = (WuBuGru *)calloc(1, sizeof(WuBuGru));
    if (!g) return NULL;
    g->input_size = input_size;
    g->hidden = hidden_size;
    g->bidirectional = bidirectional ? 1 : 0;
    g->fwd.input_size = input_size;
    g->fwd.hidden = hidden_size;
    g->bwd.input_size = input_size;
    g->bwd.hidden = hidden_size;
    return g;
}

void wubu_gru_free(WuBuGru *g) {
    if (!g) return;
    free(g->fwd.w_ih); free(g->fwd.w_hh); free(g->fwd.b_ih); free(g->fwd.b_hh);
    free(g->bwd.w_ih); free(g->bwd.w_hh); free(g->bwd.b_ih); free(g->bwd.b_hh);
    free(g);
}

int wubu_gru_set(WuBuGru *g, int dir,
                 const float *w_ih, const float *w_hh,
                 const float *b_ih, const float *b_hh) {
    if (!g || !w_ih || !w_hh || !b_ih || !b_hh) return -1;
    GruDir *d = (dir == 1) ? &g->bwd : &g->fwd;
    int h = d->hidden;
    size_t ih_n = (size_t)3 * h * d->input_size;
    size_t hh_n = (size_t)3 * h * h;
    d->w_ih = (float *)malloc(ih_n * sizeof(float));
    d->w_hh = (float *)malloc(hh_n * sizeof(float));
    d->b_ih = (float *)malloc((size_t)3 * h * sizeof(float));
    d->b_hh = (float *)malloc((size_t)3 * h * sizeof(float));
    if (!d->w_ih || !d->w_hh || !d->b_ih || !d->b_hh) return -1;
    memcpy(d->w_ih, w_ih, ih_n * sizeof(float));
    memcpy(d->w_hh, w_hh, hh_n * sizeof(float));
    memcpy(d->b_ih, b_ih, (size_t)3 * h * sizeof(float));
    memcpy(d->b_hh, b_hh, (size_t)3 * h * sizeof(float));
    return 0;
}

/* Run one direction over x[T x in]; writes h states into hseq[T x hidden]. */
static void gru_dir_run(const GruDir *d, const float *x, int T, float *hseq) {
    const int h = d->hidden;
    const int in = d->input_size;
    float *h_prev = (float *)calloc((size_t)h, sizeof(float));
    float *g = (float *)malloc((size_t)3 * h * sizeof(float));
    if (!h_prev || !g) { free(h_prev); free(g); return; }

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * in;
        /* g = W_ih x_t + b_ih + W_hh h_prev + b_hh — rows are independent */
#pragma omp parallel for schedule(static) if(h >= 64)
        for (int row = 0; row < 3 * h; row++) {
            float acc = d->b_ih[row] + d->b_hh[row];
            const float *wi = d->w_ih + (size_t)row * in;
            const float *wh = d->w_hh + (size_t)row * h;
            for (int i = 0; i < in; i++) acc += wi[i] * xt[i];
            for (int j = 0; j < h; j++) acc += wh[j] * h_prev[j];
            g[row] = acc;
        }
        float *ht = hseq + (size_t)t * h;
        const float *whn = d->w_hh + (size_t)(2 * h) * h; /* W_hn row block */
        for (int j = 0; j < h; j++) {
            float r = sigmoidf(g[j]);
            float z = sigmoidf(g[h + j]);
            /* n = tanh(W_in x + b_in + r * (W_hn h_prev + b_hn)) —
             * the reset gate gates the HIDDEN term, not the input term.
             * g[2h+j] currently holds the un-gated sum; rebuild gated. */
            float hh = d->b_hh[2 * h + j];
            for (int k = 0; k < h; k++) hh += whn[(size_t)j * h + k] * h_prev[k];
            float n = tanhf(g[2 * h + j] - hh + r * hh);
            ht[j] = (1.0f - z) * n + z * h_prev[j];
        }
        memcpy(h_prev, ht, (size_t)h * sizeof(float));
    }
    free(h_prev);
    free(g);
}

int wubu_gru_forward(WuBuGru *g, const float *x, int T, float *out) {
    if (!g || !x || !out || T < 1) return -1;
    const int h = g->hidden;
    float *hf = (float *)malloc((size_t)T * h * sizeof(float));
    float *hb = (g->bidirectional) ? (float *)malloc((size_t)T * h * sizeof(float)) : NULL;
    if (!hf || (g->bidirectional && !hb)) {
        free(hf); free(hb);
        return -1;
    }
    gru_dir_run(&g->fwd, x, T, hf);
    if (g->bidirectional) {
        /* backward: run over reversed time, store reversed */
        float *xr = (float *)malloc((size_t)T * g->input_size * sizeof(float));
        if (!xr) { free(hf); free(hb); return -1; }
        for (int t = 0; t < T; t++)
            memcpy(xr + (size_t)t * g->input_size, x + (size_t)(T - 1 - t) * g->input_size,
                   (size_t)g->input_size * sizeof(float));
        gru_dir_run(&g->bwd, xr, T, hb);
        free(xr);
        for (int t = 0; t < T; t++) {
            float *o = out + (size_t)t * 2 * h;
            memcpy(o, hf + (size_t)t * h, (size_t)h * sizeof(float));
            /* reverse the backward-direction states back to chronological order */
            memcpy(o + h, hb + (size_t)(T - 1 - t) * h, (size_t)h * sizeof(float));
        }
    } else {
        memcpy(out, hf, (size_t)T * h * sizeof(float));
    }
    free(hf);
    free(hb);
    return 0;
}
