/* wubu_gru.h — self-contained GRU layer forward (C11).
 *
 * Implements PyTorch nn.GRU(batch_first=True) semantics for a single layer,
 * optional bidirectional. Gate order r,z,n; weights in PyTorch layout:
 *   w_ih: [3*hidden x input]  rows [W_ir; W_iz; W_in]
 *   w_hh: [3*hidden x hidden] rows [W_hr; W_hz; W_hn]
 *   b_ih / b_hh: [3*hidden]
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_GRU_H
#define WUBU_GRU_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WuBuGru WuBuGru;

/* Create a GRU layer. bidirectional=1 doubles the output width. */
WuBuGru *wubu_gru_create(int input_size, int hidden_size, int bidirectional);
void wubu_gru_free(WuBuGru *g);

/* Set weights for one direction. dir: 0 = forward, 1 = backward.
 * Weights are copied. Returns 0 on success. */
int wubu_gru_set(WuBuGru *g, int dir,
                 const float *w_ih, const float *w_hh,
                 const float *b_ih, const float *b_hh);

/* Forward: x[T x input_size] row-major -> out[T x out_size] row-major,
 * out_size = hidden_size * (1 + bidirectional). out may alias x? (no). */
int wubu_gru_forward(WuBuGru *g, const float *x, int T, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GRU_H */
