#ifndef WUBU_RVC_HUBERT_H
#define WUBU_RVC_HUBERT_H

/* wubu_rvc_hubert.h — Real HuBERT content encoder (C11, zero-dependency).
 *
 * Faithful port of fairseq HubertModel.extract_features for the
 * lj1995 hubert_base.pt used by RVC:
 *   - 7-block conv feature extractor (Conv1d no-bias, GroupNorm dim, GELU;
 *     strides [5,2,2,2,2,2,2], kernels [10,3,3,3,3,2,2])
 *   - layer_norm (512)
 *   - post_extract_proj Linear(512 -> 768)
 *   - pos_conv: Conv1d(768, k=128, pad=64, groups=16) weight_norm
 *     (W[oc,ic,k] = g[k] * v[oc,ic,k]/||v[oc,ic,:]||) + SamePad + GELU
 *   - encoder.layer_norm (pre-norm)
 *   - 12 TransformerSentenceEncoderLayer (post-LN):
 *       x = LN(x + MHA(x)); x = LN(x + fc2(GELU(fc1(x))))
 *   - v1: output layer 9 then final_proj (768 -> 256)
 *   - v2: output layer 12 (768 dim)
 *
 * Weights load from the WUBU flat binary produced by
 * tools/extract_hubert_weights.py. No fairseq, no torch, no Python at
 * runtime.
 *
 * License: WaefreBeorn-UMV3
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* conv feature extractor: 7 blocks, each 512 out channels */
    float *conv_w[7];        /* [out, in, k] */
    float *conv_gn_w[7];     /* GroupNorm gamma (512) */
    float *conv_gn_b[7];     /* GroupNorm beta (512) */
    int    conv_k[7];
    int    conv_s[7];
    int    conv_in[7];

    /* post-extract */
    float *layer_norm_w;     /* 512 */
    float *layer_norm_b;     /* 512 */
    float *post_proj_w;      /* [768, 512] */
    float *post_proj_b;      /* 768 */

    /* pos_conv (weight_norm dim=2: g[k] * v/||v|| over kernel) */
    float *pos_w_g;          /* [1, 1, 128] */
    float *pos_w_v;          /* [768, 48, 128] */
    float *pos_b;            /* 768 */

    /* encoder pre-norm */
    float *enc_ln_w;         /* 768 */
    float *enc_ln_b;         /* 768 */

    /* 12 transformer layers */
    float *attn_q_w[12], *attn_q_b[12];
    float *attn_k_w[12], *attn_k_b[12];
    float *attn_v_w[12], *attn_v_b[12];
    float *attn_o_w[12], *attn_o_b[12];
    float *attn_ln_w[12], *attn_ln_b[12];
    float *fc1_w[12], *fc1_b[12];
    float *fc2_w[12], *fc2_b[12];
    float *fin_ln_w[12], *fin_ln_b[12];

    /* v1 final projection */
    float *final_proj_w;     /* [256, 768] */
    float *final_proj_b;     /* 256 */

    int loaded;
} WuBuHubert;

/* Load weights from WUBU bin. Returns 0 on success. */
int wubu_hubert_load(WuBuHubert *h, const char *bin_path);

/* Free all weight buffers. */
void wubu_hubert_free(WuBuHubert *h);

/* Extract content features from 16kHz mono PCM.
 * pcm: n_samples floats in [-1, 1].
 * version: 1 -> layer 9 + final_proj (256 dim), 2 -> layer 12 (768 dim).
 * feats_out: [n_frames * dim] where dim = 256 or 768.
 * Returns n_frames, or -1 on error. */
int wubu_hubert_extract_real(const WuBuHubert *h,
                             const float *pcm, int n_samples,
                             int version,
                             float *feats_out, int max_feats);

/* Debug: dump intermediate stage activations to outputs/rvc_ref/c_*.npy
 * for C-vs-torch parity diffing. Returns n_frames or -1. */
int wubu_hubert_debug_dump(const WuBuHubert *h,
                           const float *pcm, int n_samples);

/* Conv feature extractor output length for n_samples input. */
int wubu_hubert_output_length(int n_samples);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RVC_HUBERT_H */
