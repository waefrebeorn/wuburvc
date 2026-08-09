/* wubu_vk.h — Vulkan compute accelerator for the WuBuMedia engine.
 *
 * Cross-vendor GPU kernels (NVIDIA/AMD/Intel) via the Vulkan compute API —
 * a C-native alternative to CUDA that needs no vendor SDK. Same math as the
 * CPU kernels (wubu_rvc_real.c conv1d_c) and the CUDA kernels
 * (wubu_rvc_cuda.cu k_conv1d).
 *
 * Opaque struct; no god header. C11 only.
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_VK_H
#define WUBU_VK_H

#include <stddef.h>

typedef struct WuBuVk WuBuVk;   /* opaque */
typedef struct WuBuRVCModel WuBuRVCModel; /* forward decl (wubu_rvc_parity.h) */

/* Create the Vulkan context: instance, first compute-capable physical
 * device, device + compute queue, the conv1d pipeline (SPIR-V embedded),
 * and a growable storage-buffer pool. Returns NULL on failure. */
WuBuVk *wubu_vk_create(void);

/* Tear everything down. */
void wubu_vk_destroy(WuBuVk *vk);

/* conv1d — identical signature/behavior to wubu_rvc_real.c conv1d_c:
 *   in   [in_ch * n]      (col-major: channel-major, n contiguous)
 *   w    [out_ch * in_ch * k]
 *   b    [out_ch] or NULL
 *   out  [out_ch * n_out]
 * n_out = (n + 2*pad - dil*(k-1) - 1)/stride + 1.
 * The buffers are reallocated lazily (host-visible); uploads and the
 * readback are synchronous. Returns 0 on success, -1 on failure. */
int wubu_vk_conv1d(WuBuVk *vk,
                   const float *in, int in_ch, int n,
                   const float *w, const float *b,
                   int out_ch, int k, int stride, int pad, int dil,
                   float *out, int n_out);

/* elt helper: mode 0 a+=b, 1 a=b. */
int wubu_vk_elt(WuBuVk *vk, float *a, const float *b, size_t n, int mode);

/* ConvTranspose1d (gather) — w is [in_ch * out_ch * k]. */
int wubu_vk_convt1d(WuBuVk *vk,
                    const float *in, int in_ch, int n,
                    const float *w, const float *b,
                    int out_ch, int k, int stride, int pad,
                    float *out, int n_out);

/* Full GeneratorNSF on Vulkan — mirrors wubu_generator_nsf_cuda (flow stays
 * on the CPU; conv_pre, cond, sine, upsample blocks + noise convs + MRF,
 * conv_post run on the GPU). Returns the sample count or -1. */
int wubu_vk_generator_nsf(WuBuVk *vk, WuBuRVCModel *model,
                          const float *z, int nF, int inter_channels,
                          const float *nsff0, const float *g,
                          float *out_audio, int max_samples,
                          int inject_noise, int use_snake);

/* ── training backward ops (synchronous, host buffers) ──
 * conv1d_bwd: din[ic,src] += g*w, dw[oc,ic,tap] += x*g, db[oc] += g.
 *   in [in_ch*n_in], w [out_ch*in_ch*k], dout [out_ch*n_out],
 *   din/dw/db must be ZEROED by the caller (accumulation target).
 * convT1d_bwd: w layout [in_ch*out_ch*k] (PyTorch ConvTranspose1d).
 * act_bwd: mode 0 lrelu (x = PRE-act), 1 tanh (x = POST-act).
 * Returns 0 on success, -1 on failure. */
int wubu_vk_bwd_conv1d(WuBuVk *vk,
                       const float *in, int in_ch,
                       const float *w, const float *dout,
                       int out_ch, int k, int stride, int pad, int dil,
                       int n_in, int n_out,
                       float *din, float *dw, float *db);
int wubu_vk_bwd_convt1d(WuBuVk *vk,
                        const float *in, int in_ch,
                        const float *w, const float *dout,
                        int out_ch, int k, int stride, int pad,
                        int n_in, int n_out,
                        float *din, float *dw, float *db);
int wubu_vk_bwd_act(WuBuVk *vk, const float *x, const float *dout,
                    float *din, int mode, size_t n);

/* public activation op (forward): mode 0 lrelu0.1, 2 tanh, 4 x*=scalar.
 * x is host [n_ch*n]; o may be NULL. Synchronous. */
int wubu_vk_act(WuBuVk *vk, float *x, const float *o, int mode,
                int n_ch, int n, float sc);

#endif /* WUBU_VK_H */
