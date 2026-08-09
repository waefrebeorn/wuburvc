#ifndef WUBU_RVC_REAL_H
#define WUBU_RVC_REAL_H

/* wubu_rvc_real.h — WuBuRVC REAL synthesis pipeline (v1 + v2).
 *
 * This is the actual RVC voice-conversion math, ported faithfully from
 * Mangio-RVC-Fork lib/infer_pack/models.py (SynthesizerTrnMs768NSFsid /
 * SynthesizerTrnMs256NSFsid). No placeholders, no synthetic kernels:
 *
 *   infer(phone, phone_lengths, pitch, nsff0, sid):
 *     g      = emb_g(sid).unsqueeze(-1)                    # [1, gin, 1]
 *     m_p, logs_p, x_mask = enc_p(phone, pitch, lengths)   # TextEncoder
 *     z_p    = (m_p + exp(logs_p)*randn*0.66666) * x_mask
 *     z      = flow(z_p, x_mask, g=g, reverse=True)        # 4x ResidualCoupling
 *     o      = dec(z * x_mask, nsff0, g=g)                 # GeneratorNSF
 *
 * RVC v1: content_dim=256 (HuBERT layer 9 + final_proj), TextEncoder256.
 * RVC v2: content_dim=768 (HuBERT layer 12), TextEncoder768.
 * Everything else (flow + generator) is shared, parameterized by config.
 *
 * License: WaefreBeorn-UMV3
 */

#include "wubu_rvc_parity.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Real tensor helpers ──
 * weight_norm (weight_g + weight_v) resolution for flow/encoder tensors.
 * W = weight_g * (weight_v / ||weight_v||) per channel.
 * out must be n_elements floats. Returns 0 on success. */
int wubu_denorm_tensor(const RVCTensor *g, const RVCTensor *v,
                       float *out, int n_elements);

/* Conv tile size: 8192 (default) = byte-identical reference output;
 * 2048 = speed mode (~3% faster, ~1 LSB diff at tile boundaries — for
 * real-time use, NOT for master rendering). */
void wubu_set_conv_tile(int tile);
int wubu_get_conv_tile(void);

/* Fast-math switch: 0 (default) = libm expf/tanhf (byte-identical output);
 * 1 = folded-poly/bit-trick approximations (~7e-6 accuracy) — speed mode.
 * Defined in wubu_math.h (shared), set by the CLI in --mode speed. */
void wubu_set_fast_math(int on);
int wubu_get_fast_math(void);

/* ── Real RVC synthesis (the whole acoustic model) ──
 * model: loaded WuBuRVCModel (tensor map must contain enc_p.*, flow.*,
 *        dec.*, emb_g.weight).
 * content: [n_frames * content_dim] HuBERT content (256 v1 / 768 v2).
 * f0_coarse: [n_frames] int 1..255 (mel-quantized pitch).
 * nsff0: [n_frames] float Hz pitch (raw f0, 0=unvoiced).
 * sid: speaker id (0 for single-speaker models).
 * out_audio: [n_frames * 400] float waveform at model sample rate.
 * Returns number of output samples, or -1 on error.
 *
 * NOTE: deterministic inference — the randn term in z_p is fixed to 0
 * (RVC reference uses torch.randn_like; for parity tests we pass
 * randn_scale=0.0f, the exact reference value is 0.66666f). */
int wubu_rvc_synthesize_real(WuBuRVCModel *model,
                             const float *content, int n_frames, int content_dim,
                             const int *f0_coarse, const float *nsff0,
                             int sid, float randn_scale,
                             float *out_audio, int max_samples,
                             int use_snake);

/* GPU generator variant (wubu_rvc_cuda.cu): flow on CPU, GeneratorNSF on CUDA. */
int wubu_rvc_synthesize_real_cuda(WuBuRVCModel *model,
                                  const float *content, int n_frames, int content_dim,
                                  const int *f0_coarse, const float *nsff0,
                                  int sid, float randn_scale,
                                  float *out_audio, int max_samples,
                                  int use_snake);

/* Vulkan generator variant (wubu_vk.c): flow on CPU, GeneratorNSF via
 * Vulkan compute shaders (cross-vendor — NVIDIA/AMD/Intel). */
int wubu_rvc_synthesize_real_vk(WuBuRVCModel *model,
                                const float *content, int n_frames, int content_dim,
                                const int *f0_coarse, const float *nsff0,
                                int sid, float randn_scale,
                                float *out_audio, int max_samples,
                                int use_snake);

/* CUDA GeneratorNSF (defined in wubu_rvc_cuda.cu). */
int wubu_generator_nsf_cuda(WuBuRVCModel *model,
                            const float *z, int nF, int inter_channels,
                            const float *nsff0, const float *g,
                            float *out_audio, int max_samples,
                            int inject_noise, int use_snake);

/* ── TextEncoder768 / TextEncoder256 (enc_p) ──
 * Returns 0 on success. m/logs are [n_frames * inter_channels] each. */
int wubu_enc_p_forward(WuBuRVCModel *model,
                       const float *phone, int n_frames, int content_dim,
                       const int *pitch,
                       float *m_out, float *logs_out, float *x_mask_out);

/* ── Flow reverse (ResidualCouplingBlock, 4 flows + Flip) ──
 * in/out: [n_frames * inter_channels], g: [gin] (speaker embedding row). */
int wubu_flow_reverse(WuBuRVCModel *model,
                      const float *z_p, int n_frames, int inter_channels,
                      const float *g, const float *x_mask,
                      float *z_out);

/* ── GeneratorNSF (dec) with f0 sine excitation ──
 * z: [n_frames * inter_channels], nsff0: [n_frames] Hz.
 * out: audio samples at model sr (40000 for v2 Cartman). */
int wubu_generator_nsf(WuBuRVCModel *model,
                       const float *z, int n_frames, int inter_channels,
                       const float *nsff0, const float *g,
                       float *out, int max_samples,
                       int inject_noise,
                       int use_snake);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RVC_REAL_H */
