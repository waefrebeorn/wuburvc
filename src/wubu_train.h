/* wubu_train.h — WuBuRVC training engine (C11, OpenMP).
 *
 * Real generator (HiFi-GAN decoder) training:
 *   forward  → mel(192,F) → conv_pre → ups×4 + MRF → conv_post → tanh → audio
 *   loss     → MSE (differentiable; STFT loss kept for evaluation)
 *   backward → tanh' / conv1d / convT1d / LReLU / MRF chain rule
 *   update   → AdamW (decoupled weight decay, Loshchilov & Hutter 2019)
 *
 * The registry points at the EXACT arrays inference reads:
 *   - ups layers      → model->hifi_upsample_denorm[i] (inference source)
 *   - conv_pre/conv_post/resblocks → tensor .data (de-normalized in place)
 * so a trained model is a model the engine can immediately synthesize with.
 *
 * Verification: wubu_train_gradcheck() compares analytic gradients against
 * finite differences (Triple-DA standard); wubu_train_step() must decrease
 * loss over epochs (see src/test_train.c).
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_TRAIN_H
#define WUBU_TRAIN_H

#include "wubu_rvc_parity.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── AdamW optimizer ── */
typedef struct WuBuAdamW WuBuAdamW;

WuBuAdamW *wubu_adamw_create(int n_params, float lr, float beta1, float beta2,
                             float eps, float weight_decay);
int        wubu_adamw_init_param(WuBuAdamW *opt, int idx, int n_elem);
void       wubu_adamw_step(WuBuAdamW *opt, int idx, float *param, const float *grad);
void       wubu_adamw_free(WuBuAdamW *opt);

/* ── Trainable-parameter registry ──
 * One entry per weight/bias array that gets gradient + AdamW state.
 * `data` is the array inference actually reads; `grad` is accumulated by
 * the backward pass and consumed by the AdamW step. */
typedef struct {
    char   name[128];
    float *data;
    float *grad;
    int    n;
} WuBuTrainParam;

typedef struct {
    WuBuTrainParam *params;
    int    count;
    int    cap;
} WuBuTrainRegistry;

int  wubu_train_registry_build(WuBuRVCModel *model, WuBuTrainRegistry *reg);
int  wubu_train_registry_find(const WuBuTrainRegistry *reg, const char *name);
void wubu_train_registry_free(WuBuTrainRegistry *reg);
void wubu_train_registry_zero_grads(WuBuTrainRegistry *reg);

/* ── Losses ── */
float wubu_mse_loss(const float *a, const float *b, int n);
float wubu_mae_loss(const float *a, const float *b, int n);
float wubu_stft_loss(const float *a, const float *b, int n, int sr);
float wubu_gan_g_loss(float d_real_out);
float wubu_gan_d_loss(float d_real_out, float d_fake_out);

/* ── Decoder forward (generator only, mirror of the parity-verified path).
 * mel_in: (192, n_frames) col-major (flow output; first 80 = mel, rest zero).
 * Returns sample count (n_frames * upsample_rate) or -1. */
int wubu_decoder_forward(WuBuRVCModel *model, const float *mel_in, int n_frames,
                         float *audio, int max_samples);

/* ── One training step: forward → MSE → backward → AdamW.
 * Returns 1 when converged (loss < 0.1), 0 otherwise, -1 on error. */
int wubu_train_step(WuBuRVCModel *model, WuBuTrainRegistry *reg, WuBuAdamW *opt,
                    const float *mel_in, int n_frames,
                    const float *wav, int n_samples,
                    float *loss_out, int epoch);

/* ── CUDA-accelerated training (wubu_train_cuda.cu).
 * Same math as wubu_train_step but the decoder forward runs on the GPU
 * with cached pre-activations, and the backward conv/convT kernels run on
 * the GPU; grads are pulled back to the host registry per step, then the
 * host AdamW updates (weights live on host — the model is always
 * synthesis-ready). Opaque cache type for ABI safety. */
typedef struct TrainCacheCuda TrainCacheCuda;

int  wubu_train_forward_cuda(WuBuRVCModel *model, const float *mel_in, int n_frames,
                             float *audio, int max_samples, TrainCacheCuda *cache);
TrainCacheCuda *wubu_train_cache_alloc_cuda(void);
void wubu_train_cache_free_cuda(TrainCacheCuda *cache);
int  wubu_train_backward_cuda(WuBuRVCModel *model, TrainCacheCuda *cache,
                              const float *mel_in, const float *d_audio,
                              WuBuTrainRegistry *reg);
int  wubu_train_step_cuda(WuBuRVCModel *model, WuBuTrainRegistry *reg, WuBuAdamW *opt,
                          const float *mel_in, int n_frames,
                          const float *wav, int n_samples,
                          float *loss_out, int epoch);

/* ── Vulkan training backend (wubu_train_vk.c) ── */
typedef struct WuBuVk WuBuVk;   /* opaque (wubu_vk.h) */
typedef struct TrainCacheVk TrainCacheVk;

int  wubu_train_forward_vk(WuBuVk *vk, WuBuRVCModel *model, const float *mel_in,
                           int n_frames, float *audio, int max_samples,
                           TrainCacheVk *cache);
TrainCacheVk *wubu_train_cache_alloc_vk(void);
void wubu_train_cache_free_vk(TrainCacheVk *cache);
int  wubu_train_backward_vk(WuBuVk *vk, WuBuRVCModel *model, TrainCacheVk *cache,
                            const float *mel_in, const float *d_audio,
                            WuBuTrainRegistry *reg);
int  wubu_train_step_vk(WuBuVk *vk, WuBuRVCModel *model, WuBuTrainRegistry *reg,
                        WuBuAdamW *opt, const float *mel_in, int n_frames,
                        const float *wav, int n_samples,
                        float *loss_out, int epoch);

/* Public wrapper around the static decoder_backward (Triple-DA A/B for the
 * CUDA training backend). Returns sample count or -1. */
int wubu_train_backward_cpu(WuBuRVCModel *model, const float *mel_in, int n_frames,
                            const float *wav, int n_samples,
                            WuBuTrainRegistry *reg);

/* ── Gradient check vs finite differences (Triple-DA).
 * Probes up to n_probes random weights; returns max relative error
 * (a value < 1e-3 means the analytic backward is correct). */
float wubu_train_gradcheck(WuBuRVCModel *model, WuBuTrainRegistry *reg,
                           const float *mel_in, int n_frames,
                           const float *wav, int n_samples,
                           int n_probes, int seed);

/* Train from scratch: initialize RVC v2 generator weights (Xavier-ish). */
int wubu_init_weights_rvc2(WuBuRVCModel *model);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TRAIN_H */
