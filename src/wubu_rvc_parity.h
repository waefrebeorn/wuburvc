#ifndef WUBU_RVC_PARITY_H
#define WUBU_RVC_PARITY_H

/* wubu_rvc_parity.h — Full Mangio-RVC-Fork parity for WuBuRVC.
 *
 * Our goal: load .pth/.index models from Mangio-RVC-Fork and produce
 * output that matches (or exceeds) the reference quality — in pure C11.
 *
 * Pipeline (mirrors Mangio-RVC-Fork/extract_web.py + models.py):
 *   1. HuBERT content encoder (layer 12, 768-dim, 16kHz, hop=320)
 *   2. RMVPE pitch extractor (U-Net, 320-dim F0)
 *   3. Hubert content fusion + top-k FAISS retrieval
 *   4. VITS posterior encoder (Glow, 4 flow layers)
 *   5. HiFi-GAN vocoder (multi-period + multi-scale)
 *
 * License: WaefreBeorn-UMV3
 */

#include "wubu_rvc.h"

/* Forward declare — full struct defined in wubu_rvc_parity.c via wubu_rvc.c */
struct WuBuRVCModel;

/* ── HuBERT content encoder ── */
#define HUBERT_INPUT_SR    16000    /* Required sample rate */
#define HUBERT_HOP_SIZE    320      /* 20ms at 16kHz */
#define HUBERT_HIDDEN_DIM  256      /* Transformer hidden */
#define HUBERT_N_LAYERS    12       /* 12 transformer layers */
#define HUBERT_N_HEADS     8       /* 8 attention heads */
#define HUBERT_VOCAB_SIZE  300     /* Quantized token vocab */
#define HUBERT_CONTENT_DIM_768 768  /* v2: layer 12 output */
#define HUBERT_CONTENT_DIM_256 256  /* v1: layer 9 + final_proj */

/* HuBERT layer selection (research-backed):
 * - Layer 6: best phonetic content (ContentVec paper)
 * - Layer 9: best speaker disentanglement (ContentVec paper)
 * - Layer 12: full context (RVC v2 standard)
 * - Layer 6 of WavLM: optimal for VC (WavLM paper) */
#define HUBERT_LAYER_PHONE    6    /* phonetic content */
#define HUBERT_LAYER_DISCNT   9    /* speaker disentangled */
#define HUBERT_LAYER_FULL     12   /* full context (v2 default) */

/* Mind-meld fusion weights: blend ContentVec (layer 9, speaker-disentangled)
 * with HuBERT (layer 12, full context) for max quality.
 * WavLM (layer 6) is the lock-in replacement with 22.6% better content.
 * Weights sum to 1.0 for normalized fusion. */
#define MIND_MELD_CONTENTVEC_W  0.35f   /* speaker-disentangled content */
#define MIND_MELD_HUBERT_W      0.45f   /* full context (v2 baseline) */
#define MIND_MELD_WAVLM_W       0.20f   /* noise-robust content (lock-in) */

/* HuBERT model structure (matches hubert_base.pt) */
typedef struct {
    /* Feature encoder: 512-dim conv (7-layer) */
    float conv_weights[7][512][512];  /* [layer][out_ch][in_ch*5] */
    int   conv_kernel[7];
    int   conv_stride[7];
    /* Post conv feature projection */
    float post_conv_weight[512][256];
    float post_conv_bias[256];
    /* Transformer encoder layers (12 × TransformerLayer) */
    float enc_self_attn_in_proj[768 * 3];
    float enc_self_attn_out_proj[768 * 768];
    float enc_self_attn_in_proj_bias[768 * 3];
    float enc_self_attn_out_proj_bias[768];
    float enc_ffn_1[768 * 3072];    /* linear1 */
    float enc_ffn_2[3072 * 768];    /* linear2 */
    float enc_ffn_1_bias[3072];
    float enc_ffn_2_bias[768];
    float enc_attn_norm[768];
    float enc_attn_norm_1[768];
    float enc_final_norm[768];
    /* Final projection (v1) */
    float final_proj[256 * 768];
    float final_proj_bias[256];
} WuBuHuBERT;

/* WavLM — lock-in HuBERT replacement (same architecture, 22.6% better content).
 * Same 12-layer transformer, 768 hidden, 8 heads — compatible with HuBERT
 * .pth weights. Noise-aware pre-training gives superior speaker/content
 * disentanglement. Use layer 6 for optimal VC (per WavLM paper). */
typedef WuBuHuBERT WuBuWavLM;

/* Mind-meld: fuse ContentVec (speaker-disentangled) + HuBERT (full context)
 * + WavLM (noise-robust) for maximum quality. All three share the same
 * architecture, so a single weight set can serve all three roles.
 * Returns fused content features that preserve phonetic content while
 * removing speaker identity leakage. */
int wubu_content_mind_meld(const WuBuHuBERT *hubert,
                            const WuBuWavLM *wavlm,
                            const float *pcm, int n_samples,
                            int version,
                            float *feats_out, int max_feats);

/* RMVPE pitch extractor (U-Net based) */
typedef struct {
    int   n_fft;
    int   hop_length;
    int   num_mel;
    float *mel_filterbank;  /* [num_mel * (n_fft/2+1)] */
    /* U-Net encoder */
    float *unet_enc;          /* Pre-trained weights */
    /* CNN feature extractor */
    float *cnn_weights;
    float *cnn_bias;
} WuBuRMVPE;

/* RVC model structure — mirrors Mangio-RVC-Fork .pth format */
struct WuBuRVCModel;

/* Vocal separator types */
#define WUBU_VOCAL_SEP_NONE   0
#define WUBU_VOCAL_SEP_UVR5   1
#define WUBU_VOCAL_SEP_MDX    2
#define WUBU_VOCAL_SEP_DEMUCS 3

/* Training metadata (for .wubu format) */
typedef struct {
    char   dataset_hash[33];
    char   base_model[256];
    char   content_encoder[64];
    char   vocoder[32];
    int    epochs;
    int    sample_rate;
    int    mel_channels;
    int    hidden_channels;
    int    spec_channels;
    int    segment_size;
    int    n_flow_layers;
    int    version;
    int    use_pitch_guidance;
    float  pitch_range_min;
    float  pitch_range_max;
    char   vocal_separator[32];
    char   training_notes[256];
} WuBuTrainingMeta;

struct WuBuRVCModel {
    /* Version: 1 (v1) or 2 (v2) */
    int version;

    /* Speaker embedding */
    float speaker_emb[512];
    int   n_speakers;
    int   has_spk_embed;  /* 1 = multi-speaker model with spk_embed_dim */

    /* HuBERT (also serves as ContentVec and WavLM — same architecture) */
    WuBuHuBERT hubert;
    /* WavLM — same weights as HuBERT but used at layer 6 for noise-robust
     * content extraction (mind-meld). NULL if not configured. */
    WuBuWavLM *wavlm;
    int use_mind_meld;  /* 1 = fuse 3 encoders, 0 = single HuBERT */

    /* RMVPE */
    WuBuRMVPE rmvpe;

    /* VITS posterior encoder */
    /* Flow: 4 Glow coupling layers, residual coupling */
    int   n_flow_layers;
    int   hidden_channels;
    float *flow_g_low[4];      /* coupling: low-frequency path */
    float *flow_g_high[4];     /* coupling: high-frequency path */
    float *actnorm_weight[4];
    float *actnorm_bias[4];
    float *actnorm_log_s[4];
    float *actnorm_logdet[4];

    /* HiFi-GAN generator */
    int   n_residual_layers;
    float *hifi_upsample[4];   /* 4x upsampling in 4 layers */
    int    hifi_upsample_len[4];
    float *hifi_upsample_denorm[4]; /* weight_norm de-normalized (weight_g * weight_v/||v||) */
    int    hifi_upsample_denorm_len[4];
    float *hifi_mrf[4];        /* multi-receptive-field blocks */
    int    hifi_mrf_len[4];
    float *hifi_mrf0;          /* resblocks convs1 (alias for convenience) */
    int    hifi_mrf0_len;
    float *hifi_mrf1;          /* resblocks convs2 (alias for convenience) */
    int    hifi_mrf1_len;
    float *hifi_input_conv;
    int    hifi_input_conv_len;

    /* Vocoder: NSF-HiFiGAN */
    float *vocoder_conv_pre;
    int    vocoder_conv_pre_len;
    float *vocoder_ups[4];
    float *vocoder_mrf[4];
    float *vocoder_conv_post;
    int    vocoder_conv_post_len;

    /* FAISS index data (for retrieval) */
    float *retrieval_features;  /* Training set content features */
    int    n_index_vectors;
    int    index_dim;
    float *retrieval_vectors;   /* [n_index * dim] */
    int    n_retrieval;         /* top-k retrieval */

    /* Tensor map (loaded from .pth / .bin) */
    RVCTensor *tensors;
    int       n_tensors;
    float    *weight_blob;     /* raw blob backing tensor data */
    size_t    weight_blob_size;

    /* Config */
    int   sample_rate;
    int   mel_channels;
    int   upsample_rate;
    int   upsample_rates[8];  /* e.g. [10,10,2,2] for 40k Cartman, [12,10,2,2] for 48k Miku */
    int   upsample_kernel_sizes[8];  /* e.g. [16,16,4,4] for 40k */
    int   n_upsample_layers;
    /* MRF resblock topology (model config, agnostic — no hardcoded [3,7,11]) */
    int   resblock_k[8];       /* kernel size per stack */
    int   resblock_dil[8][8];  /* dilations per stack per conv pair */
    int   n_resblock_pairs;    /* conv pairs per stack */
    int   n_mrf_stacks;        /* resblock stack count */
    float version_f;

    /* Training provenance (from .wubu meta or inferred from .pth) */
    WuBuTrainingMeta meta;
    int   version_loaded;     /* RVC version (1 or 2) */
    float last_loss;          /* last training loss */
    int   last_epoch;         /* last training epoch */
    float last_snake_sat;     /* snake run: fraction of samples at tanh saturation (0..1) */
    int   vocoder_type;       /* 0 = HiFi-GAN, 1 = BigVGAN, 2 = MRF-HiFiGAN */
    int   vocal_separator;    /* 0 = none, 1 = UVR5, 2 = MDX-Net, 3 = Demucs */

    /* Runtime */
    int   loaded;
    int   in_memory;
};

typedef struct WuBuRVCModel WuBuRVCModel;

/* ── API ── */

/* Load a Mangio-RVC-Fork .pth model file.
 * The .pth is a PyTorch state_dict saved via torch.save().
 * We parse the ZIP container (PyTorch uses ZIP format) and
 * extract weight names + raw float data.
 * Returns NULL on failure. */
WuBuRVCModel *wubu_rvc_load_model(const char *pth_path);

/* Load FAISS .index file (retrieval cache).
 * The .index is a flat L2 IVF index. We parse the binary format
 * and load centroids + vectors into memory. */
int wubu_rvc_load_index(WuBuRVCModel *model, const char *index_path);

/* Extract HuBERT content features from 16kHz PCM.
 * Input: pcm[16k samples], n_samples
 * Output: feats[n_frames * 768] (v2) or feats[n_frames * 256] (v1)
 * Matches Mangio-RVC-Fork layer 12 (v2) or layer 9 + final_proj (v1). */
int wubu_hubert_extract(const WuBuHuBERT *hubert,
                         const float *pcm, int n_samples,
                         int version,
                         float *feats_out, int max_feats);

/* Extract pitch (F0) using RMVPE.
 * Input: pcm[16k samples], n_samples
 * Output: f0[n_frames] — F0 values in Hz (0 = unvoiced)
 * Uses the same U-Net architecture as Mangio-RVC-Fork's RMVPE. */
int wubu_rmvpe_extract(const WuBuRMVPE *rmvpe,
                        const float *pcm, int n_samples,
                        float *f0_out, int max_frames);

/* Top-k retrieval from FAISS index.
 * Returns indices of k nearest neighbors in the training set. */
int wubu_rvc_retrieve(const WuBuRVCModel *model,
                       const float *query_feat, int dim,
                       int k, int *out_indices);

/* ── Exact HiFi-GAN kernel (transposed conv + MRF + conv_post) ──
 * Replaces the simplified linear-interpolation wubu_kernel_hifigan.
 * Produces output matching PyTorch ConvTranspose1d (400x upsample).
 *
 * Upsample rates: [10, 10, 2, 2] → total 400x
 * Upsample kernels: [16, 16, 4, 4]
 * Channels: 256→512→256→128→64→1 (conv_post)
 *
 * Input mel: (n_frames, 256) — pre-conv_pre + PixelShuffle
 * Output audio: (n_frames * ~396, 1) — caller-allocated
 *
 * resblock_w/stack/block/conv: dec.resblocks.{s}.convs{k}.{b}.weight (de-norm)
 * resblock_k: kernel sizes [3,7,11] for convs1, [3,7,11] for convs2
 * resblock_dil: dilations [(1,3),(1,3),(1,3)] for stacks 0,1,2
 */
int wubu_kernel_hifigan_exact(const WuBuRVCModel *model,
                               const float *mel_input,
                               int n_frames,
                               float *output,
                               int max_output,
                               int use_snake);

/* Synthesize audio from content features + F0.
 * Full RVC pipeline: content → flow posterior → generator → HiFi-GAN.
 * Matches Mangio-RVC-Fork's VITS + NSF-HiFiGAN output.
 * Returns number of output samples, or -1 on error. */
int wubu_rvc_synthesize_full(WuBuRVCModel *model,
                                const float *content_feats,
                                const float *f0,
                                int n_frames,
                                int content_dim,
                                float *output,
                                int max_samples);

/* Clean up */
void wubu_rvc_model_free(WuBuRVCModel *model);

/* ── Flat-binary weight loader (WUBU format) ──
 * Reads the WUBU-format binary produced by tools/extract_rvc_weights.py.
 * Maps tensor names to internal weight arrays (hifi_upsample, hifi_mrf, etc.).
 * This is the bridge between a real .pth checkpoint and our fused kernels. */
int wubu_rvc_load_weights(WuBuRVCModel *model, const char *bin_path);

/* Lookup a tensor by name (substring match). Returns NULL if not found. */
const RVCTensor *wubu_rvc_find_tensor(const WuBuRVCModel *model, const char *name);

/* Apply weight normalization de-normalization:
 * W = weight_g * (weight_v / ||weight_v||)
 * Resolves weight_g + weight_v decomposition used in RVC v2 checkpoints.
 * per_ch = total_elements / n_channels */
void wubu_rvc_denormalize_weight(const float *weight_g, const float *weight_v,
                                  float *out, int n_elements, int n_channels);

/* Model info */
typedef struct {
    int   rvc_version;       /* 1 or 2 */
    int   sample_rate;
    int   mel_channels;
    int   n_speakers;
    int   hubert_layer;
    int   content_dim;
    char  version_str[16];   /* "v1" or "v2" */
    int   index_loaded;
    long  n_index_vectors;
    int   hidden_channels;
    int   n_flow_layers;
    int   n_residual_layers;
    float retrieval_ratio;
} RVCModelInfo;

void wubu_rvc_model_info(const WuBuRVCModel *model, RVCModelInfo *out);

/* FAISS index parsing — parse binary .index file
 * (IVF + Flat L2 format used by Mangio-RVC-Fork) */
typedef struct {
    int    d;           /* dimension */
    int    nb;          /* number of vectors */
    float *centroids;   /* [nlist * d] */
    int    nlist;
    int    nprobe;
    float *vectors;     /* [nb * d] */
} WuBuFaissIndex;

WuBuFaissIndex *wubu_faiss_load(const char *path);
void wubu_faiss_free(WuBuFaissIndex *idx);
int wubu_faiss_search(const WuBuFaissIndex *idx,
                       const float *query, int dim, int k,
                       int *out_indices, float *out_distances);

/* HuBERT weight loading — parse PyTorch .pt/.pth tensors
 * and map to our WuBuHuBERT struct */
int wubu_hubert_load_weights(WuBuHuBERT *hubert,
                              const char *pth_path);

/* RMVPE weight loading */
int wubu_rmvpe_load_weights(WuBuRMVPE *rmvpe,
                              const char *pth_path);

#endif /* WUBU_RVC_PARITY_H */
