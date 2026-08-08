#ifndef WUBU_RVC_H
#define WUBU_RVC_H

/* wubu_rvc.h — WuBuRVC: Our own RVC inference engine (C11).
 *
 * Key insight: We don't need RVC's legacy pipeline. We load existing .pth
 * model weights (Hubert content encoder + flow + HiFi-GAN generator/vocoder),
 * extract the tensor values, and reinterpret them in OUR OWN virtualized
 * frame buffer space. Then we execute through our own fused kernels —
 * designed like a game engine with a unified buffer abstraction that works
 * on both CPU and GPU.
 *
 * Architecture:
 *
 *   .pth/.onnx/.index files ──→ wubu_rvc_load_model()
 *       │  (gguf_reader extracts tensor names + data)
 *       ▼
 *   RVCGraph { tensor_map: name → tensor_view }
 *       │  (we map RVC's layer structure into our own IR)
 *       ▼
 *   wubu_frame_buffer_t — virtualized frame buffer space
 *       │  (unified CPU/GPU memory with layout abstraction)
 *       ▼
 *   Fused kernels (our design):
 *     • wubu_kernel_auto norm()   — ActNorm as inline buffer op
 *     • wubu_kernel_flow_couple() — Affine Coupling fused into 1 pass
 *     • wubu_kernel_hifigan()    — Upsample + MRF + LRELU fused
 *     • wubu_kernel_vocoder()    — Residual stack + tanh fused
 *       │
 *       ▼
 *   Output waveform  ←  wubu_frame_buffer_read()
 *
 * License: WaefreBeorn-UMV3
 */

#include <stddef.h>
#include <stdint.h>

/* Forward declare WuBuRVCModel (full def in wubu_rvc_parity.h) */
struct WuBuRVCModel;

#ifdef __cplusplus
extern "C" {
#endif

/* RVC model format version */
typedef enum {
    RVC_V1 = 1,
    RVC_V2 = 2,
    RVC_V3 = 3
} RVCVersion;

/* ---- Error codes (visible in both C and C++) ---- */
#define WUBU_RVC_OK           0
#define WUBU_RVC_ERR_NOGPU    -1
#define WUBU_RVC_ERR_MODEL    -2
#define WUBU_RVC_ERR_NOINIT   -3
#define WUBU_RVC_ERR_ARGS     -4
#define WUBU_RVC_ERR_CUDA     -5
#define WUBU_RVC_ERR_FILE     -6

/* ---- Virtualized Frame Buffer ---- */
/* Unified abstraction: a named region of memory that can live on CPU or GPU.
 * We design it like a game engine's render target — you bind buffers,
 * attach them to kernels, and the engine handles the transfer. */

typedef enum {
    WUBU_BUF_CPU = 0,
    WUBU_BUF_CUDA = 1,
    WUBU_BUF_MANAGED = 2
} WuBuBufferType;

typedef struct {
    void       *ptr;        /* CPU or GPU pointer */
    size_t      bytes;      /* allocated size */
    WuBuBufferType type;
    int         device_id;  /* GPU device (if CUDA) */
    char        name[64];   /* debug label */
} wubu_frame_buffer_t;

/* Create a frame buffer (allocates CPU or GPU memory) */
int wubu_frame_buffer_create(wubu_frame_buffer_t *fb, size_t n_floats,
                              WuBuBufferType type, const char *name);

/* Destroy a frame buffer */
void wubu_frame_buffer_destroy(wubu_frame_buffer_t *fb);

/* Copy data in/out (handles CPU↔GPU automatically) */
int wubu_frame_buffer_write(wubu_frame_buffer_t *fb,
                             const float *src, size_t n_floats);
int wubu_frame_buffer_read(const wubu_frame_buffer_t *fb,
                            float *dst, size_t n_floats);

/* Sync frame buffer (CPU↔GPU transfer) */
int wubu_frame_buffer_sync(wubu_frame_buffer_t *fb);

/* ---- RVC Configuration ---- */
typedef struct RVCConfig {
    char  model_path[512];
    char  index_path[512];
    char  hubert_path[512];
    RVCVersion version;
    int   sample_rate;
    int   use_cuda;
    int   gpu_device;
    int   fp16;
    int   mel_channels;
    int   hidden_channels;
    int   filter_length;
    int   hop_length;
    int   win_length;
    int   n_flow_layers;
    int   n_hifigan_upsamples;
    int   n_mrf_stacks;
    int   n_residual_layers;
    int   reservoir_size;
    double speed_factor;
    double pitch_shift;
} RVCConfig;

/* ---- RVC Model IR (Intermediate Representation) ---- */
typedef struct {
    char   name[128];    /* tensor name from .pth */
    float *data;         /* tensor data (CPU pointer) */
    int    n_dims;       /* number of dimensions */
    int    dims[4];      /* dimension sizes (max 4D for RVC) */
    int    offset;       /* byte offset in the loaded weight blob */
} RVCTensor;

typedef struct {
    RVCTensor *tensors;
    int        n_tensors;
    int        version;     /* RVC_V1, V2, V3 */
    int        sample_rate;
    int        mel_channels;
    int        hidden_channels;
    int        n_flow_layers;
    int        n_upsample_layers;
    int        upsample_rates[8];  /* e.g. [10,10,2,2] for 40k, [12,10,2,2] for 48k */
    int        upsample_kernel_sizes[8];  /* e.g. [16,16,4,4] for 40k */
    int        upsample_rate;      /* product of upsample_rates[] */
    int        n_mrf_stacks;
    int        n_residual_layers;
    /* MRF resblock topology (from model config, fields 6/7) */
    int        resblock_k[8];       /* kernel size per stack, e.g. [3,7,11] */
    int        resblock_dil[8][8];  /* dilations per stack per conv pair */
    int        n_resblock_pairs;    /* conv pairs per stack (e.g. 3) */
} RVCGraph;

/* ---- Main RVC engine ---- */
typedef struct WuBuRVC {
    RVCGraph   graph;
    wubu_frame_buffer_t workspace;
    RVCConfig  cfg;
    int        initialized;
    int        cuda_available;
    char       cuda_device_name[256];
    long       total_inferences;
    long       cache_hits;
    double     last_latency_ms;
    char      *weight_blob;
    size_t     weight_blob_size;
    struct WuBuRVCModel *model;
    int        rvc_version;
    int        sample_rate;
    int        mel_channels;
    int        hidden_channels;
    int        loaded;
    int        cuda_active;
    size_t     vram_total_mb;
    size_t     vram_used_mb;
} WuBuRVC;

/* Engine info */
typedef struct {
    int   cuda_available;
    int   cuda_device_count;
    char  cuda_device_name[256];
    int   cuda_major, cuda_minor;
    size_t vram_total_mb;
    size_t vram_used_mb;
    int   rvc_version;
    long  total_inferences;
    long  cache_hits;
    double last_latency_ms;
    int   loaded;
    int   cuda_active;
} RVCInfo;

/* Load model weights and build RVCGraph IR. */
WuBuRVC *wubu_rvc_load(const RVCConfig *cfg);
void     wubu_rvc_destroy(WuBuRVC *rvc);

/* Synthesize waveform from mel-spectrogram. */
int wubu_rvc_synthesize(WuBuRVC *rvc,
                         const float *mel_input, int n_frames, int mel_ch,
                         float *output, int n_samples);

/* Convert raw audio directly (mel extraction internal). */
int wubu_rvc_convert_audio(WuBuRVC *rvc,
                            const float *input, int n_input,
                            float *output, int n_samples);

/* Get info */
void wubu_rvc_info(const WuBuRVC *rvc, RVCInfo *out);

/* Check if a model file is actually loaded */
int wubu_rvc_is_model_loaded(const WuBuRVC *rvc);

/* ---- Kernel launchers (our own fused kernels) ---- */
int wubu_kernel_autonorm(wubu_frame_buffer_t *fb,
                          const float *scale, const float *bias,
                          int n_channels);

int wubu_kernel_flow_couple(wubu_frame_buffer_t *input,
                             wubu_frame_buffer_t *output,
                             const float *coupling_w,
                             const float *coupling_b,
                             int n_frames, int hidden_ch);

int wubu_kernel_hifigan(wubu_frame_buffer_t *input,
                         wubu_frame_buffer_t *output,
                         const float *upsample_w,
                         const float *upsample_b,
                         const float *mrf_w,
                         int n_input, int n_output, int hidden_ch);

int wubu_kernel_vocoder(wubu_frame_buffer_t *input,
                         wubu_frame_buffer_t *output,
                         const float *res_w, const float *res_b,
                         const float *out_w,
                         int n_samples, int n_layers);

int wubu_rvc_cuda_init(WuBuRVC *rvc);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RVC_H */
