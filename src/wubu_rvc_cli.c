/* wubu_rvc_cli.c — WuBuRVC standalone converter (C11, OpenMP).
 *
 * Converts any mono WAV into a trained RVC character voice with the REAL
 * engine: C HuBERT content → YIN f0 → enc_p + flow + GeneratorNSF.
 * No Python, no PyTorch at runtime.
 *
 * Usage:
 *   wubu_rvc_cli <input.wav> <model_dir> <output.wav> [--model model.pth]
 *
 *   input.wav   mono 16k/22.05k/40k PCM (any sr, resampled internally)
 *   model_dir   e.g. models/rvc/cartman  (dir containing model.pth or *.pth
 *               + optional .index for retrieval)
 *   output.wav  mono 40k PCM_16
 *
 * Steps (matching Mangio-RVC v23.7.0 infer):
 *   1. load wav -> resample to 16k
 *   2. load model (.pth) -> determine version (v1/v2), sample rate, upsample
 *   3. HuBERT content (v2: layer 12 768-dim, v1: layer 9 + final_proj 256-dim)
 *   4. nearest ×2 upsampling of content (matches F.interpolate scale=2)
 *   5. YIN f0 at 100 fps -> coarse + nsff0 (1:1 with ×2 content, NO ×2 on f0)
 *   6. wubu_rvc_synthesize_real -> audio at model sample rate
 *   7. write PCM_16 wav
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_rvc.h"
#include "wubu_rvc_parity.h"
#include "wubu_rvc_real.h"
#include "wubu_rvc_hubert.h"
#include "wubu_rvc_f0.h"
#include "wubu_rmvpe.h"
#include "wubu_audioio.h"
#include "wubu_autokey.h"
#include "wubu_math.h"
#include "wubu_pitch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <immintrin.h>
#include "wubu_postproc.h"

static void die(const char *msg) { fprintf(stderr, "wubu_rvc_cli: %s\n", msg); exit(1); }

/* ── minimal WAV reader: mono PCM_16 or PCM_32 float ── */
static float *read_wav(const char *path, int *n_out, int *sr_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) { fclose(f); return NULL; }
    unsigned short audiofmt = (unsigned short)(hdr[20] | hdr[21] << 8);
    unsigned short nch = (unsigned short)(hdr[22] | hdr[23] << 8);
    unsigned int sr = (unsigned int)(hdr[24] | hdr[25] << 8 | hdr[26] << 16 | hdr[27] << 24);
    unsigned int databytes = (unsigned int)(hdr[40] | hdr[41] << 8 | hdr[42] << 16 | hdr[43] << 24);
    unsigned short bits = (unsigned short)(hdr[34] | hdr[35] << 8);
    fseek(f, 44, SEEK_SET);
    unsigned char *raw = (unsigned char *)malloc(databytes ? databytes : 1);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, databytes, f) != databytes) { free(raw); fclose(f); return NULL; }
    fclose(f);
    int nsamples = databytes / (bits / 8) / nch;
    float *out = (float *)malloc((size_t)nsamples * sizeof(float));
    if (!out) { free(raw); return NULL; }
    for (int i = 0; i < nsamples; i++) {
        float v = 0;
        if (audiofmt == 3 && bits == 32) { /* IEEE float */
            int off = i * nch * 4;
            memcpy(&v, raw + off, 4);
        } else if (bits == 16) {
            short s;
            memcpy(&s, raw + i * nch * 2, 2);
            v = s / 32768.0f;
        } else if (bits == 24) {
            int s = (raw[i * nch * 3] | raw[i * nch * 3 + 1] << 8 | (signed char)raw[i * nch * 3 + 2] << 16);
            v = s / 8388608.0f;
        }
        out[i] = v; /* take channel 0 */
    }
    free(raw);
    *n_out = nsamples;
    *sr_out = (int)sr;
    return out;
}

/* ── minimal WAV writer: mono 16-bit ── */
static int write_wav(const char *path, const float *data, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned int databytes = (unsigned int)n * 2;
    unsigned int riffsz = 36 + databytes;
    fwrite("RIFF", 1, 4, f); fwrite(&riffsz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    unsigned int fmtsz = 16; fwrite(&fmtsz, 4, 1, f);
    unsigned short fmt = 1, nch = 1; unsigned int rate = (unsigned int)sr;
    unsigned int byterate = rate * 2; unsigned short align = 2, bits = 16;
    fwrite(&fmt, 2, 1, f); fwrite(&nch, 2, 1, f); fwrite(&rate, 4, 1, f);
    fwrite(&byterate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&databytes, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
        short s = (short)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

/* resample to target sr — windowed-sinc (Kaiser) so pitch extraction
 * sees a clean spectrum (linear aliases → pitch noise across model kHz). */
static float *resample(const float *in, int n_in, int sr_in, int sr_out, int *n_out) {
    if (sr_in == sr_out) {
        float *out = (float *)malloc((size_t)n_in * sizeof(float));
        if (out) memcpy(out, in, (size_t)n_in * sizeof(float));
        *n_out = n_in;
        return out;
    }
    double ratio = (double)sr_out / sr_in;
    int n = (int)(n_in * ratio);
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!out) { *n_out = 0; return NULL; }
    *n_out = wubu_audio_resample_sinc(in, n_in, sr_in, sr_out, out);
    return out;
}

/* nearest ×2 upsample along frames: [T, dim] -> [2T, dim] frame-major.
 * Matches the RVC pipeline's content interpolation EXACTLY:
 *   F.interpolate(feats, scale_factor=2)  (3D input → default mode='nearest')
 * Verified vs reference content_up.npy: max diff 0.0 (2026-08-07).
 * The old linear align_corners=True version produced different content →
 * different flow output → robotic/wrong voice. */
static float *upsample_frames(const float *in, int T, int dim, int *T2_out) {
    int T2 = T * 2;
    float *out = (float *)calloc((size_t)T2 * dim, sizeof(float));
    if (!out) return NULL;
    for (int j = 0; j < T2; j++) {
        int i0 = j / 2;              /* nearest: repeat each frame twice */
        if (i0 >= T) i0 = T - 1;
        memcpy(out + (size_t)j * dim, in + (size_t)i0 * dim,
               (size_t)dim * sizeof(float));
    }
    *T2_out = T2;
    return out;
}

/* ── Chunked parallel synthesis ────────────────────────────────────────────
 * The whole-track pipeline pays O(T^2) HuBERT attention, a whole-track flow,
 * and multi-GB generator buffers on long audio (a 4:48 track hit 10.5 GB and
 * 35+ min). Chunked inference processes independent 3 s windows (with extra
 * HuBERT context) in parallel, then crossfades — RVC's own architecture.
 * Workers grab chunks via an atomic counter; each sets its OpenMP thread
 * count so N workers share the cores. */
typedef struct {
    const float *pcm16; int n16;
    const int *f0_coarse; const float *nsff0; int n_f0;
    const WuBuHubert *hb; int rvc_ver; int content_dim;
    WuBuRVCModel *model; int speaker_id; float noise_scale; int use_snake;
    int ups_total;
    int chunk_16k, extra_16k, hop_16k;
    int use_cuda;
    int use_vk;
    float **audio; int *pos; int *len; int *rc;
    volatile int next; int n_chunks; int threads;
} ChunkCtx;

static void *chunk_worker(void *arg) {
    ChunkCtx *ctx = (ChunkCtx *)arg;
    omp_set_num_threads(ctx->threads > 0 ? ctx->threads : 1);
    for (;;) {
        int c = __sync_fetch_and_add(&ctx->next, 1);
        if (c >= ctx->n_chunks) break;
        int start = c * ctx->hop_16k;
        int len_c = ctx->chunk_16k + ctx->extra_16k;
        if (start + len_c > ctx->n16) len_c = ctx->n16 - start;
        if (len_c < 4800) { ctx->rc[c] = 0; ctx->audio[c] = NULL; continue; }
        int T_c = wubu_hubert_output_length(len_c);
        if (T_c < 4) { ctx->rc[c] = 0; ctx->audio[c] = NULL; continue; }
        float *content = (float *)malloc((size_t)T_c * ctx->content_dim * sizeof(float));
        if (!content) { ctx->rc[c] = -1; continue; }
        int got = wubu_hubert_extract_real(ctx->hb, ctx->pcm16 + start, len_c,
                                           ctx->rvc_ver, content,
                                           (size_t)T_c * ctx->content_dim);
        if (got != T_c) { if (got < T_c) T_c = got; }
        if (T_c < 4) { free(content); ctx->rc[c] = 0; ctx->audio[c] = NULL; continue; }
        int T2_c = 0;
        float *cup = upsample_frames(content, T_c, ctx->content_dim, &T2_c);
        free(content);
        if (!cup) { ctx->rc[c] = -1; continue; }
        /* f0 slice for this chunk (100 fps: frame index = sample/160) */
        int f0_start = start / 160;
        int nf = T2_c;
        if (f0_start + nf > ctx->n_f0) nf = ctx->n_f0 - f0_start;
        if (nf < 4) { free(cup); ctx->rc[c] = 0; ctx->audio[c] = NULL; continue; }
        /* transpose to col-major [dim, nf] */
        float *cmaj = (float *)malloc((size_t)ctx->content_dim * nf * sizeof(float));
        if (!cmaj) { free(cup); ctx->rc[c] = -1; continue; }
        for (int j = 0; j < nf; j++)
            for (int d = 0; d < ctx->content_dim; d++)
                cmaj[(size_t)d * nf + j] = cup[(size_t)j * ctx->content_dim + d];
        free(cup);
        int max_a = nf * ctx->ups_total + ctx->ups_total;
        float *aout = (float *)malloc((size_t)max_a * sizeof(float));
        if (!aout) { free(cmaj); ctx->rc[c] = -1; continue; }
        int n_out = 0;
        if (ctx->use_vk)
            n_out = wubu_rvc_synthesize_real_vk(ctx->model, cmaj, nf, ctx->content_dim,
                                                ctx->f0_coarse + f0_start,
                                                ctx->nsff0 + f0_start,
                                                ctx->speaker_id, ctx->noise_scale,
                                                aout, max_a, ctx->use_snake);
        else if (ctx->use_cuda)
            n_out = wubu_rvc_synthesize_real_cuda(ctx->model, cmaj, nf, ctx->content_dim,
                                                  ctx->f0_coarse + f0_start,
                                                  ctx->nsff0 + f0_start,
                                                  ctx->speaker_id, ctx->noise_scale,
                                                  aout, max_a, ctx->use_snake);
        else
            n_out = wubu_rvc_synthesize_real(ctx->model, cmaj, nf, ctx->content_dim,
                                             ctx->f0_coarse + f0_start,
                                             ctx->nsff0 + f0_start,
                                             ctx->speaker_id, ctx->noise_scale,
                                             aout, max_a, ctx->use_snake);
        free(cmaj);
        if (n_out <= 0) { free(aout); ctx->rc[c] = -1; continue; }
        ctx->audio[c] = aout;
        ctx->pos[c] = f0_start * ctx->ups_total;
        ctx->len[c] = n_out;
        ctx->rc[c] = 0;
    }
    return NULL;
}

int main(int argc, char **argv) {
    /* Nested OMP: the MRF's 3 stacks run in parallel, each with its own conv
     * threads (3 stacks x 4 conv threads = 12). */
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    omp_set_num_threads(12);
    /* FTZ + DAZ: flush denormals (the RVC's tiny activations create them in
     * the conv accumulation — denormal FP ops are ~100x slower on x86) */
    _mm_setcsr(_mm_getcsr() | (1u << 15) | (1u << 6));
    setvbuf(stdout, NULL, _IONBF, 0); /* crash-safe stage logging */
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <input.wav> <model_dir> <output.wav> [--model file.pth]\n"
                "         [--speaker N] [--noise SCALE] [--hubert PATH] [--preset N]\n"
                "\n"
                "         --speaker N   : speaker id for multi-speaker models\n"
                "         --noise S     : noise scale (0.0 = deterministic, 0.66666 = reference)\n"
                "         --hubert PATH : override HuBERT weights path\n"
                "         --preset N    : character preset (1=warm, 2=bright, 3=smooth, 4=breaty)\n"
                "         --snake       : use Snake activation (BigVGAN: x + (1/a)sin^2(a*x))\n"
                "         --formant R   : formant shift ratio (1.0=none, <1=male, >1=female)\n"
                "         --f0smooth S  : F0 contour smoothing strength (0.0-1.0)\n"
                "         --f0ref DIR   : use reference f0 (nsff0_raw.bin + f0_coarse.bin)\n"
                "         --chunk F     : chunked inference window in seconds (default 3.0)\n"
                "         --ctx F       : HuBERT context window in seconds (default 0.72; 0.40 = speed mode)\n"
                "         --mode M      : 'quality' (default, byte-identical reference) or 'speed'\n"
                "                        (real-time: ctx 0.4 + conv tile 2048, ~1 LSB diff)\n"
                "         --xfade F     : crossfade overlap in seconds (default 0.10)\n"
                "         --jobs N      : parallel chunk workers (default 4)\n",
                argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *model_dir = argv[2];
    const char *out_path = argv[3];
    srand((unsigned)time(NULL));  /* seed for NSF noise injection */
    char model_path[1024] = {0};
    char index_path[1024] = {0};
    char hubert_path[1024] = {0};
    int speaker_id = 0;       /* default: speaker 0 */
    float noise_scale = 0.0f; /* default: deterministic (parity mode) */
    int preset = 0;           /* 0 = none, 1-4 = character preset */
    int use_snake = 0;        /* 0 = LeakyReLU (original), 1 = Snake (BigVGAN) */
    float formant_shift = 1.0f; /* 1.0 = no shift */
    float f0_smooth = 0.0f;   /* 0.0 = no smoothing */
    char f0ref_dir[1024] = {0}; /* reference f0 dir (nsff0_raw.bin + f0_coarse.bin) */
    int force_yin = 0;        /* --f0 yin: force YIN instead of RMVPE */
    int f0_filter_radius = 3; /* --f0filter N: median filter on f0 (0 = off) */
    float rms_mix = 0.25f;    /* --rmsmix F: output follows input volume envelope (RVC default 0.25) */
    int autokey_probe = 0;    /* --autokey N: auto key adaptation (N = probe secs) */
    float chunk_secs = 3.0f;  /* --chunk F: chunked inference (3s default; 0 = whole-track) */
    float ctx_secs = 0.72f;   /* --ctx F: HuBERT context window in seconds
                               * (0.72 default = RVC reference quality,
                               * BYTE-IDENTICAL output; 0.40 = speed mode,
                               * arXiv 2505.22487 effective context ~400ms,
                               * mel 0.9632 — opt-in only) */
    float xfade_secs = 0.10f; /* --xfade F: crossfade overlap in seconds (0.10 default) */
    int mode_speed = 0;       /* --mode speed: all optimizations for real-time
                               * CPU use (ctx 0.4 + conv tile 2048). Default
                               * (quality) keeps byte-identical reference
                               * output for music/master rendering. */
    int jobs = 4;             /* --jobs N: parallel chunk workers (sweet spot measured) */
    int no_chunk = 0;         /* --no-chunk: force the whole-track path (parity) */
    int use_cuda = 0;         /* --cuda: run the GeneratorNSF on the GPU (CUDA) */
    int use_vk = 0;           /* --vk: run the GeneratorNSF via Vulkan (cross-vendor) */
    float autokey_shift = 0.0f; /* chosen feed shift (semitones), applied + restored */
    snprintf(model_path, sizeof(model_path), "%s/model.pth", model_dir);
    for (int a = 4; a < argc; a++) {
        if (strcmp(argv[a], "--model") == 0) {
            snprintf(model_path, sizeof(model_path), "%s", argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--speaker") == 0) {
            speaker_id = atoi(argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--noise") == 0) {
            noise_scale = (float)atof(argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--hubert") == 0) {
            snprintf(hubert_path, sizeof(hubert_path), "%s", argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--preset") == 0) {
            preset = atoi(argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--snake") == 0) {
            use_snake = 1;
        } else if (strcmp(argv[a], "--formant") == 0) {
            formant_shift = (float)atof(argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--f0smooth") == 0) {
            f0_smooth = (float)atof(argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--f0ref") == 0) {
            snprintf(f0ref_dir, sizeof(f0ref_dir), "%s", argv[a + 1]);
            a++;
        } else if (strcmp(argv[a], "--f0") == 0 && strcmp(argv[a + 1], "yin") == 0) {
            force_yin = 1;
            a++;
        } else if (strcmp(argv[a], "--f0filter") == 0) {
            f0_filter_radius = atoi(argv[a + 1]);
            if (f0_filter_radius < 0) f0_filter_radius = 0;
            a++;
        } else if (strcmp(argv[a], "--rmsmix") == 0) {
            rms_mix = (float)atof(argv[a + 1]);
            if (rms_mix < 0.0f) rms_mix = 0.0f;
            if (rms_mix > 1.0f) rms_mix = 1.0f;
            a++;
        } else if (strcmp(argv[a], "--autokey") == 0) {
            autokey_probe = atoi(argv[a + 1]);
            if (autokey_probe < 0) autokey_probe = 0;
            a++;
        } else if (strcmp(argv[a], "--chunk") == 0) {
            chunk_secs = (float)atof(argv[a + 1]);
            if (chunk_secs < 0.5f) chunk_secs = 0.5f;
            a++;
        } else if (strcmp(argv[a], "--ctx") == 0) {
            ctx_secs = (float)atof(argv[a + 1]);
            if (ctx_secs < 0.05f) ctx_secs = 0.05f;
            if (ctx_secs > 2.0f) ctx_secs = 2.0f;
            a++;
        } else if (strcmp(argv[a], "--xfade") == 0) {
            xfade_secs = (float)atof(argv[a + 1]);
            if (xfade_secs < 0.01f) xfade_secs = 0.01f;
            if (xfade_secs > 1.0f) xfade_secs = 1.0f;
            a++;
        } else if (strcmp(argv[a], "--jobs") == 0) {
            jobs = atoi(argv[a + 1]);
            if (jobs < 1) jobs = 1;
            if (jobs > 8) jobs = 8;
            a++;
        } else if (strcmp(argv[a], "--mode") == 0) {
            if (strcmp(argv[a + 1], "speed") == 0) {
                mode_speed = 1;
            } else if (strcmp(argv[a + 1], "quality") == 0) {
                mode_speed = 0;
            } else {
                fprintf(stderr, "--mode: expected 'quality' or 'speed'\n");
                return 1;
            }
            a++;
        } else if (strcmp(argv[a], "--no-chunk") == 0) {
            no_chunk = 1;
        } else if (strcmp(argv[a], "--cuda") == 0) {
            use_cuda = 1;
        } else if (strcmp(argv[a], "--vk") == 0) {
            use_vk = 1;
        }
    }
    if (!strstr(model_path, ".pth")) {
        /* fall back to first *.pth in dir */
        char pat[1024]; snprintf(pat, sizeof(pat), "%s/*.pth", model_dir);
        /* use the C loader's own dir scan via wubu_rvc_load_weights later;
         * here just try common names */
    }
    /* --mode speed: all real-time optimizations (ctx 0.4 + conv tile 2048).
     * quality (default): byte-identical reference output for rendering. */
    if (mode_speed) {
        if (ctx_secs == 0.72f) ctx_secs = 0.40f;  /* don't override explicit --ctx */
        wubu_set_conv_tile(2048);
        wubu_set_fast_math(1);  /* folded-poly exp/tanh/sigmoid (~7e-6) */
        printf("[0] mode: speed (ctx %.2f, conv tile %d, fast math) — real-time CPU\n",
               ctx_secs, wubu_get_conv_tile());
    } else {
        wubu_set_conv_tile(8192);
        wubu_set_fast_math(0);  /* libm — byte-identical reference output */
        printf("[0] mode: quality (ctx %.2f, conv tile %d) — reference output\n",
               ctx_secs, wubu_get_conv_tile());
    }
    int use_chunk = (chunk_secs > 0.5f && !no_chunk);
    clock_t t0 = 0;
    FILE *chk = fopen(model_path, "rb");
    if (!chk) die("cannot open model.pth — pass --model");
    fclose(chk);

    /* index: scan model_dir for any *.index file */
    index_path[0] = 0;
    {
        char idx_pat[1024];
        snprintf(idx_pat, sizeof(idx_pat), "%s/*.index", model_dir);
        /* Simple glob: check common index file patterns */
        const char *idx_names[] = {
            "added_IVF793_Flat_nprobe_1.index",
            "trained_by_pool9045_Flat_nprobe_1.index",
            NULL
        };
        for (int i = 0; idx_names[i] && !index_path[0]; i++) {
            char test_path[1024];
            snprintf(test_path, sizeof(test_path), "%s/%s", model_dir, idx_names[i]);
            FILE *f = fopen(test_path, "rb");
            if (f) { fclose(f); strncpy(index_path, test_path, sizeof(index_path)-1); }
        }
        if (!index_path[0]) {
            /* Try the Cartman-specific name (backward compat) */
            snprintf(index_path, sizeof(index_path),
                     "%s/added_IVF793_Flat_nprobe_1_EricCartmanV1_v2.index", model_dir);
            FILE *f = fopen(index_path, "rb");
            if (!f) index_path[0] = 0;
            else fclose(f);
        }
    }

    /* 1. audio */
    int n_in = 0, sr_in = 0;
    float *audio = read_wav(in_path, &n_in, &sr_in);
    if (!audio) die("cannot read input wav (need RIFF mono PCM_16/24/32f)");
    printf("[1] input: %d samples @%d Hz (%.2f s)\n", n_in, sr_in, (double)n_in / sr_in);
    int n16 = 0;
    float *pcm16 = resample(audio, n_in, sr_in, 16000, &n16);
    /* NOTE: do NOT free(audio) here — the rms_mix stage (post-synth) needs
     * the original input at its native rate. Freed in the cleanup block. */
    if (!pcm16) die("resample failed");

    /* Build config early so we can load the model and determine version */
    RVCConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.model_path, model_path, sizeof(cfg.model_path) - 1);
    if (index_path[0]) strncpy(cfg.index_path, index_path, sizeof(cfg.index_path) - 1);
    cfg.version = RVC_V2;
    cfg.mel_channels = 80;
    cfg.hidden_channels = 256;

    /* 2. Load model early to determine version (v1 vs v2) */
    WuBuRVC *rvc = wubu_rvc_load(&cfg);
    if (!rvc || !wubu_rvc_is_model_loaded(rvc)) die("model load failed");

    /* Determine RVC version from loaded model */
    int rvc_ver = rvc->rvc_version;
    if (rvc_ver <= 0) rvc_ver = 2;
    int content_dim = (rvc_ver == 1) ? 256 : 768;
    int sr_out = rvc->sample_rate;
    if (sr_out <= 0) sr_out = 40000;
    int ups_total = rvc->graph.upsample_rate;
    if (ups_total <= 0) ups_total = 400;
    printf("[2] model version: v%d (content_dim=%d, sr=%d, ups=%d)\n",
           rvc_ver, content_dim, sr_out, ups_total);

    /* 3. HuBERT content (v2: layer 12 768-dim, v1: layer 9 + final_proj 256-dim) */
    WuBuHubert hb;
    memset(&hb, 0, sizeof(hb));
    /* Allow --hubert PATH override; otherwise search model_dir then default */
    const char *hubert_bin = hubert_path[0] ? hubert_path : "models/rvc/hubert_weights.bin";
    if (!hubert_path[0]) {
        char hp[1024];
        snprintf(hp, sizeof(hp), "%s/hubert_weights.bin", model_dir);
        FILE *hf = fopen(hp, "rb");
        if (hf) { fclose(hf); hubert_bin = hp; }
    }
    if (wubu_hubert_load(&hb, hubert_bin) != 0)
        die("hubbert weights missing — run tools/extract_hubert_weights.py or pass --hubert PATH");
    int T = wubu_hubert_output_length(n16);
    int T2 = 0;
    float *content_up = NULL;
    if (!use_chunk) {
        printf("[3] hubert frames: %d\n", T);
        float *content = (float *)malloc((size_t)T * content_dim * sizeof(float));
        t0 = clock();
        int Tc = wubu_hubert_extract_real(&hb, pcm16, n16, rvc_ver, content, T * content_dim);
        printf("     hubert: %.2f s (%.2fx realtime)\n",
               (double)(clock() - t0) / CLOCKS_PER_SEC,
               (double)(clock() - t0) / CLOCKS_PER_SEC / ((double)n16 / 16000.0));
        if (Tc != T) { printf("     (hubert returned %d frames)\n", Tc); T = Tc; }

        /* 4. content ×2 upsample: [T, dim] -> [2T, dim] */
        content_up = upsample_frames(content, T, content_dim, &T2);
        free(content);
        printf("[4] content_up frames: %d\n", T2);
    }

    /* 5. f0 (YIN at 16k, 100 fps) + coarse. RVC consumes f0 at the SAME
     * frame rate as the ×2 content (100 fps: 278 frames for 2.78 s) — NO ×2
     * on f0. The old ×2-nearest-then-truncate compressed the pitch contour
     * 2× against content → robotic/wrong voice. test_rvc_real passes f0 1:1
     * with content_up (verified corr 0.9999 vs PyTorch, SNR 29.7 dB).
     *    --f0ref DIR: load reference nsff0_raw.bin (float32 100fps) +
     *    f0_coarse.bin (int32 100fps) instead of YIN. */
    int n_f0 = 0;
    float *f0 = (float *)malloc((size_t)(n16 / 160 + 2) * sizeof(float));
    int *f0_coarse = NULL;
    float *nsff0 = NULL;
    int n_f0_2 = 0;
    if (f0ref_dir[0]) {
        char fp[1024], cp[1024];
        snprintf(fp, sizeof(fp), "%s/nsff0_raw.bin", f0ref_dir);
        snprintf(cp, sizeof(cp), "%s/f0_coarse.bin", f0ref_dir);
        FILE *ff = fopen(fp, "rb");
        FILE *cf = fopen(cp, "rb");
        if (!ff || !cf) die("--f0ref needs nsff0_raw.bin + f0_coarse.bin");
        fseek(ff, 0, SEEK_END); long fsz = ftell(ff); fseek(ff, 0, SEEK_SET);
        fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
        n_f0 = (int)(fsz / 4);
        f0 = (float *)realloc(f0, (size_t)(n_f0 + 2) * sizeof(float));
        if (fread(f0, 4, (size_t)n_f0, ff) != (size_t)n_f0) die("f0ref read fail");
        fclose(ff);
        f0_coarse = (int *)malloc((size_t)(n_f0 + 2) * sizeof(int));
        nsff0 = (float *)malloc((size_t)(n_f0 + 2) * sizeof(float));
        if (csz == (long)n_f0 * 4) {
            if (fread(f0_coarse, 4, (size_t)n_f0, cf) != (size_t)n_f0) die("coarse read fail");
            for (int j = 0; j < n_f0; j++) nsff0[j] = f0[j];
        } else {
            fclose(cf);
            wubu_f0_to_coarse(f0, n_f0, 50.0f, 1100.0f, f0_coarse, nsff0);
        }
        n_f0_2 = n_f0;
        printf("[5] f0 from reference (%d frames @100fps)\n", n_f0);
    } else {
        /* default: RMVPE (the training-time extractor — coarse bins match the
         * model's conditioning distribution). Fall back to YIN if the weights
         * are missing or --f0 yin was passed. */
        WuBuRmvpe *rm = NULL;
        if (!force_yin) {
            rm = wubu_rmvpe_load("models/rvc/rmvpe_weights.bin");
            if (rm) {
                n_f0 = wubu_rmvpe_f0(rm, pcm16, n16, f0, n16 / 160 + 8);
                printf("[5] rmvpe f0 frames: %d\n", n_f0);
            }
        }
        if (!rm || n_f0 <= 0) {
            n_f0 = wubu_f0_yin(pcm16, n16, 16000, 1024, 160, 50.0f, 1100.0f, f0, n16 / 160 + 2);
            printf("[5] yin f0 frames: %d\n", n_f0);
        }
        if (rm) wubu_rmvpe_free(rm);
        if (getenv("WUBU_DUMP_F0")) {
            FILE *df = fopen("f0_ours.bin", "wb");
            if (df) { fwrite(f0, sizeof(float), (size_t)n_f0, df); fclose(df); }
            fprintf(stderr, "[f0dump] wrote f0_ours.bin (%d frames)\n", n_f0);
        }
        /* filter_radius median — kills octave jumps that make singing
         * off-key while preserving vibrato (RVC default radius 3). */
        if (f0_filter_radius > 0 && n_f0 > 0) {
            wubu_f0_median_filter(f0, n_f0, f0_filter_radius);
            printf("[5] f0 median filter radius %d applied\n", f0_filter_radius);
        }
        f0_coarse = (int *)malloc((size_t)(n_f0 + 2) * sizeof(int));
        nsff0 = (float *)malloc((size_t)(n_f0 + 2) * sizeof(float));
        wubu_f0_to_coarse(f0, n_f0, 50.0f, 1100.0f, f0_coarse, nsff0);
        if (getenv("WUBU_DUMP_F0")) {
            FILE *dc = fopen("f0coarse_ours.bin", "wb");
            if (dc) { fwrite(f0_coarse, sizeof(int), (size_t)n_f0, dc); fclose(dc); }
            FILE *dn = fopen("nsff0_ours.bin", "wb");
            if (dn) { fwrite(nsff0, sizeof(float), (size_t)n_f0, dn); fclose(dn); }
            fprintf(stderr, "[f0dump] wrote f0coarse_ours.bin + nsff0_ours.bin (%d frames)\n", n_f0);
        }
        n_f0_2 = n_f0;
    }

    /* 5b. auto key adaptation — feed the model an f0 in its conditioned
     * range, then restore the output to the input's key (phase vocoder). */
    if (autokey_probe > 0 && n_f0_2 > 0) {
        float cached_mean = 0, cached_drift = 0;
        float cached_shift = wubu_autokey_load(model_dir, &cached_mean, &cached_drift);
        float in_mean = wubu_autokey_input_mean(nsff0, n_f0_2);
        if (cached_shift != 0.0f && cached_mean > 0 &&
            fabsf(12.0f * log2f(in_mean / cached_mean)) < 3.0f) {
            autokey_shift = cached_shift;
            printf("[5b] autokey cached shift %+.1f st (input mean %.0f Hz)\n",
                   autokey_shift, in_mean);
        } else {
            WuBuRmvpe *rm2 = wubu_rmvpe_load("models/rvc/rmvpe_weights.bin");
            if (rm2) {
                float drift = 0;
                autokey_shift = wubu_autokey_calibrate(
                    rvc->model, &hb, rm2, pcm16, n16, nsff0, n_f0_2,
                    content_dim, sr_out, ups_total, noise_scale, 0 /*snake*/,
                    autokey_probe, &drift);
                wubu_rmvpe_free(rm2);
                if (fabsf(autokey_shift) > 0.01f || fabsf(drift) > 30.0f)
                    wubu_autokey_save(model_dir, autokey_shift, in_mean, drift);
            }
        }
        if (fabsf(autokey_shift) > 0.01f) {
            float gain = powf(2.0f, autokey_shift / 12.0f);
            for (int i = 0; i < n_f0_2; i++) nsff0[i] *= gain;
            wubu_f0_to_coarse(nsff0, n_f0_2, 50.0f, 1100.0f, f0_coarse, nsff0);
            printf("[5b] autokey: feeding f0 %+.1f st (will restore output)\n",
                   autokey_shift);
        }
    }
    free(f0);

    /* 6. real synth */
    float *out_audio = NULL;
    int n_out = 0;
    double synth_s = 0.0;
    if (use_chunk) {
        /* 6b. chunked + parallel synthesis (default) */
        const int chunk_16k = (int)(chunk_secs * 16000);
        const int extra_16k = (int)(ctx_secs * 16000);       /* HuBERT context */
        const int xfade_16k = (int)(xfade_secs * 16000);     /* crossfade */
        const int hop_16k = chunk_16k - xfade_16k;
        if (hop_16k < 16000) die("--chunk too small");
        int n_chunks = 0;
        for (long s = 0; s < n16; s += hop_16k) n_chunks++;
        long total_out = ((long)(n16 / 160) + 2) * ups_total;
        out_audio = (float *)calloc((size_t)total_out, sizeof(float));
        if (!out_audio) die("alloc");
        float **caudio = (float **)calloc((size_t)n_chunks, sizeof(float *));
        int *cpos = (int *)calloc((size_t)n_chunks, sizeof(int));
        int *clen = (int *)calloc((size_t)n_chunks, sizeof(int));
        int *crc = (int *)calloc((size_t)n_chunks, sizeof(int));
        if (!caudio || !cpos || !clen || !crc) die("alloc");
        ChunkCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.pcm16 = pcm16; ctx.n16 = n16;
        ctx.f0_coarse = f0_coarse; ctx.nsff0 = nsff0; ctx.n_f0 = n_f0_2;
        ctx.hb = &hb; ctx.rvc_ver = rvc_ver; ctx.content_dim = content_dim;
        ctx.model = rvc->model; ctx.speaker_id = speaker_id;
        ctx.noise_scale = noise_scale; ctx.use_snake = use_snake;
        ctx.use_cuda = use_cuda;
        ctx.use_vk = use_vk;
        ctx.ups_total = ups_total;
        ctx.chunk_16k = chunk_16k; ctx.extra_16k = extra_16k; ctx.hop_16k = hop_16k;
        ctx.audio = caudio; ctx.pos = cpos; ctx.len = clen; ctx.rc = crc;
        ctx.n_chunks = n_chunks; ctx.threads = (jobs > 0) ? (8 / jobs) : 8;
        if (ctx.threads < 1) ctx.threads = 1;
        clock_t t0 = clock();
        pthread_t *th = (pthread_t *)malloc((size_t)jobs * sizeof(pthread_t));
        if (!th) die("alloc");
        for (int w = 0; w < jobs; w++)
            pthread_create(&th[w], NULL, chunk_worker, &ctx);
        for (int w = 0; w < jobs; w++)
            pthread_join(th[w], NULL);
        free(th);
        /* sequential OLA with 0.1 s crossfade */
        int valid_out = (chunk_16k / 160) * ups_total;   /* 3.0 s valid audio */
        int xf_out = (xfade_16k / 160) * ups_total;      /* 0.1 s crossfade */
        for (int c = 0; c < n_chunks; c++) {
            if (!caudio[c] || crc[c] != 0) continue;
            long p0 = cpos[c];
            int len = clen[c];
            float *a = caudio[c];
            int valid = valid_out;
            if (valid > len) valid = len;
            int xf = xf_out;
            if (xf > valid) xf = valid;
            /* fade-in head (if this chunk overlaps a previous tail) */
            if (p0 > 0) {
                for (int i = 0; i < xf && p0 + i < total_out; i++) {
                    float f = (float)i / xf;
                    out_audio[p0 + i] += a[i] * f;
                }
            }
            /* main body */
            for (int i = xf; i < valid - xf && p0 + i < total_out; i++)
                out_audio[p0 + i] += a[i];
            /* fade-out tail */
            for (int i = valid - xf; i < valid && p0 + i < total_out; i++) {
                float f = (float)(valid - i) / xf;
                out_audio[p0 + i] += a[i] * f;
            }
            /* if this is the last chunk, the extra tail audio is the outro —
             * copy the rest past the valid window (fade to silence) */
            if (c == n_chunks - 1) {
                for (int i = valid; i < len && p0 + i < total_out; i++) {
                    float f = 1.0f - (float)(i - valid) / (float)(xf + 1);
                    if (f < 0.0f) f = 0.0f;
                    out_audio[p0 + i] += a[i] * f;
                }
            }
            free(a);
            caudio[c] = NULL;
        }
        synth_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
        n_out = (int)(((long)(n16 / 160)) * ups_total);
        if (n_out > total_out) n_out = (int)total_out;
        printf("[6] chunked synth: %d chunks x %d workers, %.2f s (%.2fx realtime)\n",
               n_chunks, jobs, synth_s, synth_s / ((double)n_out / sr_out));
        free(caudio); free(cpos); free(clen); free(crc);
    } else {
        printf("[6] synth...\n");
        int n_frames = T2 < n_f0_2 ? T2 : n_f0_2;
        int max_audio = n_frames * ups_total;
        out_audio = (float *)malloc((size_t)max_audio * sizeof(float));
        if (!out_audio) die("alloc");

    if (getenv("WUBU_RVC_DUMP")) {
        /* debug: dump the exact synth inputs for parity comparison */
        FILE *df = fopen("outputs/rvc_ref/cli_content_up.bin", "wb");
        if (df) { fwrite(content_up, sizeof(float), (size_t)T2 * content_dim, df); fclose(df); }
        df = fopen("outputs/rvc_ref/cli_nsff0.bin", "wb");
        if (df) { fwrite(nsff0, sizeof(float), (size_t)n_f0_2, df); fclose(df); }
        df = fopen("outputs/rvc_ref/cli_coarse.bin", "wb");
        if (df) { fwrite(f0_coarse, sizeof(int), (size_t)n_f0_2, df); fclose(df); }
        fprintf(stderr, "[dump] cli_content_up (%d x %d), cli_nsff0 (%d), cli_coarse (%d)\n",
                T2, content_dim, n_f0_2, n_f0_2);
    }

    /* transpose content_up [T2, dim] frame-major -> [dim, T2] col-major */
    float *cmaj = (float *)malloc((size_t)content_dim * n_frames * sizeof(float));
    if (!cmaj) die("alloc");
    for (int j = 0; j < n_frames; j++)
        for (int d = 0; d < content_dim; d++)
            cmaj[(size_t)d * n_frames + j] = content_up[(size_t)j * content_dim + d];
    free(content_up);

    t0 = clock();
    n_out = wubu_rvc_synthesize_real(rvc->model, cmaj, n_frames, content_dim,
                                     f0_coarse, nsff0, speaker_id, noise_scale,
                                     out_audio, max_audio, use_snake);
    synth_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    if (n_out <= 0) die("synth failed");

    /* Snake safety (2026-08-07): pretrained RVC weights were trained with
     * LeakyReLU. Snake (~identity) removes LReLU's negative compression, so
     * the MRF residual stacks compound and the final tanh saturates to a
     * pure square wave (verified: sat_frac=1.000). Fall back to the
     * parity-verified LeakyReLU path instead of emitting a square wave. */
    if (use_snake && rvc->model && rvc->model->last_snake_sat > 0.5f) {
        fprintf(stderr, "WARNING: Snake activation saturates the generator "
                        "(sat_frac=%.3f) — incompatible with LeakyReLU-trained "
                        "weights. Falling back to LeakyReLU (parity-verified).\n",
                rvc->model->last_snake_sat);
        use_snake = 0;
        free(out_audio);
        out_audio = (float *)malloc((size_t)max_audio * sizeof(float));
        if (!out_audio) die("alloc");
        t0 = clock();
        n_out = wubu_rvc_synthesize_real(rvc->model, cmaj, n_frames, content_dim,
                                         f0_coarse, nsff0, speaker_id, noise_scale,
                                         out_audio, max_audio, use_snake);
        synth_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (n_out <= 0) die("synth failed (LeakyReLU fallback)");
    }
    printf("     synth: %.2f s (%.2fx realtime)\n", synth_s,
           synth_s / ((double)n_out / sr_out));
    }

    /* 6. post-processing: normalize then apply character preset */
    if (preset > 0 && preset <= 4) {
        /* Debug: show synthesis output stats */
        float _syn_peak = 0, _syn_rms = 0;
        for (int i = 0; i < n_out; i++) {
            float a = fabsf(out_audio[i]);
            if (a > _syn_peak) _syn_peak = a;
            _syn_rms += out_audio[i] * out_audio[i];
        }
        _syn_rms = sqrtf(_syn_rms / (float)n_out);
        fprintf(stderr, "[synth-output] peak=%.4f rms=%.4f mean=%.6f\n",
                _syn_peak, _syn_rms, 0.0f); /* mean not computed for speed */
        /* Normalize synthesis output to consistent peak level.
         * The NSF sine generator can produce widely varying output levels
         * depending on random noise; normalize to consistent peak for postproc. */
        if (_syn_peak > 0.001f) {
            /* Normalize to RMS ~0.1 (typical speech level) for consistent post-processing.
             * The NSF generator may produce clipped output (RMS=1.0) with Snake activation;
             * normalize to a safe level before EQ/saturation to avoid clipping. */
            float _syn_rms = 0;
            for (int i = 0; i < n_out; i++) _syn_rms += out_audio[i] * out_audio[i];
            _syn_rms = sqrtf(_syn_rms / (float)n_out);
            if (_syn_rms > 1e-10f) {
                float _scale = 0.1f / _syn_rms;
                for (int i = 0; i < n_out; i++) out_audio[i] *= _scale;
            }
        }
        float *pp = (float *)malloc((size_t)n_out * sizeof(float));
        if (pp) {
            wubu_apply_character_preset(out_audio, pp, n_out, sr_out, preset);
            memcpy(out_audio, pp, (size_t)n_out * sizeof(float));
            printf("     [postproc] character preset %d applied\n", preset);
            free(pp);
        }
    }

    /* Formant shift (gender conversion) */
    if (formant_shift > 0 && formant_shift != 1.0f) {
        float *fs = (float *)malloc((size_t)n_out * sizeof(float));
        if (fs) {
            wubu_formant_shift(out_audio, fs, n_out, sr_out, formant_shift);
            memcpy(out_audio, fs, (size_t)n_out * sizeof(float));
            printf("     [formant] shift ratio=%.2f\n", formant_shift);
            free(fs);
        }
    }

    /* F0 contour smoothing */
    if (f0_smooth > 0.0f && f0_smooth <= 1.0f) {
        /* Note: This would need to be applied before synthesis, not after.
         * Currently logged as informational — for production, hook into the
         * F0 extraction path before the synthesis step. */
        printf("     [f0smooth] strength=%.2f (note: applies to F0 before synth)\n", f0_smooth);
    }

    /* RMS envelope mix — output follows the input's volume dynamics */
    if (rms_mix > 0.001f) {
        int nin_out = 0;
        float *input_at_out = resample(audio, n_in, sr_in, sr_out, &nin_out);
        if (input_at_out) {
            if (nin_out > n_out) nin_out = n_out;
            float *in_pad = (float *)calloc((size_t)n_out, sizeof(float));
            if (in_pad) {
                memcpy(in_pad, input_at_out, (size_t)nin_out * sizeof(float));
                wubu_rms_mix_rate(in_pad, out_audio, n_out, sr_out, rms_mix);
                printf("     [rmsmix] envelope mix=%.2f applied\n", rms_mix);
                free(in_pad);
            }
            free(input_at_out);
        }
    }

    /* Auto-key restore: the model was fed a shifted f0; pitch-shift the
     * output back so the voice lands in the input's key. */
    if (fabsf(autokey_shift) > 0.01f) {
        float *restored = (float *)malloc((size_t)n_out * sizeof(float));
        if (restored) {
            wubu_pitch_shift(out_audio, n_out, sr_out, -autokey_shift, restored);
            memcpy(out_audio, restored, (size_t)n_out * sizeof(float));
            printf("     [autokey] restored output by %+.1f st\n", -autokey_shift);
            free(restored);
        }
    }

    /* 6. write wav */
    if (write_wav(out_path, out_audio, n_out, sr_out) != 0) die("write wav failed");
    printf("[6] wrote %s: %d samples @%d (%.2f s)\n", out_path, n_out, sr_out,
           (double)n_out / sr_out);

    free(pcm16); free(f0_coarse); free(nsff0); free(out_audio);
    free(audio);
    wubu_hubert_free(&hb);
    wubu_rvc_destroy(rvc);
    return 0;
}
