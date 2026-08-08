/* test_rvc_real.c — REAL RVC parity test: WuBuRVC C11 vs PyTorch reference.
 *
 * Loads the Cartman checkpoint through the real pipeline:
 *   1. C HuBERT content encoder (wubu_rvc_hubert.c) on real PCM
 *      -> compared against reference content.npy (torch HuBERT)
 *   2. Real acoustic model (wubu_rvc_real.c: enc_p + flow + GeneratorNSF)
 *      with reference f0/coarse
 *      -> compared against reference output_audio.npy (Mangio infer)
 *
 * Reference files come from tools/gen_reference_real.py.
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_rvc.h"
#include "wubu_rvc_parity.h"
#include "wubu_rvc_real.h"
#include "wubu_rvc_hubert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Minimal .npy float32 loader (v1.0/v2.0 headers) */
static float *load_npy(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned char ver[2];
    fread(ver, 1, 2, f);
    unsigned int hlen;
    if (ver[0] == 1) {
        unsigned short hl; fread(&hl, 2, 1, f); hlen = hl;
    } else {
        fread(&hlen, 4, 1, f);
    }
    fseek(f, hlen, SEEK_CUR);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen, SEEK_SET);
    long dsize = fsize - (6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen);
    int n = (int)(dsize / sizeof(float));
    if (n <= 0) { fclose(f); return NULL; }
    float *data = (float *)malloc((size_t)dsize);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, sizeof(float), (size_t)n, f) != (size_t)n) { free(data); fclose(f); return NULL; }
    fclose(f);
    *n_out = n;
    return data;
}

static int *load_npy_int(const char *path, int *n_out) {
    /* load int64 npy as int */
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned char ver[2]; fread(ver, 1, 2, f);
    unsigned int hlen;
    if (ver[0] == 1) { unsigned short hl; fread(&hl, 2, 1, f); hlen = hl; }
    else { fread(&hlen, 4, 1, f); }
    fseek(f, hlen, SEEK_CUR);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen, SEEK_SET);
    long dsize = fsize - (6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen);
    int n = (int)(dsize / 8);
    if (n <= 0) { fclose(f); return NULL; }
    long long *tmp = (long long *)malloc((size_t)dsize);
    if (!tmp) { fclose(f); return NULL; }
    if (fread(tmp, 8, (size_t)n, f) != (size_t)n) { free(tmp); fclose(f); return NULL; }
    fclose(f);
    int *out = (int *)malloc((size_t)n * sizeof(int));
    if (!out) { free(tmp); return NULL; }
    for (int i = 0; i < n; i++) out[i] = (int)tmp[i];
    free(tmp);
    *n_out = n;
    return out;
}

static void stats(const float *a, int n, double *mean, double *std, double *rms) {
    double s = 0, sq = 0;
    for (int i = 0; i < n; i++) { s += a[i]; sq += a[i] * a[i]; }
    *mean = s / n;
    *rms = sqrt(sq / n);
    double v = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - *mean; v += d * d; }
    *std = sqrt(v / n);
}

int main(void) {
    const char *REF = "outputs/rvc_ref";
    char path[600];

    printf("=== WuBuRVC REAL Parity Test (C11 vs PyTorch reference) ===\n\n");

    /* 1. Load model */
    RVCConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.model_path, "models/rvc/cartman/EricCartmanV1_e650_s10400.pth",
            sizeof(cfg.model_path) - 1);
    strncpy(cfg.index_path, "models/rvc/cartman/added_IVF793_Flat_nprobe_1_EricCartmanV1_v2.index",
            sizeof(cfg.index_path) - 1);
    cfg.version = RVC_V2;
    cfg.sample_rate = 40000;
    cfg.mel_channels = 80;
    cfg.hidden_channels = 256;

    WuBuRVC *rvc = wubu_rvc_load(&cfg);
    if (!rvc || !wubu_rvc_is_model_loaded(rvc)) {
        printf("FAIL: model not loaded\n");
        return 1;
    }
    printf("[1] model loaded (%d tensors, hidden=%d)\n",
           rvc->model->n_tensors, rvc->model->hidden_channels);

    /* 2. HuBERT content */
    printf("\n[2] HuBERT content encoder (C11) vs torch reference...\n");
    WuBuHubert hb;
    memset(&hb, 0, sizeof(hb));
    if (wubu_hubert_load(&hb, "models/rvc/hubert_weights.bin") != 0) {
        printf("FAIL: hubert weights not loaded\n");
        return 1;
    }
    int npcm = 0;
    float *pcm = load_npy("outputs/rvc_ref/pcm16k.npy", &npcm);
    if (!pcm) { printf("FAIL: pcm16k.npy missing — run tools/gen_reference_real.py first\n"); return 1; }
    printf("     pcm16k: %d samples\n", npcm);

    int n_ref_content = 0;
    float *ref_content = load_npy("outputs/rvc_ref/content.npy", &n_ref_content);
    int T_hubert = n_ref_content / 768;
    if (!ref_content || T_hubert < 1) { printf("FAIL: content.npy missing\n"); return 1; }
    printf("     reference content: %d frames x 768\n", T_hubert);

    float *c_content = (float *)malloc((size_t)T_hubert * 768 * sizeof(float));
    clock_t t0 = clock();
    int T_c = wubu_hubert_extract_real(&hb, pcm, npcm, 2, c_content,
                                        T_hubert * 768);
    double hubert_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("     HuBERT extract: %.2f s (%.2fx realtime for %.2f s audio)\n",
           hubert_s, hubert_s / ((double)npcm / 16000.0), (double)npcm / 16000.0);
    if (T_c != T_hubert) {
        printf("     WARN: C frames %d vs ref %d (shapes differ; truncating)\n", T_c, T_hubert);
        if (T_c < T_hubert) T_hubert = T_c;
    }
    /* compare */
    double max_d = 0, sum_d = 0;
    for (int i = 0; i < T_hubert * 768; i++) {
        double d = fabs(c_content[i] - ref_content[i]);
        if (d > max_d) max_d = d;
        sum_d += d;
    }
    double mean_d = sum_d / (T_hubert * 768);
    double cm, cs, cr, rm, rs, rr;
    stats(c_content, T_hubert * 768, &cm, &cs, &cr);
    stats(ref_content, T_hubert * 768, &rm, &rs, &rr);
    printf("     C11   content: mean=%.5f std=%.5f rms=%.5f\n", cm, cs, cr);
    printf("     torch content: mean=%.5f std=%.5f rms=%.5f\n", rm, rs, rr);
    printf("     max abs diff=%.5f mean abs diff=%.5f\n", max_d, mean_d);
    printf("     HUBERT PARITY: %s\n",
           max_d < 1e-3 ? "PASS (max diff < 1e-3)"
                        : (max_d < 1e-2 ? "PASS (max diff < 1e-2)"
                                        : "NEEDS_WORK"));

    /* 3. Load f0 + coarse from reference (real RMVPE) */
    int n_coarse = 0, n_nsff0 = 0, n_up = 0;
    int *f0_coarse = load_npy_int("outputs/rvc_ref/f0_coarse.npy", &n_coarse);
    if (!f0_coarse) {
        printf("FAIL: f0_coarse.npy\n"); return 1;
    }
    float *nsff0 = load_npy("outputs/rvc_ref/nsff0.npy", &n_nsff0);
    float *content_up = load_npy("outputs/rvc_ref/content_up.npy", &n_up);
    if (!nsff0 || !content_up) { printf("FAIL: nsff0/content_up\n"); return 1; }
    printf("\n[3] f0: %d frames (RMVPE), content_up %d frames (%d dim)\n",
           n_coarse, n_up / 768, 768);

    /* content_up.npy is [T, dim] FRAME-major (row-major). The C engine
     * expects content as [dim, T] COLUMN-major ([channels, time], matching
     * PyTorch (B,C,T) with B=1). Transpose here so the synthesizer reads
     * the same content as the torch reference. */
    {
        int Tf = n_up / 768;
        float *cmaj = (float *)malloc((size_t)n_up * sizeof(float));
        if (!cmaj) { printf("FAIL: cmaj alloc\n"); return 1; }
        for (int j = 0; j < Tf; j++)
            for (int d = 0; d < 768; d++)
                cmaj[(size_t)d * Tf + j] = content_up[(size_t)j * 768 + d];
        free(content_up);
        content_up = cmaj;
        /* probe: after transpose, content_up[d*Tf+0] should equal orig frame 0
         * (duplicated frames 0==1 in the reference), so cmaj[0]==cmaj[278] */
        printf("     [probe] cmaj[0]=%.6f cmaj[1]=%.6f cmaj[278]=%.6f\n",
               cmaj[0], cmaj[1], cmaj[278]);
    }

    /* 4. Real acoustic model */
    printf("\n[4] Real acoustic model (enc_p + flow + GeneratorNSF)...\n");
    int n_frames = n_up / 768;
    if (n_frames > n_coarse) n_frames = n_coarse;
    int max_audio = n_frames * 400;
    float *audio = (float *)malloc((size_t)max_audio * sizeof(float));
    if (!audio) return 1;

    t0 = clock();
    /* use_snake=0 for parity (match original LeakyReLU training) */
    int n_out = wubu_rvc_synthesize_real(rvc->model, content_up, n_frames, 768,
                                         f0_coarse, nsff0, 0, 0.0f,
                                         audio, max_audio, 0);
    double synth_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("     synth: %.2f s (%.2fx realtime for %.2f s audio)\n",
           synth_s, synth_s / ((double)n_out / 40000.0), (double)n_out / 40000.0);
    if (n_out <= 0) {
        printf("FAIL: synthesize_real returned %d\n", n_out);
        return 1;
    }
    printf("     C11 output: %d samples\n", n_out);

    /* 5. Compare vs reference audio */
    int n_ref = 0;
    float *ref_audio = load_npy("outputs/rvc_ref/output_audio.npy", &n_ref);
    if (!ref_audio) { printf("FAIL: output_audio.npy\n"); return 1; }
    int ncmp = n_out < n_ref ? n_out : n_ref;
    max_d = 0; sum_d = 0;
    for (int i = 0; i < ncmp; i++) {
        double d = fabs(audio[i] - ref_audio[i]);
        if (d > max_d) max_d = d;
        sum_d += d;
    }
    double mean_d2 = sum_d / ncmp;
    stats(audio, ncmp, &cm, &cs, &cr);
    stats(ref_audio, ncmp, &rm, &rs, &rr);
    /* SNR = 20*log10(rms_ref / rms_error); rms_error ~ mean abs * sqrt(pi/2)
     * for Gaussian noise, but we compute exact rms of the diff. */
    double err_sq = 0;
    for (int i = 0; i < ncmp; i++) {
        double d = audio[i] - ref_audio[i];
        err_sq += d * d;
    }
    double err_rms = sqrt(err_sq / ncmp);
    double snr_db = 20.0 * log10(rr / (err_rms + 1e-12));
    printf("\n=== Comparison (same content+f0, deterministic) ===\n");
    printf("WuBuRVC C11  : mean=%.6f std=%.6f rms=%.6f\n", cm, cs, cr);
    printf("PyTorch ref  : mean=%.6f std=%.6f rms=%.6f\n", rm, rs, rr);
    printf("Max abs diff: %.8f\n", max_d);
    printf("Mean abs diff: %.8f\n", mean_d2);
    printf("Error RMS: %.8f   SNR: %.2f dB\n", err_rms, snr_db);
    printf("SYNTH PARITY: %s\n",
           (snr_db >= 25.0) ? "PASS (SNR >= 25 dB; float32 accumulation)"
                            : "NEEDS_WORK");

    /* write C11 wav for listening */
    {
        FILE *wf = fopen("outputs/rvc_ref/output_c11.wav", "wb");
        if (wf) {
            /* 16-bit PCM mono 40k */
            unsigned char hdr[44];
            int sr = 40000, ch = 1, bps = 16;
            int data_bytes = n_out * 2;
            memset(hdr, 0, 44);
            memcpy(hdr, "RIFF", 4);
            unsigned int riffsz = 36 + data_bytes;
            memcpy(hdr + 4, &riffsz, 4);
            memcpy(hdr + 8, "WAVE", 4);
            memcpy(hdr + 12, "fmt ", 4);
            unsigned int fmtsz = 16;
            memcpy(hdr + 16, &fmtsz, 4);
            unsigned short fmt = 1, nch = ch;
            unsigned int rate = sr, byterate = sr * ch * bps / 8;
            unsigned short align = ch * bps / 8, bits = bps;
            memcpy(hdr + 20, &fmt, 2);
            memcpy(hdr + 22, &nch, 2);
            memcpy(hdr + 24, &rate, 4);
            memcpy(hdr + 28, &byterate, 4);
            memcpy(hdr + 32, &align, 2);
            memcpy(hdr + 34, &bits, 2);
            memcpy(hdr + 36, "data", 4);
            memcpy(hdr + 40, &data_bytes, 4);
            fwrite(hdr, 1, 44, wf);
            for (int i = 0; i < n_out; i++) {
                short s = (short)(audio[i] * 32767.0f);
                fwrite(&s, 2, 1, wf);
            }
            fclose(wf);
            printf("Wrote outputs/rvc_ref/output_c11.wav (%d samples @40k)\n", n_out);
        }
    }

    free(audio); free(ref_audio); free(pcm); free(ref_content); free(c_content);
    free(f0_coarse); free(nsff0); free(content_up);
    wubu_hubert_free(&hb);
    wubu_rvc_destroy(rvc);
    printf("\n=== DONE ===\n");
    return 0;
}
