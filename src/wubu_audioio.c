/* wubu_audioio.c — robust WAV read/write + resample (C11).
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_audioio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── internal: parse one WAV into interleaved float32 (stereo) ──
 * Handles PCM 16/24/32-int, IEEE float, extensible. Returns channels (1/2)
 * or 0 on failure. */
static int wav_parse(const unsigned char *d, long dlen,
                     float **out, int *n_out, int *sr_out) {
    if (dlen < 12 || memcmp(d, "RIFF", 4) != 0 || memcmp(d + 8, "WAVE", 4) != 0)
        return 0;
    long pos = 12;
    int fmt_tag = 0, ch = 0, sr = 0, bits = 0, data_off = -1, data_len = 0;
    while (pos + 8 <= dlen) {
        unsigned int sz = (unsigned int)(d[pos + 4] | (d[pos + 5] << 8) |
                                         (d[pos + 6] << 16) | (d[pos + 7] << 24));
        if (memcmp(d + pos, "fmt ", 4) == 0 && sz >= 16) {
            fmt_tag = d[pos + 8] | (d[pos + 9] << 8);
            ch = d[pos + 10] | (d[pos + 11] << 8);
            sr = (int)((unsigned int)(d[pos + 12] | (d[pos + 13] << 8) |
                                      (d[pos + 14] << 16) | (d[pos + 15] << 24)));
            /* fmt data layout: tag(8) ch(10) sr(12) br(16) align(20) bits(22) */
            bits = d[pos + 22] | (d[pos + 23] << 8);
        } else if (memcmp(d + pos, "data", 4) == 0) {
            data_off = (int)(pos + 8);
            data_len = (int)sz;
        }
        pos += 8 + sz + (sz & 1);
    }
    if (data_off < 0 || ch < 1 || ch > 2 || sr <= 0) return 0;
    /* extensible: real format is in the SubFormat GUID (bytes 26..30 of fmt) */
    if (fmt_tag == 0xFFFE) fmt_tag = 3; /* assume float; bits check below */

    long nf = data_len / (ch * (bits / 8));
    if (nf <= 0) return 0;
    float *buf = (float *)malloc((size_t)nf * ch * sizeof(float));
    if (!buf) return 0;

    if (fmt_tag == 3 && bits == 32) {
        for (long i = 0; i < nf * ch; i++) {
            const unsigned char *p = d + data_off + (size_t)i * 4;
            unsigned int u = (unsigned int)(p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned int)p[3] << 24));
            float f;
            memcpy(&f, &u, 4);
            buf[i] = f;
        }
    } else if (bits == 16) {
        for (long i = 0; i < nf * ch; i++) {
            const unsigned char *p = d + data_off + (size_t)i * 2;
            short s = (short)(p[0] | (p[1] << 8));
            buf[i] = (float)s / 32768.0f;
        }
    } else if (bits == 24) {
        for (long i = 0; i < nf * ch; i++) {
            const unsigned char *p = d + data_off + (size_t)i * 3;
            int v = p[0] | (p[1] << 8) | (p[2] << 16);
            if (v & 0x800000) v |= ~0xFFFFFF;
            buf[i] = (float)v / 8388608.0f;
        }
    } else if (bits == 32 && fmt_tag == 1) {
        for (long i = 0; i < nf * ch; i++) {
            const unsigned char *p = d + data_off + (size_t)i * 4;
            int v = (int)((unsigned int)(p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned int)p[3] << 24)));
            buf[i] = (float)v / 2147483648.0f;
        }
    } else {
        free(buf);
        return 0;
    }
    *out = buf;
    *n_out = (int)nf;
    *sr_out = sr;
    return ch;
}

WuBuAudio *wubu_audio_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 12) { fclose(f); return NULL; }
    unsigned char *d = (unsigned char *)malloc((size_t)fsize);
    if (!d) { fclose(f); return NULL; }
    if (fread(d, 1, (size_t)fsize, f) != (size_t)fsize) { free(d); fclose(f); return NULL; }
    fclose(f);

    float *buf = NULL;
    int n = 0, sr = 0;
    int ch = wav_parse(d, fsize, &buf, &n, &sr);
    free(d);
    if (ch <= 0 || !buf) return NULL;
    WuBuAudio *a = (WuBuAudio *)calloc(1, sizeof(WuBuAudio));
    if (!a) { free(buf); return NULL; }
    a->n = n;
    a->sr = sr;
    a->data = buf; /* interleaved ch */
    if (ch == 2) {
        /* take left channel */
        float *mono = (float *)malloc((size_t)n * sizeof(float));
        if (!mono) { free(buf); free(a); return NULL; }
        for (int i = 0; i < n; i++) mono[i] = buf[i * 2];
        free(buf);
        a->data = mono;
    }
    return a;
}

void wubu_audio_free(WuBuAudio *a) {
    if (!a) return;
    free(a->data);
    free(a);
}

WuBuAudio *wubu_audio_read_stereo(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 12) { fclose(f); return NULL; }
    unsigned char *d = (unsigned char *)malloc((size_t)fsize);
    if (!d) { fclose(f); return NULL; }
    if (fread(d, 1, (size_t)fsize, f) != (size_t)fsize) { free(d); fclose(f); return NULL; }
    fclose(f);

    float *buf = NULL;
    int n = 0, sr = 0;
    int ch = wav_parse(d, fsize, &buf, &n, &sr);
    free(d);
    if (ch <= 0 || !buf) return NULL;
    WuBuAudio *a = (WuBuAudio *)calloc(1, sizeof(WuBuAudio));
    if (!a) { free(buf); return NULL; }
    a->n = n;
    a->sr = sr;
    if (ch == 1) {
        float *st = (float *)malloc((size_t)n * 2 * sizeof(float));
        if (!st) { free(buf); free(a); return NULL; }
        for (int i = 0; i < n; i++) { st[i * 2] = buf[i]; st[i * 2 + 1] = buf[i]; }
        free(buf);
        a->data = st;
    } else {
        a->data = buf;
    }
    return a;
}

int wubu_audio_write(const char *path, const float *data, int n, int sr, int stereo) {
    if (!path || !data || n <= 0 || sr <= 0) return -1;
    int ch = stereo ? 2 : 1;
    unsigned int data_bytes = (unsigned int)n * ch * 2;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* RIFF header (PCM 16) */
    fwrite("RIFF", 1, 4, f);
    unsigned int riff = 36 + data_bytes;
    fwrite(&riff, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    unsigned int fmt_sz = 16;
    unsigned short fmt_tag = 1, nch = (unsigned short)ch;
    unsigned int srate = (unsigned int)sr, brate = (unsigned int)sr * ch * 2;
    unsigned short balign = (unsigned short)(ch * 2), bbits = 16;
    fwrite(&fmt_sz, 4, 1, f);
    fwrite(&fmt_tag, 2, 1, f);
    fwrite(&nch, 2, 1, f);
    fwrite(&srate, 4, 1, f);
    fwrite(&brate, 4, 1, f);
    fwrite(&balign, 2, 1, f);
    fwrite(&bbits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    for (int i = 0; i < n * ch; i++) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        short s = (short)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

int wubu_audio_resample(const float *in, int n, int in_sr, int out_sr, float *out) {
    if (!in || !out || n <= 0 || in_sr <= 0 || out_sr <= 0) return -1;
    if (in_sr == out_sr) {
        memcpy(out, in, (size_t)n * sizeof(float));
        return n;
    }
    double ratio = (double)out_sr / (double)in_sr;
    int n_out = (int)(n * ratio);
    if (n_out < 1) n_out = 1;
    for (int i = 0; i < n_out; i++) {
        double pos = (double)i / ratio;
        int i0 = (int)pos;
        int i1 = i0 + 1 < n ? i0 + 1 : i0;
        double frac = pos - i0;
        out[i] = (float)(in[i0] + (in[i1] - in[i0]) * frac);
    }
    return n_out;
}

int wubu_audio_slice(const float *in, int n, int start, int len, float *out) {
    if (!in || !out || start < 0) return 0;
    int copied = 0;
    for (int i = start; i < n && copied < len; i++) out[copied++] = in[i];
    return copied;
}

void wubu_audio_mix_pan(const float *src, int n, float gain, float pan,
                        float *l, float *r) {
    if (!src || !l || !r) return;
    if (pan < -1.0f) pan = -1.0f;
    if (pan > 1.0f) pan = 1.0f;
    float lg = (pan <= 0.0f) ? 1.0f : (1.0f - pan);
    float rg = (pan >= 0.0f) ? 1.0f : (1.0f + pan);
    for (int i = 0; i < n; i++) {
        l[i] += src[i] * gain * lg;
        r[i] += src[i] * gain * rg;
    }
}

/* ── windowed-sinc (Kaiser) resampler ── */
#define WUBU_SINC_TAPS 64
static double kaiser_bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 32; k++) {
        term *= (x * x) / (4.0 * k * k);
        sum += term;
        if (term < 1e-18) break;
    }
    return sum;
}

int wubu_audio_resample_sinc(const float *in, int n, int in_sr, int out_sr,
                             float *out) {
    if (!in || !out || n <= 0 || in_sr <= 0 || out_sr <= 0) return -1;
    if (in_sr == out_sr) {
        memcpy(out, in, (size_t)n * sizeof(float));
        return n;
    }
    double ratio = (double)out_sr / (double)in_sr;
    int n_out = (int)(n * ratio);
    if (n_out < 1) n_out = 1;
    const double beta = 14.8;
    const double i0beta = kaiser_bessel_i0(beta);
    const int M = WUBU_SINC_TAPS;
    const double half = M / 2.0;
    /* cutoff = min(1, 1/ratio) * 0.95 — anti-alias */
    const double cutoff = (ratio < 1.0 ? ratio : 1.0) * 0.95;
    for (int i = 0; i < n_out; i++) {
        double pos = (double)i / ratio;   /* input-space position */
        int center = (int)pos;
        double frac = pos - center;
        double acc = 0.0, wsum = 0.0;
        for (int j = 0; j < M; j++) {
            int idx = center - (int)half + j;
            if (idx < 0 || idx >= n) continue;
            double t = (double)(j - half + frac); /* distance from sinc center */
            double x = M_PI * t * cutoff;
            double sinc = (fabs(t) < 1e-9) ? cutoff : cutoff * sin(x) / x;
            double win = kaiser_bessel_i0(beta * sqrt(1.0 - (t * t) / (half * half)))
                         / i0beta;
            acc += in[idx] * sinc * win;
            wsum += sinc * win;
        }
        out[i] = (wsum > 1e-12) ? (float)(acc / wsum) : 0.0f;
    }
    return n_out;
}
