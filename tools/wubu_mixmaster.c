/* wubu_mixmaster.c — stem mixer + mastering CLI (C11).
 *
 * Usage:
 *   wubu_mixmaster <out.wav> <sr> <stem:gain:pan> [stem:gain:pan ...]
 *
 * Each stem is a mono or stereo WAV. Mono stems are panned (pan -1..1,
 * 0 = center). All stems are summed to a stereo bus, then run through the
 * default mastering chain (EQ -> compressor -> saturation -> width ->
 * limiter -> loudness RMS -18, true-peak -1). Output: PCM16 stereo WAV.
 *
 * Examples:
 *   wubu_mixmaster master.wav 48000 drums.wav:1.0:0 bass.wav:1.0:0 \
 *     guitar.wav:0.8:0.1 cartman_vocal.wav:1.0:0
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_audioio.h"
#include "wubu_master.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *m) { fprintf(stderr, "wubu_mixmaster: %s\n", m); exit(1); }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <out.wav> <sr> <stem:gain:pan> [...]\n", argv[0]);
        return 1;
    }
    const char *out_path = argv[1];
    int sr = atoi(argv[2]);
    if (sr <= 0) die("bad sample rate");

    int n_stems = argc - 3;
    /* first pass: load all stems to find max length */
    WuBuAudio **stems = (WuBuAudio **)calloc((size_t)n_stems, sizeof(WuBuAudio *));
    float *gains = (float *)calloc((size_t)n_stems, sizeof(float));
    float *pans = (float *)calloc((size_t)n_stems, sizeof(float));
    int max_n = 0;
    if (!stems || !gains || !pans) die("alloc");
    for (int i = 0; i < n_stems; i++) {
        char tmp[1024];
        snprintf(tmp, sizeof(tmp), "%s", argv[i + 3]);
        char *colon = strchr(tmp, ':');
        if (!colon) die("stem must be path:gain:pan");
        *colon = 0;
        char *gstr = colon + 1;
        char *pstr = strchr(gstr, ':');
        if (pstr) *pstr = 0;
        gains[i] = (float)atof(gstr);
        pans[i] = pstr ? (float)atof(pstr + 1) : 0.0f;
        stems[i] = wubu_audio_read_stereo(tmp);
        if (!stems[i]) die("cannot read stem");
        if (stems[i]->sr != sr) {
            /* resample interleaved stereo to the bus rate */
            int n_out = (int)((double)stems[i]->n * sr / stems[i]->sr);
            if (n_out < 1) n_out = 1;
            float *tmpbuf = (float *)malloc((size_t)n_out * 2 * sizeof(float));
            if (!tmpbuf) die("alloc");
            float *mono_in = (float *)malloc((size_t)stems[i]->n * sizeof(float));
            if (!mono_in) die("alloc");
            for (int j = 0; j < stems[i]->n; j++) mono_in[j] = stems[i]->data[j * 2];
            float *mono_out = (float *)malloc((size_t)n_out * sizeof(float));
            if (!mono_out) die("alloc");
            wubu_audio_resample(mono_in, stems[i]->n, stems[i]->sr, sr, mono_out);
            for (int j = 0; j < n_out; j++) {
                tmpbuf[j * 2] = mono_out[j];
                tmpbuf[j * 2 + 1] = mono_out[j];
            }
            free(mono_in); free(mono_out);
            free(stems[i]->data);
            stems[i]->data = tmpbuf;
            stems[i]->n = n_out;
            stems[i]->sr = sr;
        }
        if (stems[i]->n > max_n) max_n = stems[i]->n;
    }

    /* mix bus (interleaved stereo) */
    float *bus = (float *)calloc((size_t)max_n * 2, sizeof(float));
    if (!bus) die("alloc");
    for (int i = 0; i < n_stems; i++) {
        const float *d = stems[i]->data; /* interleaved ch */
        int ch = 2; /* read_stereo always returns interleaved 2ch */
        (void)ch;
        float lg = gains[i] * (pans[i] <= 0.0f ? 1.0f : (1.0f - pans[i]));
        float rg = gains[i] * (pans[i] >= 0.0f ? 1.0f : (1.0f + pans[i]));
        for (int j = 0; j < stems[i]->n; j++) {
            bus[j * 2] += d[j * 2] * lg;
            bus[j * 2 + 1] += d[j * 2 + 1] * rg;
        }
        wubu_audio_free(stems[i]);
    }
    free(stems); free(gains); free(pans);

    /* master */
    WuBuMasterOpts opts;
    wubu_master_default(&opts);
    wubu_master_process(&opts, bus, max_n, sr);

    wubu_audio_write(out_path, bus, max_n, sr, 1);
    free(bus);
    printf("OK: %s (%d samples, %d Hz, stereo)\n", out_path, max_n, sr);
    return 0;
}
