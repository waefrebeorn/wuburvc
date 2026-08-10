/* tt_cvec.c — probe: does the C11 loader read ContentVec weights and do the
 * features differ from HuBERT? Loads both bins, extracts content on a short
 * clip, prints first frames. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_rvc_hubert.h"

int main(int argc, char **argv) {
    const char *clip = argc > 1 ? argv[1] : "C:/Users/eman5/wuburvc/out/demo/sb/cig_dry_15s.wav";
    /* load wav via raw read: 16-bit mono */
    FILE *f = fopen(clip, "rb");
    if (!f) { fprintf(stderr, "no clip\n"); return 1; }
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, f) != 44) return 1;
    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int ch = hdr[22] | (hdr[23] << 8);
    int n = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    n /= 2; /* bytes -> samples (16-bit) */
    short *pcm = (short *)malloc((size_t)n * 2);
    if (fread(pcm, 2, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    float *pcmf = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) pcmf[i] = pcm[i] / 32768.0f;
    int n16 = n; /* assume 32k -> treat directly at 16k-ish for a short probe */

    const char *bins[2] = {
        "C:/Users/eman5/WuBuMedia/models/rvc/hubert_weights.bin",
        "C:/Users/eman5/WuBuMedia/models/rvc/contentvec_weights.bin"
    };
    for (int bi = 0; bi < 2; bi++) {
        WuBuHubert h;
        memset(&h, 0, sizeof(h));
        int rc = wubu_hubert_load(&h, bins[bi]);
        printf("[%d] load %s rc=%d\n", bi, bins[bi], rc);
        if (rc != 0) continue;
        int T = wubu_hubert_output_length(n16);
        float *feat = (float *)malloc((size_t)T * 768 * 4);
        int got = wubu_hubert_extract_real(&h, pcmf, n16, 2, feat, (size_t)T * 768);
        printf("    frames=%d got=%d first8: ", T, got);
        for (int i = 0; i < 8; i++) printf("%.4f ", feat[i]);
        printf("\n");
        free(feat);
        wubu_hubert_free(&h);
    }
    free(pcm); free(pcmf);
    return 0;
}
