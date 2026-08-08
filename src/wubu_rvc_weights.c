/* wubu_rvc_weights.c — Flat-binary .pth weight loader for WuBuRVC.
 *
 * Reads the WUBU binary format produced by tools/extract_rvc_weights.py:
 *   [4B] magic "WUBU"
 *   [4B] n_tensors (uint32)
 *   per tensor: [1B name_len] [name] [4B n_dims] [dims...] [4B data_len] [data]
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_rvc_parity.h"
#include "wubu_rvc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Load a WUBU-format binary weight file and map tensors into the model.
 * Returns 0 on success, -1 on failure. */
int wubu_rvc_load_weights(WuBuRVCModel *model, const char *bin_path) {
    if (!model || !bin_path) return -1;

    FILE *f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "WuBuRVC: cannot open weight file %s\n", bin_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);

    /* Verify magic */
    if (memcmp(buf, "WUBU", 4) != 0) {
        free(buf);
        fprintf(stderr, "WuBuRVC: bad magic in %s (not a WUBU weight file)\n", bin_path);
        return -1;
    }

    uint32_t n_tensors = read_u32(buf + 4);

    /* Allocate tensor map */
    model->tensors = (RVCTensor *)calloc((size_t)n_tensors, sizeof(RVCTensor));
    model->n_tensors = (int)n_tensors;
    if (!model->tensors) { free(buf); return -1; }

    size_t off = 8;
    int stored = 0;  /* index into model->tensors[] (separate from loop counter) */

    for (uint32_t t = 0; t < n_tensors && off < (size_t)fsize; t++) {
        uint8_t name_len = buf[off]; off += 1;
        if (off + name_len > (size_t)fsize) break;

        char name[256];
        memcpy(name, buf + off, name_len);
        name[name_len < 255 ? name_len : 255] = '\0';
        off += name_len;

        uint32_t n_dims = read_u32(buf + off); off += 4;
        if (off + n_dims * 4 > (size_t)fsize) break;

        int dims[4] = {0,0,0,0};
        uint32_t total = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            dims[d] = (int)read_u32(buf + off); off += 4;
            total *= dims[d];
        }

        uint32_t data_len = read_u32(buf + off); off += 4;
        if (off + data_len > (size_t)fsize) break;

        /* Allocate + copy tensor data */
        float *data = (float *)malloc((size_t)data_len);
        if (!data) { off += data_len; continue; }
        memcpy(data, buf + off, data_len);
        off += data_len;

        float *wp = data;

        /* Map dec.ups.N.weight_v → hifi_upsample[N] (the actual conv weight) */
        if (strstr(name, "dec.ups.") && strstr(name, "weight_v")) {
            for (int i = 0; i < 4; i++) {
                char pat[64];
                snprintf(pat, sizeof(pat), "dec.ups.%d.weight_v", i);
                if (strstr(name, pat) != NULL) {
                    model->hifi_upsample[i] = wp;
                    model->hifi_upsample_len[i] = (int)total;
                    break;
                }
            }
        }
        if (strstr(name, "dec.resblocks.0.convs1.0.weight_v")) {
            if (model->hifi_mrf0 == NULL) {
                model->hifi_mrf0 = wp;
                model->hifi_mrf0_len = (int)total;
            }
        }
        if (strstr(name, "dec.resblocks.0.convs2.0.weight_v")) {
            if (model->hifi_mrf1 == NULL) {
                model->hifi_mrf1 = wp;
                model->hifi_mrf1_len = (int)total;
            }
        }
        if (strstr(name, "noise_convs.0.weight")) {
            if (model->vocoder_conv_pre == NULL) {
                model->vocoder_conv_pre = wp;
                model->vocoder_conv_pre_len = (int)total;
            }
        }
        if (strstr(name, "dec.conv_post.weight")) {
            if (model->vocoder_conv_post == NULL) {
                model->vocoder_conv_post = wp;
                model->vocoder_conv_post_len = (int)total;
            }
        }

        /* Store tensor in lookup map (uses stored as index, not loaded) */
        if ((uint32_t)stored < n_tensors) {
            strncpy(model->tensors[stored].name, name,
                    sizeof(model->tensors[0].name) - 1);
            model->tensors[stored].data = wp;
            model->tensors[stored].n_dims = (int)n_dims;
            for (int d = 0; d < 4; d++) model->tensors[stored].dims[d] = dims[d];
            stored++;
        }
    }

    /* Post-load: de-normalize weight_norm tensors (weight_g * weight_v/||v||).
     * Cartman v2 HiFi-GAN has 4 upsampling convs + 18 MRF resblock convs
     * (3 stacks × 3 blocks × 2 conv types = 18 convs, each with weight_g + weight_v). */

    /* Upsampling layers: dec.ups.N.weight_g (N,1,1) + weight_v (N, in_ch, k) */
    for (int i = 0; i < 4; i++) {
        if (model->hifi_upsample[i] && model->hifi_upsample_len[i] > 0) {
            char g_name[128];
            snprintf(g_name, sizeof(g_name), "dec.ups.%d.weight_g", i);
            const RVCTensor *wg_t = wubu_rvc_find_tensor(model, g_name);
            if (wg_t && wg_t->data) {
                int n_elems = model->hifi_upsample_len[i];
                int n_ch = wg_t->dims[0];  /* weight_g: (n_out, 1, 1) → n_out channels */
                model->hifi_upsample_denorm[i] = (float *)malloc((size_t)n_elems * sizeof(float));
                if (model->hifi_upsample_denorm[i]) {
                    wubu_rvc_denormalize_weight(wg_t->data, model->hifi_upsample[i],
                                                model->hifi_upsample_denorm[i],
                                                n_elems, n_ch);
                    model->hifi_upsample_denorm_len[i] = n_elems;
                }
            }
        }
    }

    for (int s = 0; s < 12; s++) {
        for (int k_idx = 0; k_idx < 2; k_idx++) {
            const char *conv_name = k_idx == 0 ? "convs1" : "convs2";
            for (int b = 0; b < 3; b++) {
                char key[128];
                snprintf(key, sizeof(key), "dec.resblocks.%d.%s.%d.weight_v", s, conv_name, b);
                RVCTensor *wv_t = (RVCTensor *)wubu_rvc_find_tensor(model, key);
                if (wv_t && wv_t->data) {
                    char gkey[128];
                    snprintf(gkey, sizeof(gkey), "dec.resblocks.%d.%s.%d.weight_g", s, conv_name, b);
                    const RVCTensor *wg_t = wubu_rvc_find_tensor(model, gkey);
                    if (wg_t && wg_t->data) {
                        /* De-normalize in place: weight_v gets replaced with weight_g*weight_v/||v|| */
                        int n_elems = (int)(wv_t->dims[0] * wv_t->dims[1] * wv_t->dims[2]);
                        int n_ch = wg_t->dims[0];  /* 256 */
                        float *denorm = (float *)malloc((size_t)n_elems * sizeof(float));
                        if (denorm) {
                            wubu_rvc_denormalize_weight(wg_t->data, wv_t->data,
                                                        denorm, n_elems, n_ch);
                            free(wv_t->data);
                            wv_t->data = denorm;
                        }
                    }
                }
            }
        }
    }

    /* Set model params.
     * Parse the optional config section at the end of the WUBU binary
     * (written by tools/extract_rvc_weights.py). */
    model->loaded = 1;
    model->in_memory = 1;

    /* After tensor data, read config section if present.
     * We need to seek past the tensor data to find the config.
     * The config was already consumed in the loop above, but we can
     * re-parse it from the buffer. */
    /* Actually, we'll parse config from the remaining buffer after tensors.
     * The buffer 'buf' contains the entire file; after the last tensor,
     * there's [4B config_len][config_data]. */
    /* Re-derive config position from the loop */
    /* We already know 'off' after the loop — but it was a local variable.
     * Instead, let's re-scan the file for the config section. */
    /* Fallback: infer from tensor shapes + defaults. */

    /* Infer upsample configuration: try config section first,
     * fall back to inferring from tensor kernel sizes. */
    /* Try to read config from the file tail */
    FILE *f2 = fopen(bin_path, "rb");
    int config_found = 0;
    if (f2) {
        fseek(f2, 0, SEEK_END);
        long fsz = ftell(f2);
        /* Read last sizeof(uint32_t) for config_len */
        fseek(f2, fsz - 4, SEEK_SET);
        uint32_t config_len = 0;
        if (fread(&config_len, 4, 1, f2) == 1 && config_len > 0 && config_len < 1024) {
            fseek(f2, fsz - 4 - config_len, SEEK_SET);
            uint8_t *cfg_buf = (uint8_t *)malloc(config_len);
            if (cfg_buf && fread(cfg_buf, 1, config_len, f2) == config_len) {
                size_t coff = 0;
                while (coff + 5 <= config_len) {
                    uint8_t field_id = cfg_buf[coff];
                    uint32_t n_vals = read_u32(cfg_buf + coff + 1);
                    size_t consumed = 5;
                    if (field_id == 1 && n_vals > 0 && n_vals <= 8) {
                        /* upsample_rates */
                        for (uint32_t v = 0; v < n_vals && v < 8; v++) {
                            if (coff + 5 + v * 5 >= config_len) break;
                            uint8_t type = cfg_buf[coff + 5 + v * 5];
                            int32_t val = (int32_t)read_u32(cfg_buf + coff + 5 + v * 5 + 1);
                            model->upsample_rates[v] = val;
                        }
                        model->n_upsample_layers = (int)n_vals;
                        consumed = 5 + n_vals * 5;
                        config_found = 1;
                    }
                    if (field_id == 2 && n_vals >= 1) {
                        if (coff + 6 <= config_len) {
                            int32_t sr_val = (int32_t)read_u32(cfg_buf + coff + 6);
                            model->sample_rate = sr_val;
                            consumed = 5 + n_vals * 5;
                        }
                    }
                    if (field_id == 3 && n_vals >= 1) {
                        if (coff + 6 <= config_len) {
                            model->hidden_channels = (int32_t)read_u32(cfg_buf + coff + 6);
                            consumed = 5 + n_vals * 5;
                        }
                    }
                    if (field_id == 4 && n_vals >= 1) {
                        if (coff + 6 <= config_len) {
                            model->mel_channels = (int32_t)read_u32(cfg_buf + coff + 6);
                            consumed = 5 + n_vals * 5;
                        }
                    }
                    if (field_id == 5 && n_vals >= 1) {
                        if (coff + 6 <= config_len) {
                            model->version = (int32_t)read_u32(cfg_buf + coff + 6);
                            model->version_f = (float)model->version;
                            consumed = 5 + n_vals * 5;
                        }
                    }
                    if (field_id == 6 && n_vals > 0 && n_vals <= 8) {
                        /* resblock kernel sizes per stack */
                        for (uint32_t v = 0; v < n_vals && v < 8; v++) {
                            if (coff + 5 + v * 5 >= config_len) break;
                            model->resblock_k[v] = (int32_t)read_u32(cfg_buf + coff + 5 + v * 5 + 1);
                        }
                        if (n_vals > model->n_mrf_stacks) model->n_mrf_stacks = (int)n_vals;
                        consumed = 5 + n_vals * 5;
                    }
                    if (field_id == 7 && n_vals > 0 && n_vals <= 64) {
                        /* flat resblock dilations: stack-major [s0p0,s0p1,...,s1p0,...] */
                        int total = (int)n_vals;
                        int n_stacks = model->n_mrf_stacks > 0 ? model->n_mrf_stacks : 1;
                        int pairs = total / n_stacks;
                        if (pairs < 1) pairs = 1;
                        if (pairs > 8) pairs = 8;
                        model->n_resblock_pairs = pairs;
                        for (int v = 0; v < total && v < 64; v++) {
                            int st = v / pairs;
                            int pk = v % pairs;
                            if (st < 8 && pk < 8)
                                model->resblock_dil[st][pk] =
                                    (int32_t)read_u32(cfg_buf + coff + 5 + v * 5 + 1);
                        }
                        consumed = 5 + total * 5;
                    }
                    coff += consumed;
                }
            }
            free(cfg_buf);
        }
        fclose(f2);
    }

    /* If config wasn't found in the binary, infer from tensor shapes.
     * ConvTranspose1d weight shape is (in_ch, out_ch, k).
     * We use kernel size as a proxy: k=16→stride 10, k=4→stride 2. */
    if (!config_found || model->n_upsample_layers == 0) {
        int n_ups = 0;
        for (int i = 0; i < 4 && i < 8; i++) {
            char key[128];
            snprintf(key, sizeof(key), "dec.ups.%d.weight_v", i);
            const RVCTensor *t = wubu_rvc_find_tensor(model, key);
            if (!t) { snprintf(key, sizeof(key), "dec.ups.%d.weight", i); t = wubu_rvc_find_tensor(model, key); }
            if (t && t->n_dims >= 3) {
                int k = t->dims[2];
                /* Infer stride from kernel size:
                 * Cartman (40k): k=16→stride=10, k=4→stride=2
                 * Miku (48k):    k=24→stride=12, k=20→stride=10, k=4→stride=2 */
                int stride = 1;
                if (k == 16) stride = 10;
                else if (k == 24) stride = 12;
                else if (k == 20) stride = 10;
                else if (k == 4) stride = 2;
                else {
                    /* Generic: try common RVC rates */
                    /* For non-standard kernels, we cannot infer stride reliably.
                     * Fall back to channel ratio if applicable. */
                    int in_ch = t->dims[0];
                    int out_ch = t->dims[1];
                    if (in_ch > out_ch && in_ch / out_ch >= 2) stride = in_ch / out_ch;
                }
                model->upsample_rates[i] = stride;
                n_ups++;
            }
        }
        if (n_ups > 0) {
            model->n_upsample_layers = n_ups;
            /* Also try kernel size=24 for Miku (48k, k=24→stride=12) */
            /* Re-check: if k=24, we should have caught it above.
             * For 48k models: k=[24,20,4,4], rates=[12,10,2,2] */
        }
    }

    /* Compute upsample_rate as product of all upsample_rates */
    int ups_total = 1;
    int n_ups = model->n_upsample_layers > 0 ? model->n_upsample_layers : 4;
    for (int i = 0; i < n_ups && i < 8; i++) {
        if (model->upsample_rates[i] == 0) model->upsample_rates[i] = 1;
        ups_total *= model->upsample_rates[i];
    }
    model->upsample_rate = ups_total;

    /* Hidden channels: dec.conv_pre out channels / 2 (PixelShuffle doubles) */
    if (model->hidden_channels == 0) {
        const RVCTensor *cp = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
        if (cp && cp->n_dims >= 2) model->hidden_channels = cp->dims[0] / 2;
        else model->hidden_channels = 256;
    }

    /* Sample rate: default 40000 for Cartman v2 */
    if (model->sample_rate == 0 || model->sample_rate == 22050) {
        model->sample_rate = 40000;
    }

    /* Mel channels (RVC v2: 80) */
    if (model->mel_channels == 0) model->mel_channels = 80;

    fprintf(stderr, "WuBuRVC: loaded %d/%u tensors from %s (%zu bytes)\n",
            stored, n_tensors, bin_path, (size_t)fsize);

    free(buf);
    return 0;
}

/* Lookup a tensor by exact name in the loaded model. Returns NULL if not found.
 * Uses exact strcmp only — substring matching is unsafe (e.g.,
 * "dec.resblocks.0.convs1.0.weight_v" would match "dec.resblocks.0.convs1.0.weight_v.bias"). */
const RVCTensor *wubu_rvc_find_tensor(const WuBuRVCModel *model,
                                       const char *name) {
    if (!model || !name || !model->tensors) return NULL;
    for (int i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

/* Apply weight normalization de-normalization: W = weight_g * (weight_v / ||weight_v||)
 * Resolves weight_g + weight_v decomposition used in RVC v2 checkpoints.
 * Output is written to 'out' (caller allocates, size = n_elements). */
void wubu_rvc_denormalize_weight(const float *weight_g, const float *weight_v,
                                  float *out, int n_elements, int n_channels) {
    if (!weight_g || !weight_v || !out || n_elements <= 0 || n_channels <= 0) return;
    for (int ch = 0; ch < n_channels; ch++) {
        float norm_sq = 0.0f;
        int per_ch = n_elements / n_channels;
        const float *v_ch = weight_v + (size_t)ch * per_ch;
        for (int i = 0; i < per_ch; i++) {
            norm_sq += v_ch[i] * v_ch[i];
        }
        float norm = sqrtf(norm_sq) + 1e-8f;
        float g = weight_g[ch];
        float scale = g / norm;
        for (int i = 0; i < per_ch; i++) {
            out[(size_t)ch * per_ch + i] = v_ch[i] * scale;
        }
    }
}
