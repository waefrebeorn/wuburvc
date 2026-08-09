#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_rvc_parity.h"
#include "wubu_vk.h"

static double corr1(const float *a, const float *b, int n) {
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double va = 0, vb = 0, cov = 0;
    for (int i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        va += da * da; vb += db * db; cov += da * db;
    }
    if (va == 0 || vb == 0) return 0.0;
    return cov / sqrt(va * vb);
}

int main(void) {
    const char *model_path = "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth";
    const char *bin_path = "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth.weights.bin";
    WuBuRVCModel *model = wubu_rvc_load_model(model_path);
    if (!model) { printf("load FAIL\n"); return 1; }
    wubu_rvc_load_weights(model, bin_path);
    int F = 16;
    float *mel = (float *)malloc((size_t)192 * F * sizeof(float));
    unsigned s = 12345;
    for (int i = 0; i < 192 * F; i++) { s = s * 1664525u + 1013904223u; mel[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }

    const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    int pre_in = pre_w->dims[1], init_ch = pre_w->dims[0], pk = pre_w->dims[2];
    printf("conv_pre: %d x %d x %d\n", init_ch, pre_in, pk);

    WuBuVk *vk = wubu_vk_create();
    if (!vk) { printf("vk create FAIL\n"); return 1; }
    float *cpu_out = (float *)malloc((size_t)init_ch * F * sizeof(float));
    float *vk_out = (float *)malloc((size_t)init_ch * F * sizeof(float));

    /* CPU conv1d from wubu_rvc_real.h */
    extern int conv1d_c(const float *in, int in_ch, int n, const float *w, const float *b,
                        int out_ch, int k, int stride, int pad, int dil, float *out, int n_out);
    conv1d_c(mel, pre_in, F, pre_w->data, pre_b->data, init_ch, pk, 1, (pk - 1) / 2, 1, cpu_out, F);

    int rc = wubu_vk_conv1d(vk, mel, pre_in, F, pre_w->data, pre_b->data,
                            init_ch, pk, 1, pk / 2, 1, vk_out, F);
    printf("vk conv1d rc=%d\n", rc);
    double c = corr1(cpu_out, vk_out, init_ch * F);
    double maxd = 0;
    for (int i = 0; i < init_ch * F; i++) {
        double d = fabs(cpu_out[i] - vk_out[i]);
        if (d > maxd) maxd = d;
    }
    printf("conv_pre: corr=%.10f maxdiff=%.6f\n", c, maxd);
    printf("cpu[0]=%.6f cpu[1]=%.6f vk[0]=%.6f vk[1]=%.6f\n",
           cpu_out[0], cpu_out[1], vk_out[0], vk_out[1]);

    wubu_vk_destroy(vk);
    wubu_rvc_model_free(model);
    return 0;
}
