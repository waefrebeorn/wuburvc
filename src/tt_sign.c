#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* Directly check: d|X|/dx[n] = (Re cos + Im sin)/|X|  OR  (Re cos - Im sin)/|X| */
int main(void) {
    int w = 64, nb = w/2+1;
    float x[64];
    for (int i = 0; i < w; i++) x[i] = 0.5f * sinf(2 * (float)M_PI * 220 * i / 16000) + 0.2f * sinf(2 * (float)M_PI * 440 * i / 16000);
    /* DFT at bin 8 with our convention: ph = -2pi*k*t/w */
    int k = 8; float re = 0, im = 0;
    for (int t = 0; t < w; t++) { float ph = -2.0f * (float)M_PI * k * t / w; re += x[t] * cosf(ph); im += x[t] * sinf(ph); }
    float m = sqrtf(re*re + im*im);
    printf("bin %d: re=%.4f im=%.4f |X|=%.4f\n", k, re, im, m);
    /* FD at n=10 */
    float h = 1e-4f; int n0 = 10;
    float save = x[n0];
    x[n0] = save + h;
    float re2 = 0, im2 = 0;
    for (int t = 0; t < w; t++) { float ph = -2.0f * (float)M_PI * k * t / w; re2 += x[t] * cosf(ph); im2 += x[t] * sinf(ph); }
    float mp = sqrtf(re2*re2 + im2*im2);
    x[n0] = save - h;
    float re3 = 0, im3 = 0;
    for (int t = 0; t < w; t++) { float ph = -2.0f * (float)M_PI * k * t / w; re3 += x[t] * cosf(ph); im3 += x[t] * sinf(ph); }
    float mm = sqrtf(re3*re3 + im3*im3);
    x[n0] = save;
    float fd = (mp - mm) / (2 * h);
    float phn = -2.0f * (float)M_PI * k * n0 / w;
    float plus  = (re * cosf(phn) + im * sinf(phn)) / m;
    float minus = (re * cosf(phn) - im * sinf(phn)) / m;
    printf("FD d|X|/dx[%d] = %.6f\n", n0, fd);
    printf("PLUS  = %.6f\n", plus);
    printf("MINUS = %.6f\n", minus);
    return 0;
}
