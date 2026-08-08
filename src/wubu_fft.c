/* wubu_fft.c — shared radix-2 complex FFT/IFFT (see wubu_fft.h). */
#include "wubu_fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void wubu_fft(WuBuCpx *a, int n, int inv) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { WuBuCpx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inv ? -1.0 : 1.0);
        WuBuCpx wl = { (float)cos(ang), (float)sin(ang) };
        for (int i = 0; i < n; i += len) {
            WuBuCpx w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                WuBuCpx u = a[i + j];
                WuBuCpx v = { a[i + j + len / 2].re * w.re - a[i + j + len / 2].im * w.im,
                              a[i + j + len / 2].re * w.im + a[i + j + len / 2].im * w.re };
                a[i + j] = (WuBuCpx){ u.re + v.re, u.im + v.im };
                a[i + j + len / 2] = (WuBuCpx){ u.re - v.re, u.im - v.im };
                WuBuCpx nw = { w.re * wl.re - w.im * wl.im, w.re * wl.im + w.im * wl.re };
                w = nw;
            }
        }
    }
    if (inv)
        for (int i = 0; i < n; i++) { a[i].re /= n; a[i].im /= n; }
}
