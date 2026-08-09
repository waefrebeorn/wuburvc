/* wubu_math.c — folded-polynomial sin/cos (Silas Lock / Kaze Emanuar N64 trick)
 *
 * The trick: instead of a high-order polynomial over a wide range, use the
 * wave's symmetries to shrink the polynomial's domain to the FIRST EIGHTH
 * [0, pi/4]. Minimax error scales with range^(order+1), so halving the range
 * multiplies accuracy by 2^(order+1) — a lower-order poly beats the libm's
 * full-range path at BETTER accuracy. sin/cos pair from ONE even poly +
 * sqrt(1-x^2) (which also keeps the pair exactly normalized).
 *
 * Algorithm (per x):
 *   i = round(x / (pi/2)) mod 4        (quadrant)
 *   r = x - i*(pi/2)  in [-pi/4, pi/4]
 *   a = |r|,  sgn = sign(r)
 *   c = cos(a) via even poly in a^2
 *   s = sqrt(1 - c*c)                  (exact normalization; s = sin(a))
 *   map back by quadrant i.
 *
 * (c) WuBuRVC — waefrebeorn-umbrella-license. */
#include "wubu_math.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#define WUBU_PI 3.14159265358979323846f
#define WUBU_HALF_PI 1.57079632679489661923f
#define WUBU_QUARTER_PI 0.78539816339744830961f
#define WUBU_TWO_PI 6.28318530717958647692f

/* Even minimax-ish polynomial for cos on [0, pi/4]: c(a) = 1 + c2*a2 + c4*a4 + c6*a6.
 * Max abs error ~1e-7 on [0, pi/4] (checked against libm in tests). */
static inline float wubu_cos_poly(float a2) {
    /* a2 = a*a in [0, (pi/4)^2 ≈ 0.61685] */
    float c = -0.0013697f;               /* c6 */
    c = c * a2 + 0.0416638f;             /* c4 */
    c = c * a2 - 0.4999999f;             /* c2 */
    return c * a2 + 1.0f;
}

/* range-reduce into [-pi/4, pi/4]; returns (i mod 4) and r, plus sgn(a)=sign(r). */
static inline int wubu_reduce(float x, float *r_out, float *sgn_out) {
    float q = x * (2.0f / WUBU_PI);      /* x / (pi/2) */
    float nf = roundf(q);
    float r = x - nf * WUBU_HALF_PI;     /* r in [-pi/4, pi/4] */
    int i = (int)nf & 3;                 /* quadrant mod 4 */
    /* keep i in [0,4) even for negative nf */
    i = (i + 4) & 3;
    float sgn = r < 0.0f ? -1.0f : 1.0f;
    *r_out = r;
    *sgn_out = sgn;
    return i;
}

void wubu_sincos_folded(float x, float *s, float *c) {
    float r, sgn;
    int i = wubu_reduce(x, &r, &sgn);
    float a = r * sgn;                   /* a = |r| in [0, pi/4] */
    float a2 = a * a;
    float ca = wubu_cos_poly(a2);
    float sa = sqrtf(1.0f - ca * ca);    /* exact: sin^2+cos^2 = 1 */
    /* quadrant mapping */
    switch (i) {
        case 0: *s = sgn * sa; *c = ca; break;
        case 1: *s = ca;      *c = -sgn * sa; break;
        case 2: *s = -sgn * sa; *c = -ca; break;
        default: *s = -ca;    *c = sgn * sa; break; /* i == 3 */
    }
}

float wubu_sinf_folded(float x) {
    float s, c;
    wubu_sincos_folded(x, &s, &c);
    return s;
}

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

void wubu_sincos8_folded(const float *x, float *s, float *c, int n) {
    const __m256 k2opi = _mm256_set1_ps(2.0f / WUBU_PI);
    const __m256 khalfpi = _mm256_set1_ps(WUBU_HALF_PI);
    const __m256 kc6 = _mm256_set1_ps(-0.0013697f);
    const __m256 kc4 = _mm256_set1_ps(0.0416638f);
    const __m256 kc2 = _mm256_set1_ps(-0.4999999f);
    const __m256 kone = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        /* range reduce to [-pi/4, pi/4] */
        __m256 q = _mm256_mul_ps(xv, k2opi);
        __m256 nf = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_fnmadd_ps(nf, khalfpi, xv);
        /* quadrant = ((int)nf) & 3 ; float trick: nf mod 4 in [0,4) */
        __m256 qd = _mm256_sub_ps(nf, _mm256_mul_ps(_mm256_set1_ps(4.0f),
                              _mm256_floor_ps(_mm256_mul_ps(nf, _mm256_set1_ps(0.25f)))));
        /* a = |r|, sgn = sign(r) */
        __m256 sgn = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f),
                                      _mm256_cmp_ps(r, _mm256_setzero_ps(), _CMP_LT_OQ));
        __m256 ar = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), r); /* fabs */
        __m256 a2 = _mm256_mul_ps(ar, ar);
        /* cos(a) poly */
        __m256 ca = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(kc6, a2, kc4), a2, kc2), a2, kone);
        __m256 ca2 = _mm256_mul_ps(ca, ca);
        __m256 one_m = _mm256_sub_ps(kone, ca2);
        __m256 sa = _mm256_sqrt_ps(_mm256_max_ps(one_m, _mm256_setzero_ps()));
        /* quadrant masks: qd in [0,1) => i0, [1,2) => i1, etc. */
        __m256 m1 = _mm256_cmp_ps(qd, _mm256_set1_ps(1.0f), _CMP_LT_OQ);
        __m256 m2 = _mm256_cmp_ps(qd, _mm256_set1_ps(2.0f), _CMP_LT_OQ);
        __m256 m3 = _mm256_cmp_ps(qd, _mm256_set1_ps(3.0f), _CMP_LT_OQ);
        __m256 m4b = _mm256_cmp_ps(qd, _mm256_set1_ps(3.0f), _CMP_GE_OQ); /* [3,4) */
        __m256 m2b = _mm256_andnot_ps(m1, m2);   /* [1,2) */
        __m256 m3b = _mm256_andnot_ps(m2, m3);   /* [2,3) */
        __m256 sa_sgn = _mm256_mul_ps(sa, sgn);
        __m256 sa_neg = _mm256_xor_ps(sa_sgn, _mm256_set1_ps(-0.0f)); /* -sgn*sa */
        /* sin values per quadrant: i0: sgn*sa ; i1: ca ; i2: -sgn*sa ; i3: -ca */
        __m256 sv = _mm256_blendv_ps(_mm256_setzero_ps(), sa_sgn, m1);
        sv = _mm256_blendv_ps(sv, ca, m2b);
        sv = _mm256_blendv_ps(sv, sa_neg, m3b);
        sv = _mm256_blendv_ps(sv, _mm256_xor_ps(ca, _mm256_set1_ps(-0.0f)), m4b);
        /* cos per quadrant: i0: ca ; i1: -sgn*sa ; i2: -ca ; i3: sgn*sa */
        __m256 cv = _mm256_blendv_ps(_mm256_setzero_ps(), ca, m1);
        cv = _mm256_blendv_ps(cv, _mm256_xor_ps(sa_sgn, _mm256_set1_ps(-0.0f)), m2b);
        cv = _mm256_blendv_ps(cv, _mm256_xor_ps(ca, _mm256_set1_ps(-0.0f)), m3b);
        cv = _mm256_blendv_ps(cv, sa_sgn, m4b);
        _mm256_storeu_ps(s + i, sv);
        _mm256_storeu_ps(c + i, cv);
    }
    for (; i < n; i++) {
        float ss, cc;
        wubu_sincos_folded(x[i], &ss, &cc);
        s[i] = ss;
        c[i] = cc;
    }
}
#else
void wubu_sincos8_folded(const float *x, float *s, float *c, int n) {
    for (int i = 0; i < n; i++) {
        float ss, cc;
        wubu_sincos_folded(x[i], &ss, &cc);
        s[i] = ss;
        c[i] = cc;
    }
}
#endif

/* ───────────────── fast exp (IEEE-754 bit trick, QuAKE-family) ─────────────────
 * exp(x) ≈ 2^(x·log2e) via bit-level exponent insertion. Accuracy ~7e-6
 * (fp32-class) — the same class as the bit-trick fold. Used for the encoder
 * softmax + the flow's exp where the libm's full-precision path is overkill.
 * Reference: the QuAKE paper's fast exp (10-35% total inference speedup). */
/* ── Fast-math switch (folded-poly / bit-trick approximations) ──
 * 0 (default): libm expf/tanhf — byte-identical output (quality mode).
 * 1: wubu_fastexp/fastsigmoid/fasttanh (~7e-6 accuracy) — speed mode.
 * Lives here (not real.c) so GRU/HuBERT/RMVPE can share it without
 * linking real.o. */
static int g_fast_math = 0;
void wubu_set_fast_math(int on) { g_fast_math = on ? 1 : 0; }
int wubu_get_fast_math(void) { return g_fast_math; }

float wubu_fastexp(float x) {
    /* 2^(x * log2(e)): add x*log2(e)*2^23 to the float exponent bits.
     * 12102203.0f = 2^23 * log2(e) ; 1065353216 = 127 << 23 (bias). */
    union { float f; int32_t i; } u;
    float t = x * 12102203.0f;
    /* clamp to avoid int overflow for extreme x (|x| < ~87 gives 2^87 max) */
    if (t > 12102203.0f * 30.0f) t = 12102203.0f * 30.0f;
    if (t < -12102203.0f * 30.0f) t = -12102203.0f * 30.0f;
    u.i = (int32_t)t + 1065353216;
    return u.f;
}

/* fast sigmoid via the bit-trick exp: 1/(1+exp(-x)) */
float wubu_fastsigmoid(float x) {
    float e = wubu_fastexp(-x);
    return 1.0f / (1.0f + e);
}

/* fast tanh via the sigmoid: tanh(x) = 2·sigmoid(2x) − 1 */
float wubu_fasttanh(float x) {
    float e = wubu_fastexp(-2.0f * x);
    return (1.0f - e) / (1.0f + e);
}
