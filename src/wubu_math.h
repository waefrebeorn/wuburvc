/* wubu_math.h — folded-polynomial sin/cos (Silas Lock / Kaze Emanuar N64 trick)
 * C11, opaque, no deps. The wave's symmetry shrinks the polynomial domain to
 * [0, pi/4]; a lower-order poly beats the libm's full-range reduction at
 * BETTER accuracy, and the sqrt identity yields sin AND cos from one poly.
 * (c) WuBuRVC — waefrebeorn-umbrella-license. */
#ifndef WUBU_MATH_H
#define WUBU_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Scalar: compute sin(x) and cos(x) with the folded polynomial. */
void wubu_sincos_folded(float x, float *s, float *c);

/* Scalar sine only (the NSF sine excitation hot path). */
float wubu_sinf_folded(float x);

/* Fast exp (IEEE-754 bit trick): ~7e-6 accuracy, fp32-class — for softmax,
 * activations, and the flow's exp where libm precision is overkill. */
float wubu_fastexp(float x);

/* Fast tanh via the sigmoid: 2·sigmoid(2x)−1, same accuracy class. */
float wubu_fasttanh(float x);

/* AVX2 (8-wide): compute 8 sines + 8 cosines. Requires __AVX2__ + __FMA__;
 * on non-AVX2 builds falls back to the scalar loop. */
void wubu_sincos8_folded(const float *x, float *s, float *c, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MATH_H */
