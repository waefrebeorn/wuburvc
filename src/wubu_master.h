/* wubu_master.h — stereo mastering chain (C11).
 *
 * The mastering engine of the WuBuDesk audio production suite. Chain order:
 *   EQ (biquad peaking/shelf/highpass/lowpass)
 *   -> compressor (RMS detector, feed-forward, makeup)
 *   -> saturation (soft tanh tube)
 *   -> stereo width
 *   -> limiter (attack/release peak gain reduction)
 *   -> loudness normalize (RMS target with true-peak safety)
 *
 * All DSP is plain C11, per-sample, sample-rate aware. Designed to run on
 * the WuBuDesk engine and later WuBuOS. Self-contained module.
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_MASTER_H
#define WUBU_MASTER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_EQ_PEAK = 0,   /* peaking bell */
    WUBU_EQ_LOWSHELF,   /* low shelf */
    WUBU_EQ_HIGHSHELF,  /* high shelf */
    WUBU_EQ_HIGHPASS,   /* high-pass (rumble cut) */
    WUBU_EQ_LOWPASS     /* low-pass */
} WuBuEqType;

typedef struct {
    WuBuEqType type;
    float f;        /* center/corner frequency Hz */
    float gain_db;  /* shelf/bell gain (ignored for pass filters) */
    float q;        /* quality (peak/shelf) */
} WuBuEqBand;

typedef struct {
    float threshold_db;  /* -40..0 */
    float ratio;         /* 1..20 */
    float attack_ms;
    float release_ms;
    float makeup_db;
    int   stereo_link;   /* 1 = shared gain from max(L,R) */
} WuBuCompOpts;

typedef struct {
    float drive;         /* 0..1 pre-gain into tanh */
    float mix;           /* 0..1 dry/wet */
} WuBuSatOpts;

typedef struct {
    float width;         /* 0 = mono, 1 = original, >1 = wider */
} WuBuStereoOpts;

typedef struct {
    float ceiling_db;    /* -1..0, typical -1 */
    float release_ms;
} WuBuLimiterOpts;

typedef struct {
    float rms_target;    /* 0 = skip; -14..-23 dBFS typical (-18) */
    float true_peak_db;  /* safety ceiling, e.g. -1 */
} WuBuLoudOpts;

typedef struct {
    WuBuEqBand    eq[8];
    int           n_eq;
    WuBuCompOpts  comp;
    WuBuSatOpts   sat;
    WuBuStereoOpts stereo;
    WuBuLimiterOpts limiter;
    WuBuLoudOpts  loud;
} WuBuMasterOpts;

/* Process interleaved stereo float audio [n*2] in place.
 * Returns 0 on success. */
int wubu_master_process(const WuBuMasterOpts *opts, float *lr, int n, int sr);

/* Convenience: build a default master chain (gentle bus compressor,
 * saturation, limiter at -1 dBTP, RMS -18). */
void wubu_master_default(WuBuMasterOpts *opts);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MASTER_H */
