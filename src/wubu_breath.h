/* wubu_breath.h — breath-existence detection for RVC conversion.
 *
 * Solves the breathiness inconsistency (knowledge/BREATH_REALISM_RESEARCH.md):
 *  - phantom breaths: flatness-based uv marks SILENCE as unvoiced, so the
 *    generator's noise branch fires on silence
 *  - missing breaths: RVC training drops breaths (issue #65), model can't
 *    render them, and f0 interpolation keeps uv≈1 so no noise texture
 *
 * Detect per-frame (10 ms hop) classes: SILENCE / BREATH / CONSONANT /
 * VOICED using the INTERSPEECH 2024 rule-based recipe (duration ≥300 ms,
 * ZCR, spectral variance) plus energy. Output masks align to the f0 frame
 * grid (100 fps) so the generator can gate noise per frame.
 *
 * All functions are C11, allocation-free (caller provides buffers).
 */
#ifndef WUBU_BREATH_H
#define WUBU_BREATH_H

#define WUBU_BREATH_MAX_FRAMES 65536

/* per-frame class */
enum {
    WUBU_BREATH_SILENCE = 0,
    WUBU_BREATH_BREATH  = 1,
    WUBU_BREATH_CONSONANT = 2,
    WUBU_BREATH_VOICED  = 3
};

/* Detected breath event (for logging/metrics) */
typedef struct {
    int start_frame;   /* in 100 fps frame units */
    int end_frame;
    float peak_zcr;
    float peak_rms;
    float dur_ms;
} WuBuBreathEvent;

typedef struct {
    WuBuBreathEvent events[256];
    int n_events;
    int class_total[4];     /* frames per class */
    int n_frames;
    float voiced_frac;      /* frames classified voiced / total */
    float breath_frac;
} WuBuBreathStats;

/* Detect breath/silence/consonant/voiced per frame.
 * pcm:  16k mono float samples (normalized -1..1), n samples.
 * f0:   100 fps fundamental (Hz, 0 = unvoiced) for voicing corroboration.
 * out_class: caller buffer n_frames ints (WUBU_BREATH_*), NULL ok.
 * out_breath_gain: caller buffer n_frames floats (0..2), NULL ok.
 * stats: optional stats out.
 * Returns n_frames (10 ms frames).
 */
int wubu_breath_detect(const float *pcm, int n,
                       const float *f0, int n_f0,
                       int *out_class, float *out_breath_gain,
                       WuBuBreathStats *stats);

#endif /* WUBU_BREATH_H */
