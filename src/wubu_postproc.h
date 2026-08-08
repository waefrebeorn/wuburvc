#ifndef WUBU_POSTPROC_H
#define WUBU_POSTPROC_H

/* wubu_postproc.h — Audio post-processing pipeline for RVC voice conversion.
 *
 * License: WaefreBeorn-UMV3
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Post-processing options. Zero/zero fields = no effect.
 * Applied in order: mud_cut → presence_boost → de_ess → harmonic → limit → rms */
typedef struct {
    /* EQ */
    float mud_cut_db;          /* Low-shelf at 200Hz. -1 to -3dB typical. 0 = off */
    float presence_boost_db;   /* Peaking EQ at 3kHz. +1 to +3dB typical. 0 = off */

    /* De-essing (dynamic HF attenuation) */
    float de_ess_strength;     /* 0.0 to 0.5. Higher = more de-essing */
    float de_ess_threshold;    /* Energy threshold above which de-essing activates */

    /* Harmonic enhancement (subtle saturation) */
    float harmonic_drive;      /* 0.0 to 0.3. Higher = more saturation warmth */

    /* Output leveling */
    float rms_target;          /* Target RMS (0 = skip). Typical: 0.10-0.25 */
} WuBuPostProcOpts;

/* Full post-processing pipeline. Input and output must be n floats [-1..1]. */
void wubu_post_process(const float *input, float *output, int n, int sr,
                       const WuBuPostProcOpts *opts);

/* Character voice presets */
#define WUBU_PRESET_WARM    1   /* Warm, present (Cartman-style) */
#define WUBU_PRESET_BRIGHT  2   /* Bright, energetic (Kenny-style) */
#define WUBU_PRESET_SMOOTH  3   /* Smooth, neutral (Stan-style) */
#define WUBU_PRESET_BREATHY 4   /* Breathiness, airy (Kyle-style) */

/* Apply a character preset to audio. preset must be one of WUBU_PRESET_*. */
void wubu_apply_character_preset(const float *input, float *output, int n, int sr,
                                 int preset);

/* ── Formant shifting (gender conversion) ──
 * Shifts formant frequencies to make voice sound more masculine or feminine
 * without changing pitch. Uses phase-vocoder-based time/pitch separable processing.
 * shift_ratio: >1.0 raises formants (more feminine), <1.0 lowers them (more masculine)
 * Research: Applio Formant Shift, PSOLA-based voice transformation.
 * Note: This is a simplified phase-vocoder implementation; for production,
 * consider a full PSOLA or real-time WSOLA implementation. */
void wubu_formant_shift(const float *input, float *output, int n, int sr,
                        float shift_ratio);

/* ── F0 contour smoothing ──
 * Smooths the F0 contour to reduce jitter and add natural vibrato.
 * strength: 0.0 = no change, 1.0 = full smoothing.
 * This reduces the "robotic" quality from frame-wise F0 extraction artifacts.
 * Research: VibE-SVC (arXiv:2606.17126), Smart-Median smoothing. */
void wubu_f0_smooth(float *f0, int n, float strength);

/* ── RMS envelope mix (RVC "rms_mix_rate", default 0.25) ──
 * Makes the converted output follow the INPUT volume envelope so dynamics
 * survive conversion. envelope = input_env*mix + output_env*(1-mix), then
 * output is rescaled by envelope/output_env. mix=0 → pure output envelope,
 * mix=1 → exact input envelope. Frame RMS: window 2048, hop 512 at out_sr.
 * in/out must be same length n at out_sr. In place on out. */
void wubu_rms_mix_rate(const float *input, float *output, int n, int sr,
                       float mix);

/* ── Adaptive mixing (index rate) ──
 * Blends original (unconverted) features with converted features based on
 * energy threshold. High-energy voiced regions use more converted features;
 * low-energy regions (silence, breath) keep original for stability.
 * index_rate: 0.0 = no retrieval mixing, 1.0 = full retrieval.
 * Research: RVC index_rate parameter (FAQ #11). */
void wubu_adaptive_feature_blend(const float *src_feat, const float *ref_feat,
                                  float *output, int n_frames, int dim,
                                  float index_rate, const float *energy);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_POSTPROC_H */
