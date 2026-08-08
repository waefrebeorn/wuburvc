#ifndef WUBU_RVC_F0_H
#define WUBU_RVC_F0_H

/* wubu_rvc_f0.h — Real pitch (F0) extraction, YIN (autocorrelation).
 *
 * Replaces the fake zero-crossing estimator. YIN (de Cheveigné & Kawahara
 * 2002) with:
 *   - half-wave difference function
 *   - cumulative mean normalized difference (CMND)
 *   - absolute threshold + parabolic interpolation
 *   - voiced/unvoiced decision with median smoothing
 *
 * Output frame rate is configurable (RVC uses 100 fps at 16 kHz = hop 160,
 * i.e. window 1024, hop 160).
 *
 * License: WaefreBeorn-UMV3
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Extract F0 (Hz) from mono PCM. pcm: n_samples floats in [-1,1] at sr Hz.
 * f0_out: n_frames floats (0.0 = unvoiced). Returns n_frames.
 * window: analysis window (e.g. 1024). hop: frame hop (e.g. 160).
 * fmin/fmax: pitch search range (e.g. 50..1100 Hz). */
int wubu_f0_yin(const float *pcm, int n_samples, int sr,
                int window, int hop, float fmin, float fmax,
                float *f0_out, int max_frames);

/* F0 to RVC coarse pitch (1..255) + raw Hz copy, matching Mangio:
 *   f0_mel = 1127*ln(1+f0/700); normalized to [f0_mel_min..f0_mel_max]
 *   into 254 bins + 1 => [1..255]; unvoiced (f0<=0) -> 1.
 * coarse_out: int n_frames; f0bak_out: float n_frames (Hz, may alias f0_in).
 * f0_min/f0_max: the RVC search range (50/1100). */
int wubu_f0_to_coarse(const float *f0, int n_frames,
                     float f0_min, float f0_max,
                     int *coarse_out, float *f0bak_out);

/* Median filter on an F0 contour (RVC "filter_radius", default radius 3).
 *
 * Kills octave jumps / single-frame pitch spikes that make converted
 * singing go off-key ("hoarse" sound), while radius 1-3 preserves vibrato
 * (5-8 Hz modulation survives a ±1 frame median at 100 fps). Only voiced
 * frames participate in the median — unvoiced (0) frames are left as
 * unvoiced so voicing decisions are not corrupted. radius must be odd >= 1.
 * Modifies f0 in place. */
void wubu_f0_median_filter(float *f0, int n_frames, int radius);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RVC_F0_H */
