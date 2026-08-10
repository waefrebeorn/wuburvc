/* wubu_harmony.h — dual-fundamental (harmony) pitch detection.
 *
 * RVC's f0 extractors (RMVPE/YIN) return ONE pitch per frame. When the
 * source vocal contains isolated harmony vocals (two simultaneous note
 * tones — thirds/fourths/fifths above the lead), the extractor can flip
 * between the two notes frame-to-frame, causing a pitch "shift" and a
 * robotic warbling in the converted voice.
 *
 * This module detects BOTH fundamentals per frame via harmonic-sum
 * salience over the STFT:
 *   1. Per-frame magnitude spectrum (Hann-windowed FFT)
 *   2. Salience(f) = sum_k |X(k*f)|/k over harmonics (log-spaced grid)
 *   3. Top candidates + continuity bonus toward the previous primary →
 *      stable LEAD line (no note-flipping)
 *   4. A second candidate, not a harmonic multiple of the lead, above a
 *      relative-salience threshold → HARMONY fundamental
 *
 * The harmony fundamental is then injected into the generator's sine
 * excitation so the model renders both tones — no retraining needed.
 *
 * C11, minimal includes. License: WaefreBeorn-UMV3
 */
#ifndef WUBU_HARMONY_H
#define WUBU_HARMONY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Detect dual fundamentals on 16k mono PCM.
 *
 * pcm          : input samples (mono, sr Hz; pass 16k like the f0 path)
 * n_samples    : number of samples
 * sr           : sample rate
 * primary_in   : existing single-f0 contour (Hz, 100 fps, 0 = unvoiced),
 *                from RMVPE/YIN — used as the continuity anchor
 * n_frames     : length of primary_in (== pcm_len/160)
 * window, hop  : STFT analysis window/hop (1024/160 typical)
 * fmin, fmax   : f0 search range (50..1100)
 *
 * primary_out  : STABILIZED lead f0 (Hz, same length) — the input contour
 *                re-picked toward continuity so it never flips to the
 *                harmony note mid-phrase
 * harmony_out  : second fundamental (Hz, 0 = monophonic frame)
 * harmony_gain : 0..~1 relative strength of the harmony vs lead (for the
 *                sine-excitation blend; 0 when monophonic)
 * max_frames   : capacity of the output arrays
 *
 * Returns number of frames processed, or -1 on error.
 */
int wubu_harmony_detect(const float *pcm, int n_samples, int sr,
                        const float *primary_in, int n_frames,
                        float *primary_out, float *harmony_out,
                        float *harmony_gain, int max_frames,
                        int window, int hop, float fmin, float fmax);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HARMONY_H */
