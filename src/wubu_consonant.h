/* wubu_consonant.h — spectral-flatness voicing/consonant detection.
 *
 * The generator's uv mask (voiced/unvoiced) currently comes from
 * nsff0[fi] > 0 — but RVC's get_f0 INTERPOLATES f0 through unvoiced holes
 * before the generator sees it, so nsff0 is almost always > 0 and the uv
 * mask never fires. Unvoiced consonants (s/sh/f, plosive bursts, breath)
 * therefore get a sine excitation instead of the noise excitation the
 * vocoder learned on — the classic "S sounds metallic / consonants mix
 * up" RVC artifact.
 *
 * Fix (source-filter SOTA — ESTVocoder et al.): detect voicing from the
 * signal itself via spectral flatness (Wiener entropy). Voiced frames have
 * a peaked, harmonic spectrum (low flatness); unvoiced consonants/breath
 * have a flat, noise-like spectrum (high flatness). The resulting uv mask
 * is passed to the generator so unvoiced frames get noise excitation.
 *
 * C11, minimal includes. License: WaefreBeorn-UMV3
 */
#ifndef WUBU_CONSONANT_H
#define WUBU_CONSONANT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Compute a per-frame voicing mask from spectral flatness + energy.
 *
 * pcm       : input samples (mono, 16k preferred — the f0 path rate)
 * n_samples : number of samples
 * sr        : sample rate
 * n_frames  : number of output frames (== pcm_len/160 for 100 fps)
 * f0        : existing f0 contour (Hz, 100 fps, 0 = unvoiced) — used to
 *             REFINE the flatness decision (a frame with f0>0 and a peaked
 *             spectrum is definitely voiced)
 * window,hop: STFT window/hop (1024/160 typical)
 *
 * uv_out    : [n_frames] float, 1.0 = voiced, 0.0 = unvoiced/consonant
 * flat_out  : optional [n_frames] raw spectral flatness (0..1)
 *
 * Returns n_frames, or -1 on error.
 */
int wubu_consonant_uv(const float *pcm, int n_samples, int sr,
                      const float *f0, int n_frames,
                      float *uv_out, float *flat_out,
                      int window, int hop);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CONSONANT_H */
