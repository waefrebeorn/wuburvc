#ifndef WUBU_PITCH_H
#define WUBU_PITCH_H

/* wubu_pitch.h — phase-vocoder pitch shifter (no tempo change).
 *
 * The auto-key stage of the RVC pipeline: models trained on a narrow
 * pitch range (e.g. a child voice) pull the output off key when the input
 * sits outside that range. Fix: synthesize with the f0 pre-shifted into
 * the model's comfort zone, then pitch-shift the OUTPUT back to the
 * input's key with a phase vocoder (STFT -> frequency-domain phase
 * manipulation -> overlap-add).
 *
 * C11, opaque state, minimal includes.
 * License: WaefreBeorn-UMV3
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Pitch-shift in place-free: out gets n samples at sr, shifted by
 * `semitones` (positive = up). Duration is preserved. out may alias in.
 * Returns 0 on success. */
int wubu_pitch_shift(const float *in, int n, int sr, float semitones,
                     float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PITCH_H */
