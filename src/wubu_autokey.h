#ifndef WUBU_AUTOKEY_H
#define WUBU_AUTOKEY_H

/* wubu_autokey.h — automatic key adaptation for RVC models.
 *
 * Some models (e.g. a child voice trained mostly on speech) pull the output
 * off key when the input sits outside their trained pitch range — measured
 * as a constant cents offset plus octave flips. The fix: feed the model an
 * f0 shifted into its well-conditioned range (usually +1 octave for a low
 * input), then pitch-shift the OUTPUT back to the input's key with the
 * phase vocoder (wubu_pitch).
 *
 * Calibration is per model, cached in <model_dir>/model_key.json:
 *   {"shift": 12, "input_mean": 182.5, "drift": 4.2}
 * Reused when a new input's mean f0 is within ±3 semitones of the cached
 * mean (the model's pull is range-dependent).
 *
 * C11, minimal includes. License: WaefreBeorn-UMV3
 */

#include "wubu_rvc_parity.h"
#include "wubu_rvc_hubert.h"
#include "wubu_rmvpe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Calibrate the pitch feed shift (semitones) for `model` given a probe of
 * the input. Returns the shift to FEED the model (0 = none). Fills
 * *drift_cents (output-vs-input residual after restore, small = good).
 * probe_secs: length of the probe (>=3). The probe pcm is the FIRST
 * probe_secs*16000 samples of the input at 16 kHz. */
float wubu_autokey_calibrate(WuBuRVCModel *model, WuBuHubert *hb,
                             WuBuRmvpe *rm,
                             const float *pcm16, int n16,
                             const float *f0_hz, int n_f0,
                             int content_dim, int sr_out, int ups_total,
                             float noise_scale, int use_snake,
                             int probe_secs, float *drift_cents);

/* Load a cached autokey from <model_dir>/model_key.json. Returns the shift
 * (0 if no cache) and sets *cached_mean / *cached_drift. */
float wubu_autokey_load(const char *model_dir, float *cached_mean,
                        float *cached_drift);

/* Voiced (log) mean f0 in Hz — used to decide cache reuse. */
float wubu_autokey_input_mean(const float *f0, int n);

/* Save the calibration to <model_dir>/model_key.json. */
int wubu_autokey_save(const char *model_dir, float shift, float input_mean,
                      float drift);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_AUTOKEY_H */
