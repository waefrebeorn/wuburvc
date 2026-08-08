/* wubu_rmvpe.h — self-contained RMVPE F0 extractor (C11).
 *
 * Port of the RVC RMVPE model (E2E: DeepUnet + cnn + BiGRU + head) with the
 * librosa htk mel front-end. Loads models/rvc/rmvpe_weights.bin produced by
 * tools/extract_rmvpe_weights.py. Output: F0 in Hz at 100 fps (16 kHz,
 * hop 160), 0.0 = unvoiced — the same extractor Mangio uses at training
 * time, so coarse bins match the model's conditioning distribution.
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_RMVPE_H
#define WUBU_RMVPE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WuBuRmvpe WuBuRmvpe;

/* Load weights from the WUBU flat binary. Returns NULL on failure. */
WuBuRmvpe *wubu_rmvpe_load(const char *bin_path);
void wubu_rmvpe_free(WuBuRmvpe *r);

/* Extract F0: pcm[n_samples] at 16 kHz -> f0_out[max_frames] Hz at 100 fps
 * (hop 160). Returns frame count (0.0 = unvoiced). -1 on error. */
int wubu_rmvpe_f0(WuBuRmvpe *r, const float *pcm, int n_samples,
                  float *f0_out, int max_frames);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RMVPE_H */
