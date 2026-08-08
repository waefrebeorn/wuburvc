/* wubu_audioio.h — robust WAV read/write + resample (C11).
 *
 * The foundation module of the WuBuDesk audio production suite.
 * Handles RIFF WAV: PCM 16/24/32-int, IEEE float 32 (format 3), and
 * WAVE_FORMAT_EXTENSIBLE (0xFFFE) with proper chunk walking — the naive
 * 44-byte header assumption breaks on DAW exports (Ardour/Mixcraft write
 * float wavs with 18-byte fmt chunks / extensible headers).
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_AUDIOIO_H
#define WUBU_AUDIOIO_H

#ifdef __cplusplus
extern "C" {
#endif

/* A mono float buffer + metadata. For stereo use two buffers (L, R). */
typedef struct {
    float *data;
    int    n;      /* samples per channel */
    int    sr;     /* sample rate */
} WuBuAudio;

/* Read a WAV file. Returns mono (first channel) or NULL on failure.
 * Caller frees returned data. */
WuBuAudio *wubu_audio_read(const char *path);
void wubu_audio_free(WuBuAudio *a);

/* Read a WAV file into interleaved stereo [n * 2]. Mono sources are
 * duplicated to both channels. Caller frees. */
WuBuAudio *wubu_audio_read_stereo(const char *path);

/* Write mono or interleaved-stereo float wav (PCM 16-bit). Returns 0 ok. */
int wubu_audio_write(const char *path, const float *data, int n, int sr, int stereo);

/* Linear-interpolation resample. dst must hold n_out = round(n * out_sr / in_sr).
 * Returns n_out. */
int wubu_audio_resample(const float *in, int n, int in_sr, int out_sr, float *out);

/* Windowed-sinc (Kaiser) resampler — the PROPER way to change sample rate.
 * Linear interpolation aliases high frequencies, which smears pitch
 * (extraction noise, "different kHz → different pitch" across models).
 * Uses a 64-tap Kaiser-windowed sinc (beta=14.8) evaluated per output
 * sample; O(n*64), plenty fast offline and correct for 32k/40k/48k
 * conversions. Returns n_out. */
int wubu_audio_resample_sinc(const float *in, int n, int in_sr, int out_sr,
                             float *out);

/* Trim/copy a segment: out gets samples [start, start+n) (clamped).
 * Returns actual copied count. */
int wubu_audio_slice(const float *in, int n, int start, int len, float *out);

/* Simple stereo sum: adds src (mono) to L and R at pan [-1..1] (L=-1, R=+1). */
void wubu_audio_mix_pan(const float *src, int n, float gain, float pan,
                        float *l, float *r);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_AUDIOIO_H */
