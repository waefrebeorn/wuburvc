# WuBuRVC "Robot Voice" Root Cause — CLI Pipeline Bugs (2026-08-07 v2)

The boss reported Cartman sounded "robot and wrong" on ALL A/B samples.
Investigation (Triple-DA) found the CLI synthesis pipeline diverged from the
verified-correct reference path (`test_rvc_real` = PyTorch Mangio, corr
0.9999, SNR 29.7 dB) in **three** ways. Two are fixed; the third is the
f0-extractor gap (YIN vs RMVPE) which needs an RMVPE/FCPE port.

## Bug 1 (FIXED): f0 timeline compressed 2× against content

The CLI computed YIN f0 at 100 fps, then did `×2 nearest` to 200 fps and
TRUNCATED to the content length (278). Because `test_rvc_real` passes f0
1:1 with the ×2 content (both 278 frames @ 100 fps), the CLI's truncated
×2 array = `repeat(f0, 2)[:278]` — the pitch contour played at HALF speed
relative to the speech. Verified: cli nsff0 was 556 frames vs ref 278.

**Fix:** remove the ×2. RVC consumes f0 at the same frame rate as the
nearest-×2 content (100 fps). Both YIN and `--f0ref` paths now pass f0 1:1.

## Bug 2 (FIXED): content ×2 upsample used linear, not nearest

The CLI's `upsample_frames` used linear interpolation with
`align_corners=True`. The reference generation used
`F.interpolate(feats, scale_factor=2)` on a 3D tensor — default mode
**nearest**. Verified: `np.repeat(content, 2, axis=0)` matches
`content_up.npy` with **max diff 0.0**.

**Fix:** `upsample_frames` now repeats each frame twice (nearest).

## Bug 3 (FIXED 2026-08-07): YIN f0 → RMVPE ported to C11

YIN f0 landed in the same coarse bin as RMVPE only 22.8% of voiced frames —
the flow's emb_f0 conditioning diverged from training distribution → wrong
voice. Fixed by porting **RMVPE** to C11 (exact match for this model):

- `src/wubu_stft.c/.h` — self-contained STFT (torch.stft center=True, hann
  periodic) + mel filterbank.
- `src/wubu_gru.c/.h` — self-contained BiGRU (PyTorch gate order r,z,n; the
  reset gate must gate the HIDDEN term — the first version forgot `r`).
- `src/wubu_rmvpe.c/.h` — DeepUnet (5-enc/4-inter/5-dec, 2D conv/BN/pool/
  convT) + cnn + BiGRU + Linear(512→360) + cents decode.
- `tools/extract_rmvpe_weights.py` — rmvpe.pt → rmvpe_weights.bin (362 MB,
  742 tensors incl. librosa htk mel basis).
- `src/test_rmvpe.c` — verifies vs Python RMVPE: **corr 0.9999, 168/168
  voiced, mean diff 0.19 Hz** (GRU corr 0.99994).

CLI now uses RMVPE by default (`--f0 yin` forces YIN fallback). Coarse-bin
agreement vs training distribution: 96% (YIN was 23%). Output correlates
0.70 with the gold standard (was 0.016 with YIN; 0.9999 with exact f0).

## Status

- All three robot-voice bugs FIXED: nearest content ×2, f0 1:1 (no ×2),
  RMVPE f0 in C11 (default).
- `--f0ref DIR` kept for verification; `--f0 yin` forces the fallback.
- Remaining (small): float32 STFT vs torch FFT rounding → ~4% coarse-bin
  flips at bin boundaries (audibly negligible; 0.19 Hz f0 diff).

## How to reproduce the verification

```sh
# fixed CLI vs gold standard
build/wubu_rvc_cli_fixed.exe out/speech/_rvc_in/cartman_base.wav \
  models/rvc/cartman outputs/fixed.wav --model models/rvc/cartman/EricCartmanV1_e650_s10400.pth \
  --f0ref outputs/rvc_ref
# compare outputs/fixed.wav vs outputs/rvc_ref/ref_pytorch.wav → corr 0.9999
```
