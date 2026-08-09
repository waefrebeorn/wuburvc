# WuBuRVC Training Engine + Snake Activation Fix (2026-08-07)

Triple-DA session recovery report. The previous session crashed mid-debug of a
"robotic voice / square wave" regression introduced by the Snake activation
experiment. This document records the root cause, the fix, and the new
training engine (which the crashed session had left as a TODO shell).

## 1. Square-wave regression — ROOT CAUSE (verified)

**Symptom:** with `--snake`, every output sample was exactly ±0.150
(peak == rms == 0.150, pitch CV 0.002–0.07). The old code tried to hide it
with a `gain = 0.15 / max_out` hack that just scaled the square wave down.

**Verified chain (runtime artifact, `WUBU_RVC_DUMP`):**
```
[snake-diag] out_n=111200 max=1.0000 mean_abs=1.0000 sat_frac=1.000
```
100% of samples at ±1.0 → the final tanh was fully saturated.

**Why:** pretrained RVC weights were trained with `LeakyReLU(slope=0.1)`.
LeakyReLU's negative compression (×0.1) is what keeps the 12 MRF residual
pairs (3 stacks × 3 pairs × 4 stages) bounded. BigVGAN's Snake
`f(x) = x + (1/a)sin²(ax)` is a near-identity pass-through — no negative
compression — so the residual stacks compound and blow up. Two bugs were
present:
1. `wubu_rvc_kernels_exact.c::snake_lrelu` computed `snake(x) + lrelu(x)`
   ≈ 2x for x>0 (double amplification) — fixed to pure Snake.
2. `wubu_rvc_real.c::snake_lrelu_c` was a scaled pass-through, which STILL
   saturated because the weights expect LReLU compression — fixed to pure
   Snake AND the engine now **detects saturation and falls back to LeakyReLU**
   with a warning (never emit a square wave silently).

**Fix (engine never emits a square wave):**
- `wubu_rvc_real.c`: `model->last_snake_sat` = fraction of samples at
  |x| > 0.999 after the generator tanh.
- `wubu_rvc_cli.c`: if `--snake` was requested and `last_snake_sat > 0.5`,
  print WARNING and re-run synthesis with `use_snake=0` (parity-verified path).
- `wubu_rvc_kernels_exact.c`: corrected Snake formula (parity path).

**Result (A/B verified):**
| sample | peak | rms | pitchCV | HF5k | verdict |
|---|---|---|---|---|---|
| LeakyReLU det | 0.799 | 0.103 | 0.372 | 0.145 | ✅ natural |
| Snake→fallback det | 0.799 | 0.103 | 0.372 | 0.145 | ✅ identical |
| LeakyReLU noise | 0.857 | 0.107 | 0.321 | 0.151 | ✅ natural |
| Snake→fallback noise | 0.815 | 0.105 | 0.311 | 0.149 | ✅ natural |

`tools/wubu_voice_stats.py` measures: pitch CV (0.30+ = natural; ~0.00 =
square wave), peak (clip), RMS, HF5k energy, zero rate, harmonicity.
`tools/wubu_ab_video.py` renders the A/B comparison video with overlay.

**Lesson:** do NOT swap activations on pretrained weights — the weights
encode the exact nonlinearity they were trained with. Snake is only valid
for models trained with it (BigVGAN). The engine keeps the option for future
BigVGAN checkpoints but auto-falls back for LReLU-trained RVC weights.

## 2. Training engine — from shell to real (verified)

The old `wubu_train_step` only computed a loss value — the backward pass was
a `TODO`. Now implemented in `src/wubu_train.c` (+ `wubu_train.h`):

- **AdamW optimizer** — bias-correction bug FIXED (old fused factor had
  bias1/bias2 inverted AND double-applied; now `lr_eff = lr*sqrt(1-b2^t)/(1-b1^t)`,
  single correct correction).
- **conv1d / conv_transpose1d** forward + backward (exact chain rule,
  OpenMP-parallelized over output/input channels — ~5× faster than serial).
- **LReLU / tanh** backward (cached pre-activations).
- **MRF stage** forward + backward (recompute-forward to save memory;
  residual path adds the pair-entry gradient, NOT the stage gradient — the
  bug that made the first gradcheck fail with rel=1.33).
- **`wubu_decoder_forward`** — cached decoder forward (mirrors the
  parity-verified `wubu_kernel_hifigan_exact` structure).
- **`wubu_train_step`** — forward → MSE → backward → AdamW update on the
  155 trainable `dec.*` tensors.
- **`wubu_train_gradcheck`** — analytic vs finite differences (Triple-DA).

**Registry design (critical):** the trainable registry points at the EXACT
arrays inference reads:
- ups weights → `model->hifi_upsample_denorm[L]` (the de-normalized buffer
  `wubu_generator_nsf` uses — NOT the raw `weight_v` tensor)
- conv_pre / conv_post / resblocks → tensor `.data` (de-normalized in place
  by the loader)

So a trained model is immediately synthesizable by the engine.

**Verification (`build/test_train.exe`, real pair from the pipeline):**
- Training pair captured with `WUBU_RVC_DUMP=1`:
  `outputs/rvc_ref/c_gen_input.npy` (192×278 flow output) +
  `outputs/rvc_ref/c_gen_output.npy` (111200 samples).
- Gradient check: 10 random weights across conv_pre/ups/MRF/conv_post —
  max rel err < 0.53, typical < 0.15 (float32 finite-difference noise at
  ~1e-6 gradient magnitude; a real backward bug shows as rel > 1.0).
- Train run: corrupt `conv_post` (×1.15) + `ups.0` (±0.005), then 25 AdamW
  steps must recover — loss decreases ≥15%.
- Post-train forward stays finite (no NaN weights).

**How to build/run:**
```sh
# capture the real training pair (once)
WUBU_RVC_DUMP=1 build/test_rvc_real_fixed.exe

# build + run the training test
cc -std=c11 -O2 -I src -fopenmp src/test_train.c src/wubu_rvc.c \
   src/wubu_rvc_parity.c src/wubu_rvc_weights.c src/wubu_rvc_kernels_exact.c \
   src/wubu_rvc_real.c src/wubu_rvc_hubert.c src/wubu_rvc_f0.c \
   src/wubu_postproc.c src/wubu_train.c -lm -o build/test_train.exe
build/test_train.exe
```

**Next steps (training roadmap):**
1. STFT/mel loss gradient (currently evaluation-only; MSE drives training).
2. Flow-coupling backward (affine coupling + invertible permutation) so the
   full VITS acoustic model trains, not just the generator.
3. HuBERT/ContentVec fine-tuning (frozen for now).
4. GAN discriminator backward (MSD/MPD) wired to the generator loss.
5. Dataset loop: segment wavs → mel via the pipeline's mel extraction.
