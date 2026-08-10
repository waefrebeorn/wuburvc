# Backbone Research: ContentVec, WavLM, mHuBERT (2026-08-09)

## Verified finding: our engine ALREADY runs ContentVec

RVC's `hubert_base.pt` (the file every RVC v2 model was trained with) **IS
ContentVec legacy-500** — the name is misleading. Evidence:

- `innnky/contentvec` HF mirror ships `checkpoint_best_legacy_500.pt` (189,507,909
  bytes). Downloaded and md5'd: **b76f784c1958d4e535cd0f6151ca35e4** — byte-identical
  to `WuBuMedia/models/rvc/hubert_base.pt`.
- The annotated-RVC walkthrough (gudgud96.github.io/2024/09/26/annotated-rvc)
  confirms RVC's feature extractor "has a higher resemblance with ContentVec"
  (requires more steps to reduce source speaker info).
- RVC-Project issue #2078: "using the original contentvec there is no problem
  even though in theory we use hubert base" — i.e. hubert_base.pt == ContentVec.

**Consequence**: our C11 engine's `wubu_rvc_hubert.c` (fairseq key structure,
12 encoder layers, layer-12 features for v2) already extracts ContentVec-grade
content features. No port needed for parity with RVC training.

## Why a different backbone requires TRAINING (not just inference)

Existing .pth voice models (SpongeBob, Cartman, Cleveland, Seth) were trained
with hubert_base/ContentVec features. The enc_p + generator expect that
distribution. Swapping in WavLM/mHuBERT features at inference would MISMATCH
the trained model → worse output. The backbone upgrade only pays off when the
voice model is ALSO retrained on the new features — which our Vulkan
record-mode trainer (wubu_train_vk.c) can now do.

## Candidate upgrades (2025-2026 state of the art)

1. **WavLM (Microsoft)** — strong content + prosody representation. Base+ ~94M
   params, 7-layer conv extractor + 12 transformer layers (like HuBERT base).
   Differences: gated relative position bias in attention, LayerDrop, trained
   with denoising/masking objectives. Feasible C11 port by extending
   wubu_rvc_hubert.c (relative-position tables + gates per layer).
   Reference: Voice Privacy Challenge 2024 papers use WavLM as content encoder.
2. **mHuBERT-147 (utter-project)** — multilingual (147 langs), discrete-unit
   output (k-means tokens) rather than continuous 768-dim features. Big
   advantage for non-English album material. Requires a unit→embedding
   projection to feed enc_p; different port shape.
3. **Seed-VC (Plachtaa)** — zero-shot VC with Whisper-based linguistic content
   extractor. Different architecture family; not RVC-compatible (can't reuse
   existing .pth models).

## Recommendation

- Keep ContentVec (already running) for all EXISTING models — it matches their
  training distribution. This is the correct, non-regressive choice.
- The next backbone milestone: port WavLM-base into wubu_rvc_hubert.c (or a
  sibling module) + wire it as `--backbone wavlm`, then RETRAIN a target voice
  on WavLM features via the Vulkan trainer. Train + infer with the SAME
  backbone. This is a self-contained milestone gated on training throughput.
- For multilingual album material, evaluate mHuBERT-147 after WavLM.

## Files

- `WuBuMedia/models/rvc/hubert_base.pt` (== ContentVec legacy-500)
- `WuBuMedia/models/rvc/contentvec_weights.bin` (extracted — identical to
  hubert_weights.bin; kept for documentation, not needed separately)
- `src/wubu_rvc_hubert.c` — C11 fairseq HuBERT/ContentVec inference
- `src/wubu_train_vk.c` — Vulkan training (retrain-on-new-backbone path)
