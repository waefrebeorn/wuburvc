# WuBuDesk Audio Production / Mastering Suite (C11) — v1

Begun 2026-08-07. The foundation of the OS-scale audio production suite —
pure C11 modules that will run on Windows now and WuBuOS later.

## Modules

- `src/wubu_audioio.c/.h` — robust WAV IO. Chunk-walking parser handles
  PCM 16/24/32-int, IEEE float 32, and WAVE_FORMAT_EXTENSIBLE — real DAW
  exports (Ardour/Mixcraft) fail naive 44-byte-header readers. Also:
  resample (linear), slice, pan mix.
- `src/wubu_master.c/.h` — stereo mastering chain:
  EQ (RBJ biquads, **double precision** — a 30 Hz HPF at 48 kHz puts poles
  on the unit circle and float32 tips them unstable → the filter explodes
  over long files) → feed-forward compressor (RMS sidechain, stereo link,
  makeup) → tanh saturation → stereo width → peak limiter → RMS loudness
  normalize with true-peak safety.
- `tools/wubu_mixmaster.c` — CLI: `out.wav sr stem:gain:pan [...]` → sums
  stems (auto-resample to bus rate) → default master chain → PCM16 stereo.
- `tools/slice_stems.py` — slices Ardour float-wav stems to clean pcm16.
- `src/test_master.c` — Triple-DA test: no NaN/Inf, peak under ceiling,
  RMS at target, wav roundtrip. ALL PASS.

## Pitfalls found (Triple-DA)

1. **WAV fmt-chunk offsets**: bits-per-sample lives at byte 22 of the fmt
   chunk, NOT byte 18 (18 is byte-rate). Reading 18 gave bits=2 on float
   wavs → `bits/8 = 0` → division by zero → SIGFPE crash in read.
2. **Biquads must be double precision** for low-frequency filters near
   DC/Nyquist (30 Hz HPF @ 48 kHz).
3. **Master-chain order matters**: EQ → comp → sat → width → limiter →
   loudness (the limiter must sit AFTER saturation/width, and loudness
   normalization last with true-peak safety or it un-clamps the limiter).
4. `%` in Ardour stem filenames breaks ffmpeg input — slice with Python/C,
   not ffmpeg.

## Production demo (clevelandisgolden album)

- Stems: `C:\Users\eman5\Documents\clevelandisgolden\interchange\...\audiofiles\`
  (48 kHz mono L/R pairs, 32-bit float, 207 s).
- Pipeline: slice 40–85 s → RVC convert lead vocal (Cartman/Bart via
  `wubu_rvc_cli_fixed`) → `wubu_mixmaster` (18 stems, lead replaced) →
  mastered @ −18 dBFS RMS.
- Artifacts in `out/demo/`: original_mastered.wav, cartman_mastered.wav,
  cartman_lead_raw.wav, demo_vocals.mp4, demo_mixes.mp4.

## Next steps

- Lookahead limiter, multiband compression, LUFS (K-weighting), dithering,
  oversampling, de-esser in the master chain.
- Stereo imaging (MS tools), parallel compression.
- GUI/CLI plugin architecture for WuBuOS.
