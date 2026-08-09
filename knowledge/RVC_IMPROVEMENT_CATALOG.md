# RVC Voice Quality Improvement Catalog

> Research compiled from 50+ web searches (7-step Kevin Bacon deep-dive method)
> Sources: ArXiv, IEEE, GitHub RVC Project, Applio, ElevenLabs, HiFi-GAN/BigVGAN papers, academic literature

## 1. Pitch (F0) Extraction Improvements (10 items)

1. **RMVPE** — U-Net-based pitch extractor, outperforms CREPE on noisy polyphonic audio (key: extracts F0 directly without pre-separation)
2. **CREPE** — CNN pitch estimation, 99.98% accuracy on clean RWC-synth dataset; slower than RMVPE (RTF 0.48 vs 0.03)
3. **FCPE** — Fast context-based pitch estimation, 5.3x faster than RMVPE, competitive accuracy
4. **Harvest** — Less aggressive than CREPE; softer robotic sound (RVC issue #119)
5. **DIO (Dio)** — Part of World TTS; gentler than Harvest, softer robotic artifacts
6. **Median F0 filtering** — filter_radius≥3 smooths pitch, reduces breathiness (RVC issue #2180)
7. **F0 post-thresholding** — Remove invalid F0 values, interpolate gaps (Lucas et al. 2024)
8. **F0 smoothing** — Temporal smoothing of F0 contour to eliminate jitter (Smart-Median algorithm)
9. **Vibrato separation** — DWT-based high-frequency F0 contour extraction for vibrato control (VibE-SVC, arXiv:2606.17126)
10. **F0 interpolation** — Linear/cubic interpolation between valid F0 frames for continuity

## 2. Content Feature Extraction Improvements (8 items)

11. **HuBERT** — Masked prediction of hidden units; base model 100 clusters; layer 9 (v1) or layer 12 (v2) for content
12. **ContentVec** — HuBERT variant with 3 disentangling mechanisms; lower ABX scores than HuBERT (arXiv:2204.09224)
13. **WavLM** — Advanced SSL model; complementary to HuBERT for Mind-Meld fusion
14. **mHuBERT-147** — Multilingual HuBERT for cross-lingual prosody preservation
15. **CPM / CarBERT** — Chinese pre-trained models for multilingual support
16. **Multi-encoder fusion** — Mind-Meld weighting: HuBERT(45%) + ContentVec(35%) + WavLM(20%)
17. **Layer-weighted HuBERT** — Weighted sum of all 12 transformer layers instead of single layer
18. **Unit-based features** — Quantized SSL units via k-means clustering (100 clusters)

## 3. Post-Processing Pipeline Improvements (15 items)

19. **De-essing** — Reduce harsh sibilance in 5-10kHz range (Sonarworks, Waves Sibilance)
20. **High-frequency enhancement** — Gentle 2-5kHz presence boost for vocal cut-through
21. **EQ pre/post-processing** — Apply targeted EQ curves (cut mud at 200-300Hz, boost presence at 2-5kHz)
22. **De-reverberation** — Remove artificial reverb from converted audio (VR-DeEchoDeReverb)
23. **Clipping prevention** — Ensure max amplitude < 0.95 (the root cause of the original robotic voice!)
24. **Multi-band compression** — Compress different frequency bands independently (24 critical bands)
25. **Harmonic enhancement** — Add subtle harmonic saturation for warmth
26. **Formant shifting** — Adjust formant frequencies for natural gender conversion
27. **Pitch shift** — Fine-tune output pitch in semitones (-12 to +12)
28. **RMS envelope matching** — rms_mix_rate (0-1): blend original loudness dynamics with fixed loudness
29. **Protect voiceless consonants** — protect parameter (0-0.5): preserve breath/sssh sounds from source
30. **Denoising** — Spectral subtraction or Wiener filtering to remove noise floor
31. **Dynamic range compression** — Compressor for consistent output level
32. **Transient shaping** — Enhance attack transients for clarity
33. **Stereo imaging** — Add width/positioning for mixed output
34. **Limiting** — Soft limiter to prevent any clipping (ceiling -0.1dB to -1dB)

## 4. Model Architecture Improvements (12 items)

35. **HiFi-GAN** — Multi-period + multi-scale discriminator, LeakyReLU activation, weight normalization
36. **BigVGAN** — Snake activation (periodic inductive bias), AMP blocks, anti-aliasing filters
37. **BigVGAN v2** — Faster inference (3x), up to 240x RT, 112M params
38. **Snake activation** — f_α(x) = x + (1/α)sin²(αx); provides periodic inductive bias for audio
39. **AMP (Anti-aliased Multi-Periodicity)** — Composes multiple signal components with learnable periodicities
40. **Multi-Resolution Discriminator (MRD)** — Replaces MSD in BigVGAN for enhanced spectral sharpness
41. **VITS flow matching** — Normalizing flows for latent variable modeling (arXiv:2505.21890)
42. **HiFi-SR** — Super-resolution for 4kHz→48kHz upsampling (arXiv:2501.10045)
43. **Conditional VAE** — VITS is a CVAE with normalizing flows + adversarial training
44. **Weight normalization** — Applied to all generator conv layers (HiFi-GAN)
45. **Spectral normalization** — On discriminator (except first sub-discriminator on raw audio)
46. **Residual blocks with dilated convolutions** — HiFi-GAN MRF (Multi-Receptive-Field Fusion)
47. **PixelShuffle upsampling** — RVC v2 conv_post: 1→32 channels then PixelShuffle(x2) → 128
48. **Flow coupling layers** — 4-6 invertible coupling layers in the VITS flow

## 5. Inference Quality Parameters (10 items)

49. **Index rate / search ratio** — 0-1: How much to retrieve from training-set features (1=no timbre leakage)
50. **Extra inference time (extra_ctx)** — Load extra context for each chunk; improves quality at cost of speed
51. **Index nprobe** — Number of partitions to search in FAISS index; higher = better but slower
52. **Chunk size** — 30s processing; larger = more context but more memory
53. **Auto-detect F0 method** — RMVPE (default) vs CREPE vs Harvest vs DIO
54. **Resample to target sample rate** — resample_sr: 0=keep original, or set to 40k/48k
55. **Protect voiceless consonants** — 0.5 = disable protection; 0.33 = recommended default
56. **RMS mix rate** — 0.25 default; 0.8-1.0 for studio post-production
57. **Pitch shift (f0_up_key)** — Adjust output pitch in semitones without affecting content
58. **Filter radius** — ≥3 for median F0 filtering; reduces breathiness
59. **F0 det (F0 detection method)** — RMVPE for singing, CREPE for speech
60. **Batch size** — 16-32 for 3060/4090; affects memory and inference speed

## 6. Training Data & Dataset Quality (8 items)

61. **Minimum 10 minutes clean audio** — Good results possible with ≤10 min; 15 min recommended minimum
62. **Target 30-40 minutes** — Sweet spot for quality; diminishing returns after 60 minutes
63. **Data cleaning pipeline** — UVR5 (Kim Vocal 2 / MDX-NET) → VR-DeEchoDeReverb → Remove silence
64. **Single speaker only** — Remove overlapping voices, background chatter
65. **Keep breath sounds** — Removing all breaths makes voice sound robotic; keep natural breathing
66. **Don't cut words mid-word** — Keep phrases intact to preserve intonation patterns
67. **Consistent recording conditions** — Same mic, same room, same distance; inconsistent audio degrades quality
68. **Pitch shift augmentation** — ±2 semitones during training for data diversity
69. **Time stretch augmentation** — ±10% for robustness to speaking rate variation
70. **Speaker encoder augmentation** — Formant transform for data augmentation during SSL training

## 7. Emotional & Character Voice Control (12 items)

71. **Emotional prosody control** — TTS-style emotion tags: [tense], [relieved], [warm], [sarcastic], [dry]
72. **Phonation types** — Control breathiness, creakiness, strain via glottal source modeling
73. **Prosody transfer via reference encoder** — Encode prosody from reference audio (Tacotron GST paper)
74. **Global Style Tokens (GST)** — Bank of learned style embeddings for prosody control (Wang et al. 2018)
75. **Duration prediction** — FastPitch/FastSpeech2 variance predictors for explicit duration control
76. **Pitch contour control** — Per-phoneme pitch prediction + global contour shaping
77. **Energy prediction** — Per-phoneme energy for dynamic range control
78. **Stochastic duration predictor** — VITS-style flow-based duration for human-like rhythm (stochastic pitch prediction)
79. **Style transfer via data augmentation** — Cross-speaker style transfer using VC-augmented data
80. **Character personality embedding** — Persona tags: "pirate", "businessman", "therapist", "ogre", "godlike being"
81. **Phoneme-level control** — Per-phoneme emotion/prosody control via phoneme alignment (Montreal Forced Aligner)
82. **F0 contour shaping** — Intentional F0 contour design for emotional expression (Sakai et al.)
83. **Multi-speaker emotion model** — Joint training speaker encoder with consistency loss (arXiv:2307.00393)
84. **Expression intensity control** — Scale emotion intensity from 0 (neutral) to 1 (max)

## 8. Denoising & Noise Reduction (7 items)

85. **Spectral subtraction** — Subtract noise estimate from magnitude spectrum
86. **Wiener filtering** — Optimal linear filter for noise reduction in frequency domain
87. **Wavelet denoising** — Multi-resolution wavelet transform with adaptive thresholding (Fu 2003)
88. **Multiband dynamic compression** — 24 critical band filter channels with cross-band interaction (Kollmeier 1993)
89. **Adaptive threshold wavelet denoising** — Adjustable threshold per sub-band (Wang 2025)
90. **Noise gating** — Suppress low-level noise during silence periods
91. **De-reverberation (VR-DeEchoDeReverb)** — Remove room reverb/echo from training data

## 9. Post-Inference Audio Enhancement (8 items)

92. **Harmonic enhancement** — Add subtle odd/even harmonics for richness (tape saturation plugin-style)
93. **Transient shaping** — Enhance attack for consonants, sustain for vowels
94. **Stereo widening** — Mid-side processing for spatial placement
95. **Dynamic EQ** — Frequency-dependent compression for problem frequencies
96. **Saturation / tape emulation** — Analog warmth via soft clipping / tape-style saturation
97. **Loudness normalization** — LUFS-based normalization to -16 LUFS for streaming
98. **De-essing** — Dynamic sibilance reduction in 5-10kHz range
99. **Presence boost** — 2-5kHz EQ boost for intelligibility in mix
100. **Formant preservation** — Maintain formant structure during pitch shift (PSOLA-based)

## 10. Advanced Techniques (10+ bonus items)

101. **Anti-aliasing filters** — Low-pass filter before downsampling/upsampling to prevent aliasing
102. **Snake activation** — x + (1/α)sin²(αx); periodic inductive bias for harmonic signal modeling
103. **Filtered Snake** — Anti-aliased Snake (upsample 2x → apply Snake → downsample 2x)
104. **Vibrato extraction & control** — DWT-based high-frequency F0 contour for vibrato style transfer (VibE-SVC)
105. **PeriodWave** — Multi-period flow matching for high-fidelity waveform generation (arXiv:2408.07547)
106. **F5-TTS** — Non-autoregressive flow matching TTS (arXiv:2406.04364)
107. **HiFi-GAN Multi-Period Discriminator (MPD)** — Period-based discriminators with different periods
108. **HiFi-GAN Multi-Scale Discriminator (MSD)** — Multi-scale conv discriminators at different resolutions
109. **Mel-cepstral distance (MCD)** — Objective metric for evaluating VC spectral quality
110. **MOS / SMOS correlation** — subjective evaluation metrics correlated with objective measures
111. **Speaker consistency loss** — Jointly train speaker encoder + synthesis model for speaker fidelity
112. **Cycle consistency loss** — VC → target → source cycle for preserving linguistic content
113. **Feature matching loss** — L1 loss on discriminator features for smoother gradients
114. **KL divergence loss** — Regularization on posterior encoder latent distribution (VITS)
