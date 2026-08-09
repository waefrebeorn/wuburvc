# VULKAN GENERATOR KERNEL OPTIMIZATION RESEARCH (2026-08-09)

> Research pass: 14 online searches → root-cause chain → ≥25 actionable improvements
> for `wuburvc/src/wubu_vk.c` (Vulkan GeneratorNSF). Status 2026-08-09: parity FIXED
> (corr 1.0, maxdiff 3.1e-5) but speed is STILL 20.47s vs CPU 10.32s on the 5s
> cartman clip — the generator is correct-but-slow. This doc is the fix list.
> Method: isolate-then-compare (boss's method), Triple-DA every claim.

## TL;DR — why it's slow (measured, not guessed)

- `time` shows user 0.015s / sys 0.046s on a 21s run → the CPU is idle; the wall
  time is GPU-side waiting. The GPU compute itself is sub-ms; the stall is
  synchronization + memory architecture, NOT the conv math.
- The generator records ~294 dispatches per chunk. Every `rec_*()` used to call
  `rec_flush()` (submit + `vkQueueWaitIdle`) — ~0.1-49µs each **plus** the wait
  (NVIDIA forum: 49µs/launch WITH per-kernel sync vs 3µs without).
- The memory barrier was `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` = full pipeline
  drain between EVERY dispatch. AMD GPUOpen: barriers can cost "from less than
  1% to over 50%". With ~588 barriers/5s-clip that is the dominant cost.
- All weights are HOST_VISIBLE and re-uploaded **per chunk** (110MB .bin!).
  Host-visible memory is uncached PCIe; every upload is a mapped-memcpy +
  implicit sync. NVIDIA: staging→DEVICE_LOCAL is the canonical pattern.
- Every dispatch allocates a descriptor set from a pool (vkAllocateDescriptorSets
  per dispatch). Descriptor writes are host-visible driver work per dispatch.

## The 25+ improvements (grouped; each = one Triple-DA testable change)

### A. Synchronization (biggest win, verified by NVIDIA + AMD docs)
1. ✅ DONE (2026-08-09): single command buffer per generator call — record all
   dispatches, submit once, wait once. Removes ~294 submit+waitIdle round-trips.
   [nvidia-forum-launch-latency: 3µs no-sync vs 49µs with per-kernel sync]
2. ✅ DONE: narrow the barrier from ALL_COMMANDS to
   COMPUTE_SHADER→COMPUTE_SHADER with SHADER_WRITE→SHADER_READ|WRITE. This is
   the AMD GPUOpen "relaxed barrier" recipe; full-pipeline barriers stall the
   whole GPU. [gpuopen-vulkan-barriers]
3. ⏳ TODO: remove the barrier entirely between ops that are *producer-consumer
   within one shader* (the MRF conv→act→conv chains) by fusing them (see C).
   Each remaining barrier is still a pipeline stall.
4. ⏳ TODO: replace `vkQueueWaitIdle` at the end with a **fence** +
   `vkWaitForFences`. waitIdle waits ALL queues; a fence waits only this batch.
   With the flow (CPU) running between chunks, a fence lets the CPU chunk worker
   start the next HuBERT while the GPU finishes the previous chunk.
5. ⏳ TODO: double-buffer the command buffer + fence (frame N submit while frame
   N-1 executes) — overlaps CPU recording with GPU execution. The chunk workers
   are pthreads; the mutex serializes the generator, but recording next chunk
   can overlap.
6. ⏳ TODO: keep the `WUBU_VK_SUBMIT_PER_OP` env escape hatch for debugging but
   make single-buffer the default (already done) — document why.

### B. Memory / uploads (the 110MB-per-chunk tax)
7. ⏳ TODO: **device-local weight staging**: upload weights ONCE at model load
   into DEVICE_LOCAL storage (via a small HOST_VISIBLE staging buffer +
   vkCmdCopyBuffer), then never re-upload per chunk. The weights are constant
   across chunks AND across calls; today `upload()` memcpys the whole 110MB
   every chunk. [vulkan-tutorial-staging, VMA-usage-patterns]
8. ⏳ TODO: keep only the *activations* (z, x, sine, stage buffers) host-visible;
   weights should be DEVICE_LOCAL. Host-visible memory on NVIDIA is
   write-combined PCIe — fine for streaming activations, terrible for
   repeatedly-read weights. [reddit-host-visible-vs-device-local]
9. ⏳ TODO: use `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` or explicit
   `vkFlushMappedMemoryRanges` — today `upload()` memcpys into mapped memory
   with no flush; on some drivers the write may not be visible without a flush
   or the driver does implicit flushes (slow). Measure which.
10. ⏳ TODO: persistent mapped pointer — `vkMapMemory` ONCE per pool slot and
    keep it; today every `upload()`/`download()` maps+unmaps. Map/unmap is a
    driver call; the VMA usage patterns doc recommends persistent mapping.
11. ⏳ TODO: batch all per-chunk host uploads into ONE contiguous staging
    buffer + one copy command, instead of ~30 upload() calls per chunk.
12. ⏳ TODO: sine table + rad/rad_acc are recomputed host-side every chunk —
    they only depend on nsff0 (f0), which is already known per chunk; cache
    the sine buffer per chunk index in the ChunkCtx (only when f0 unchanged).

### C. Kernel fusion (vertical fusion — the MDPI/Modular numbers)
13. ⏳ TODO: **fuse lrelu/snake epilogue into the conv store**: the generator
    does conv → rec_act (separate dispatch + separate global read/write) at
    every layer. MDPI study: fusing 3 kernels (linear+ReLU) = 2.91-3.13x;
    PNNL: 1.3-2.0x for tensor fusion. Make the conv shader take an `act_mode`
    push constant and apply lrelu/snake/tanh in the store loop. Removes ~1
    dispatch AND one full global-memory round-trip per conv.
14. ⏳ TODO: **fuse the noise-conv add into the upsample convT store** (the
    `rec_elt(vk,4,0,14,...)` after the noise conv is a separate pass; make the
    convT shader add a second input buffer).
15. ⏳ TODO: **MRF in ONE dispatch (input-stationary)**: today each MRF pair is
    2 act + 2 conv + 2 elt = 6 dispatches; a stack is 3 pairs; 4 layers × 3
    stacks × 18 = 216 dispatches just for MRF. The skill's own FIXME says:
    "3-stack MRF batch through ONE dispatch (input-stationary)". Load the stage
    once, loop the 3 stacks × 3 pairs in the shader with shared-memory
    residency (see D). Biggest dispatch-count cut available.
16. ⏳ TODO: fuse the final conv_post + tanh (rec_conv1d + rec_act) — same
    epilogue pattern as #13.
17. ⏳ TODO: fuse `rec_elt(7,0,4,...)` + `rec_elt(5,0,7,...)` + `rec_elt(4,0,7,...)`
    (the MRF carry copies) into the first pair's dispatch via shared memory —
    the carry is a read-modify-write on the same buffer.

### D. Shader math / workgroup tuning (occupancy)
18. ⏳ TODO: **workgroup tuning**: conv shader is `local_size(128,4)` (512
    threads). NVIDIA docs: launch bounds ≤128 VGPRs = full occupancy; register
    pressure determines occupancy. Test 64, 128, 256 per dim and measure with
    `maxComputeWorkGroupSize`/`maxComputeSharedMemorySize` queries. 1 thread per
    element reduces perf by 32x per NVIDIA forum — ensure each thread does
    multiple outputs with a loop (grid-stride).
19. ⏳ TODO: **shared-memory input tiles** for conv1d: stage the input patch
    (in_ch × k window) into shared memory once per workgroup, then each thread
    accumulates from smem instead of global. Vulkan docs: smem is "the L1 cache
    you can control"; 32KB limit per workgroup.
20. ⏳ TODO: **weights in shared memory for the MRF** (ch×ch×k kernels): the
    MRF convs are tiny-channel, big-kernel (k=3/7/11); keep the weights in smem
    across the 3-pair loop. [compute-shaders-guide]
21. ⏳ TODO: **subgroup ops**: use `subgroupAdd`/`subgroupShuffle` for the MRF
    average (3 stacks → average) instead of the elt reduce — Khronos tutorial:
    subgroup = "significantly higher performance" than shared memory for
    intra-warp reductions. Requires VK_KHR_shader_subgroup (Vulkan 1.1, all
    desktop GPUs).
22. ⏳ TODO: **float4 vectorized loads/stores** in the conv inner loop where
    in_ch is a multiple of 4 (192, 256, 512 all are): load 4 input channels at
    once, 4 accumulators. Fewer memory instructions, better coalescing.
    [moderngpu performance notes]
23. ⏳ TODO: **non-temporal / streaming stores** for the big output buffers
    (out_audio, stage buffers) — `gl_StorageWriteWithoutFormattedFeedback` /
    `coherent` only where needed; avoid L2 pollution on the final conv_post.
    The skill's FIXME listed "non-temporal output writes".

### E. Precision / data layout
24. ⏳ TODO: **fp16 weights on the GPU**: store the weight buffers as fp16
    (the .pth IS fp16; the extractor upconverts to fp32). Halves weight memory
    bandwidth AND doubles L2 residency. The conv shader accumulates fp32 but
    loads fp16 weights. NVIDIA: half-precision convs are the standard GPU path
    (HiFi-GAN NGC uses fp16); fp16→fp32 conversion is free on sm_75.
    CAUTION: fp16 weights change the accumulation result → parity must be
    re-verified vs CPU (corr target still 1.0 at maxdiff ~1e-4; the CPU-vs-
    PyTorch reference itself only passes SNR≥25dB).
25. ⏳ TODO: **weight repack once at load**: reorder weights to channel-last /
    k-major so the shader's inner loop reads contiguous memory (currently the
    .bin layout is [oc][ic][k] row-major which is fine, but the MRF denorm
    arrays are [out][in][k] — check coalescing).
26. ⏳ TODO: **avoid the memset-zero upload** for the MRF accumulator: today
    `upload(vk, 6, NULL, ...)` zeroes a full stage buffer before the acc loop;
    instead, have the first stack's store write (not accumulate) and stacks 2+3
    accumulate. Removes one full global write per layer.

### F. Pipeline / driver (NVIDIA-specific)
27. ⏳ TODO: query + print device properties once (limits.maxComputeWorkGroupSize,
    maxComputeWorkGroupInvocations, maxComputeSharedMemorySize, subgroup size) —
    tune dispatch sizes to the actual sm_75 limits instead of hardcoded
    128/4/256.
28. ⏳ TODO: **pipeline caching**: `VkPipelineCache` so the driver doesn't
    recompile the 4 compute pipelines on every run (cold start cost).
29. ⏳ TODO: prefer the graphics-queue compute family if it has COMPUTE (NVIDIA
    exposes one queue family; some drivers behave better on it) — currently we
    pick the first COMPUTE-capable family which may be the transfer queue on
    other vendors.
30. ⏳ TODO: check `VK_KHR_push_descriptor` (desktop GPUs support it) — replace
    per-dispatch `vkAllocateDescriptorSets`+`vkUpdateDescriptorSets` with
    `vkCmdPushDescriptorSetKHR`. Reddit: "push descriptors to reduce the
    boilerplate needed to bind storage buffers (supported on all desktop)".

## Research sources (cited)
1. NVIDIA dev forum — kernel launch latency 3µs (no sync) vs 49µs (with sync):
   https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413
2. AMD GPUOpen — Vulkan Barriers Explained (cost 1%→50%+):
   https://gpuopen.com/learn/vulkan-barriers-explained/
3. Vulkan docs — Compute Shaders (shared memory = controllable L1, 32KB limit):
   https://docs.vulkan.org/guide/latest/compute_shaders.html
4. Vulkan tutorial — staging buffer / DEVICE_LOCAL:
   https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer
5. VMA — recommended usage patterns (persistent mapping, staging):
   https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
6. Reddit — host visible vs device local memory:
   https://www.reddit.com/r/vulkan/comments/1h60h06/so_what_are_host_visible_memory_type_and_device/
7. MDPI — kernel fusion on GPU tensor ops (2.91-3.13x for 3-kernel fusion):
   https://www.mdpi.com/2079-9292/15/5/1034
8. PNNL — how much gain from tensor kernel fusion (1.3-2.0x):
   https://www.pnnl.gov/publications/how-much-can-we-gain-tensor-kernel-fusion-gpus
9. Khronos — Vulkan Subgroup Tutorial:
   https://www.khronos.org/blog/vulkan-subgroup-tutorial
10. NVIDIA — Deep Learning Performance: Convolutional Layers (implicit GEMM):
    https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
11. NVIDIA developer blog — Vulkan Dos and Don'ts (compute overlap, barriers):
    https://developer.nvidia.com/blog/vulkan-dos-donts/
12. Modern GPU — occupancy & launch bounds:
    https://moderngpu.github.io/performance.html
13. GPUOpen — optimizing occupancy with large thread groups:
    https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/
14. NVIDIA dev forum — compute shader performance (threads per element):
    https://forums.developer.nvidia.com/t/compute-shader-performance/42669

## Status tracker
- [x] A1 single command buffer (landed in wuburvc 4210af3 follow-up)
- [x] A2 narrow barrier (landed same commit)
- [ ] A3-A6, B7-B12, C13-C17, D18-D23, E24-E26, F27-F30 (TODO)

**Next build order:** C15 (MRF one-dispatch, biggest dispatch cut) → B7
(device-local weights once) → C13 (fused epilogue) → then re-measure. Each step
must keep VK vs CPU parity at corr ≥ 0.999 with the 5s cartman A/B.
