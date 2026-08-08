# Vulkan Accelerator: the cross-vendor GPU path (verified)

## The question: "can we make it in Vulkan?" — YES, verified

The WuBuMedia engine now has THREE kernel backends with the same math:
- C11 CPU (`wubu_rvc_real.c conv1d_c`) — portable everywhere
- CUDA (`wubu_rvc_cuda.cu k_conv1d`) — NVIDIA sm_75, faster than realtime
- **Vulkan (`src/wubu_vk.c` + the GLSL shader) — C-native, vendor-neutral**

Vulkan compute shaders run on NVIDIA, AMD, Intel — one kernel source, no
vendor SDK. Khronos reports Vulkan at 85-100+% of CUDA performance on NVIDIA
hardware; ncnn/MNN/ShaderNN use it as their GPU inference backend. No RVC
ecosystem project has a Vulkan generator — this is ours.

## The verified prototype
`build/t_vk_conv.c` runs the REAL conv_pre (512x192x7 over 300 frames =
153,600 outputs) through a GLSL compute shader on the RTX 2080 SUPER:
```
VULKAN conv1d: 153600 samples | corr 1.000000 | maxdiff 0.0000008
bench: 200 x conv_pre = 9.84 ms/call   (naive path: per-call upload +
                                         synchronous wait; dispatch itself
                                         is sub-ms)
```
The maxdiff 0.0000008 = float32 rounding vs the double reference — the
Vulkan shader is as exact as the CUDA kernel (and both are closer to the
true value than the CPU's float32 accumulation).

## The module (`src/wubu_vk.h` / `src/wubu_vk.c`)
- Opaque `WuBuVk` struct; C11 only; no god header.
- `wubu_vk_create()` — instance → first compute-capable physical device →
  device + compute queue → conv1d pipeline. The SPIR-V is EMBEDDED
  (`src/conv_spv.h` generated from `src/wubu_vk_conv.comp` via
  `glslangValidator -V`), so the module has zero runtime file deps.
- `wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, stride, pad, dil,
  out, n_out)` — identical signature/behavior to the CPU conv1d_c; buffers
  grow lazily (host-visible), uploads + readback are synchronous.
- The shader (`src/wubu_vk_conv.comp`): `local_size(128,4)`, the same
  accumulation order as k_conv1d, push constants for the conv params.

## Build (MSYS2/MinGW)
```
pacman -S mingw-w64-x86_64-vulkan-headers mingw-w64-x86_64-vulkan-loader \
          mingw-w64-x86_64-glslang
glslangValidator -V src/wubu_vk_conv.comp -o <tmp>.spv  # then embed
gcc ... src/wubu_vk.c ... -L/c/msys64/mingw64/lib -lvulkan-1 -lm
```

## Pitfalls hit (worth knowing)
- `VkWriteDescriptorSet wds[4];` uninitialized → garbage pNext crashes
  vkUpdateDescriptorSets. Always `= {0}` (or memset) every Vulkan struct
  you don't fully designated-init.
- The loader is `vulkan-1.dll` (System32, ships with the NVIDIA driver);
  the MinGW import lib is `libvulkan-1.dll.a` (link `-lvulkan-1`).
- The conv shader reads `b_f[oc]` unconditionally — always pass a bias
  buffer (zeros are fine).

## Next (the full generator, per the creed)
The same stage structure as `wubu_rvc_cuda.cu`: add the Vulkan shaders for
convT1d (gather), lrelu/snake/tanh, the sine excitation (host-side like the
CUDA path), the noise convs + the MRF (all conv1d dispatches), the upsample
blocks, and a `wubu_generator_nsf_vk` driver mirroring
`wubu_generator_nsf_cuda` — then the `--vk` CLI flag. The engine stays
arch-agnostic: everything reads the model config + weight shapes at runtime.
