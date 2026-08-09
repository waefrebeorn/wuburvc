/* wubu_vk.c — Vulkan compute accelerator (implementation).
 *
 * Cross-vendor GPU kernels (NVIDIA/AMD/Intel) via the Vulkan compute API.
 * Self-contained: instance → device → compute queue → pipelines (SPIR-V
 * embedded) → generic device-buffer pool. Same math as the CPU kernels
 * (wubu_rvc_real.c) and the CUDA kernels (wubu_rvc_cuda.cu).
 *
 * The generator records ALL dispatches into ONE command buffer (device-side
 * chaining — no per-op upload/download/sync), submits once, reads back once.
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_vk.h"
#include "wubu_rvc_real.h"
#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "wubu_vk_conv_spv.h"
#include "wubu_vk_convt_spv.h"
#include "wubu_vk_act_spv.h"
#include "wubu_vk_elt_spv.h"
#include "wubu_vk_conv_tiled_spv.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* manual key builder (no snprintf — avoids CRT deps): "dec.ups.3.bias" etc. */
static void kfmt(char *o, const char *pre, int a,
                 const char *mid, int b, const char *suf) {
    char *p = o;
    const char *q;
    for (q = pre; *q; q++) *p++ = *q;
    if (a == 0) *p++ = '0';
    else { char t[8]; int n = 0; while (a) { t[n++] = (char)('0' + a % 10); a /= 10; } while (n) *p++ = t[--n]; }
    for (q = mid; *q; q++) *p++ = *q;
    if (b >= 0) {
        if (b == 0) *p++ = '0';
        else { char t[8]; int n = 0; while (b) { t[n++] = (char)('0' + b % 10); b /= 10; } while (n) *p++ = t[--n]; }
    }
    for (q = suf; *q; q++) *p++ = *q;
    *p = 0;
}

#define WUBU_VK_POOL 200
#define WUBU_VK_ZERO_SLOT 199   /* reserved: persistent GPU zero buffer — wslot() must never allocate it */

struct WuBuVk {
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue q;
    uint32_t qfam;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkDescriptorSetLayout dsl_conv, dsl_act, dsl_elt;
    VkDescriptorSetLayout dsl_conv_t;    /* tiled conv needs its OWN layout object:
                                          * mk_pipe overwrites the passed dsl, which
                                          * would clobber dsl_conv and break the conv/
                                          * convt pipelines' set binding. Same 4-storage
                                          * structure, so sets from dsl_conv are
                                          * compatible with pl_conv_t. */
    VkPipelineLayout pl_conv, pl_act, pl_elt;
    VkPipeline pipe_conv, pipe_convt, pipe_act, pipe_elt;
    VkPipelineLayout pl_conv_t;      /* stride-1 tiled conv (9-int push: +epi) */
    VkPipeline pipe_conv_t;
    VkDescriptorPool dp[4];
    VkDescriptorSet ds_conv, ds_convt, ds_act, ds_elt;
    /* per-dispatch descriptor sets (the shared ds_* are only for the public
     * single-op API; the recorded dispatches each get their own set because
     * the host-side updates would otherwise leak into the earlier dispatches) */
    VkDescriptorPool dspool;
    VkDescriptorSet dspool_sets[1024];
    uint32_t dspool_n;
    int last_written;   /* pool slot the last dispatch wrote (flush touch) */
    int no_flush;       /* record-only mode: dispatches accumulate in one
                         * command buffer, submitted once at the end. Set by
                         * the generator around its recording phase. */
    int pipe_submit;    /* pipelined mode: submit each dispatch WITHOUT wait,
                         * wait once at the end. In-order queue execution is
                         * the sync (same correctness as per-op waits, without
                         * 225 x waitIdle round-trips). SAFE only with the
                         * weight cache (wslot) — shared weight slots are never
                         * rewritten mid-record, and dspool must not wrap. */
    int cb_ring_n;      /* command-buffer ring index (pipelined submits) */
    VkCommandBuffer cb_ring[8];   /* ring: submit without wait needs a fresh cb
                                   * per in-flight submit — the single cb is
                                   * re-recorded the moment the next dispatch
                                   * starts, which races the GPU read. */
    VkFence cb_fence[8];          /* per-ring-slot fence: before REUSING a slot
                                   * (8 dispatches later) wait for its fence so
                                   * the host never re-records a cb the GPU is
                                   * still reading. Bounded 8-deep pipeline. */
    /* generic device-buffer pool */
    VkBuffer buffers[WUBU_VK_POOL];
    VkDeviceMemory pmem[WUBU_VK_POOL];
    size_t pcap[WUBU_VK_POOL];
    VkDeviceSize poff[WUBU_VK_POOL];   /* map offset (allocator alignment) */
    VkDeviceSize psize[WUBU_VK_POOL];  /* actual allocation size */
    uint32_t pflags[WUBU_VK_POOL];     /* memory type property flags (coherency) */
    /* conv1d public API uses slot 0/1/2/3 */
    size_t cap_in, cap_w, cap_b, cap_out;
    /* weight-preload cache: upload each generator tensor ONCE at first use and
     * keep it resident. Per-op mode re-uploads ~150MB of MRF/upsample weights
     * per chunk (measured bottleneck: 8.76s vs 6.45s CPU for 3s). With weights
     * resident, per-chunk uploads drop to z + sine only. ALSO makes the
     * single-buffer (no_flush) path safe: a later upload() to a shared slot
     * no longer clobbers a dispatch recorded earlier (the crashed session's
     * single-buffer bug — slots 9/12/15 were re-written mid-record). */
    int wn;                              /* number of cached weight entries */
    struct { char name[96]; int slot; } wcache[256];
    size_t wpcap[WUBU_VK_POOL];          /* per-slot capacity for weight slots 9..18 */
};

static int pick_mem_type(WuBuVk *vk, uint32_t type_bits, uint32_t *out_idx) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->phys, &mp);
    /* PASS 1: DEVICE_LOCAL | HOST_VISIBLE (VRAM-resident AND mappable).
     * On discrete NVIDIA the first HOST_VISIBLE type is host memory — every
     * dispatch round-trips stage buffers over PCIe (~28ms each on a 2080
     * SUPER, measured). VRAM-resident host-visible is the fast path: uploads
     * still work via map, but GPU reads/writes stay in VRAM. */
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            *out_idx = i;
            if (getenv("WUBU_VK_DBG"))
                fprintf(stderr, "VKDBG memtype %u flags=0x%x (DEVICE_LOCAL|HOST_VISIBLE)\n",
                        i, (unsigned)mp.memoryTypes[i].propertyFlags);
            return 0;
        }
    }
    /* PASS 2: any HOST_VISIBLE (fallback — slower, PCIe round-trip) */
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            *out_idx = i;
            if (getenv("WUBU_VK_DBG"))
                fprintf(stderr, "VKDBG memtype %u flags=0x%x (HOST_VISIBLE fallback)\n",
                        i, (unsigned)mp.memoryTypes[i].propertyFlags);
            return 0;
        }
    }
    return -1;
}

/* slot allocation: grow on demand; slot 0..3 are the public conv API. */
static int pool_slot(WuBuVk *vk, int idx, size_t need) {
    if (idx < 0 || idx >= WUBU_VK_POOL) return -1;
    if (vk->buffers[idx] != VK_NULL_HANDLE && vk->pcap[idx] >= need) return 0;
    if (vk->buffers[idx] != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->dev, vk->buffers[idx], NULL);
        vkFreeMemory(vk->dev, vk->pmem[idx], NULL);
    }
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = need ? need : 16,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(vk->dev, &bci, NULL, &vk->buffers[idx]) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(vk->dev, vk->buffers[idx], &mr);
    uint32_t mi = 0;
    if (pick_mem_type(vk, mr.memoryTypeBits, &mi) != 0) return -1;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = mr.size, .memoryTypeIndex = mi };
    if (vkAllocateMemory(vk->dev, &mai, NULL, &vk->pmem[idx]) != VK_SUCCESS) return -1;
    vkBindBufferMemory(vk->dev, vk->buffers[idx], vk->pmem[idx], 0);
    vk->pcap[idx] = need;
    vk->poff[idx] = 0;
    vk->psize[idx] = mr.size;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->phys, &mp);
    vk->pflags[idx] = mp.memoryTypes[mi].propertyFlags;
    return 0;
}

static int mem_coherent(WuBuVk *vk, int slot) {
    return (vk->pflags[slot] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

static void upload(WuBuVk *vk, int slot, const void *data, size_t sz) {
    void *p = NULL;
    vkMapMemory(vk->dev, vk->pmem[slot], 0, vk->psize[slot], 0, &p);
    if (data) memcpy(p, data, sz); else memset(p, 0, sz);
    if (!mem_coherent(vk, slot)) {
        VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                  .memory = vk->pmem[slot], .offset = 0, .size = sz };
        vkFlushMappedMemoryRanges(vk->dev, 1, &r);
    }
    vkUnmapMemory(vk->dev, vk->pmem[slot]);
}

/* weight-preload cache: find-or-upload a named tensor into its OWN pool slot.
 * Tensors are keyed by name so the generator's per-layer loops hit the cache
 * after chunk 1 — weights stay resident, per-chunk uploads drop to z+sine.
 * Each tensor gets a dedicated slot (never shared), which ALSO makes the
 * single-buffer (no_flush) path safe: no later upload() can clobber a slot a
 * previously-recorded dispatch references. Returns the slot or -1. */
static int wslot(WuBuVk *vk, const char *name, const void *data, size_t sz) {
    if (!vk || !name) return -1;
    for (int i = 0; i < vk->wn; i++) {
        if (strcmp(vk->wcache[i].name, name) == 0) return vk->wcache[i].slot;
    }
    /* find a free slot >= 12 (0-8 per-call activations, 9/10 scratch, 11 cond,
     * 199 = reserved GPU zero buffer — never hand it to a weight tensor!) */
    int slot = -1;
    for (int s = 12; s < WUBU_VK_ZERO_SLOT; s++) {
        int used = 0;
        for (int i = 0; i < vk->wn; i++) if (vk->wcache[i].slot == s) { used = 1; break; }
        if (!used) { slot = s; break; }
    }
    if (slot < 0) return -1;
    if (pool_slot(vk, slot, sz)) return -1;
    upload(vk, slot, data, sz);   /* data NULL = zero-fill (missing bias) */
    if (vk->wn < 256) {
        snprintf(vk->wcache[vk->wn].name, sizeof(vk->wcache[vk->wn].name), "%s", name);
        vk->wcache[vk->wn].slot = slot;
        vk->wn++;
    }
    return slot;
}

static void download(WuBuVk *vk, int slot, void *out, size_t sz) {
    void *p = NULL;
    vkMapMemory(vk->dev, vk->pmem[slot], 0, vk->psize[slot], 0, &p);
    if (!mem_coherent(vk, slot)) {
        VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                  .memory = vk->pmem[slot], .offset = 0, .size = sz };
        vkInvalidateMappedMemoryRanges(vk->dev, 1, &r);
    }
    memcpy(out, p, sz);
    vkUnmapMemory(vk->dev, vk->pmem[slot]);
}

/* allocate a fresh descriptor set from the per-dispatch pool (grows lazily;
 * wraps for the public single-op APIs whose per-call submits make reuse safe).
 * In no_flush (single-buffer) mode wrap is FORBIDDEN — the wrapped sets would
 * be rebound while earlier recorded dispatches still reference them. */
static VkDescriptorSet dspool_get(WuBuVk *vk, VkDescriptorSetLayout dsl) {
    if (vk->pipe_submit) {
        if (vk->dspool_n >= 1024) {
            if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG dspool EXHAUSTED n=%u\n", vk->dspool_n);
            return VK_NULL_HANDLE;   /* hard limit, never wrap */
        }
    } else if (vk->dspool_n >= 512) {
        vk->dspool_n = 0;   /* wrap: reuse old sets */
    }
    if (!vk->dspool) {
        VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 * 4 };
        VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                            .maxSets = 1024, .poolSizeCount = 1, .pPoolSizes = &dps };
        if (vkCreateDescriptorPool(vk->dev, &dpci, NULL, &vk->dspool) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    VkDescriptorSetLayout dsls[1] = { dsl };
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = vk->dspool, .descriptorSetCount = 1,
                                         .pSetLayouts = dsls };
    if (vkAllocateDescriptorSets(vk->dev, &dsai, &vk->dspool_sets[vk->dspool_n]) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return vk->dspool_sets[vk->dspool_n++];
}

/* bind the 4 conv buffers (in, w, b, out) into a descriptor set */
static void bind4(WuBuVk *vk, VkDescriptorSet ds, VkBuffer b0, VkBuffer b1,
                  VkBuffer b2, VkBuffer b3, size_t s0, size_t s1, size_t s2, size_t s3) {
    VkDescriptorBufferInfo dbi[4] = {
        {b0, 0, s0}, {b1, 0, s1}, {b2, 0, s2}, {b3, 0, s3},
    };
    VkWriteDescriptorSet wds[4] = {0};
    for (int i = 0; i < 4; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 4, wds, 0, NULL);
}

/* memory barrier between dispatches (storage-buffer dependencies need
 * explicit visibility on Vulkan — compute-to-compute is NOT implicit).
 * In no_flush (single-buffer) mode the barrier is the ONLY sync, so it
 * must always be recorded; the submit-per-op debug path handles sync
 * through the flush boundary instead.
 *
 * MUST be the narrow compute-to-compute storage-buffer barrier, NOT
 * VK_PIPELINE_STAGE_ALL_COMMANDS_BIT. The full-pipeline barrier drains
 * the whole GPU between every dispatch — measured ~30ms x 588 barriers
 * = ~18s of the VK wall time on this NVIDIA driver (the generator was
 * 2x slower than CPU for that reason, not because of submits). */
static void rec_sync(WuBuVk *vk) {
    if (getenv("WUBU_VK_NOBARRIER")) return;   /* TIMING ONLY: unsafe, no sync */
    /* Per-op mode: the submit+waitIdle boundary IS the sync — no barrier
     * needed (and the full 200-entry buffer-barrier list costs recording
     * time per dispatch). The single-buffer (no_flush) mode needs the
     * barriers but is NOT reliable on this NVIDIA driver (see generator
     * comment) — keep them only for the experiment. */
    if (!vk->no_flush) return;
    if (!vk->no_flush && getenv("WUBU_VK_SUBMIT_PER_OP")) return; /* flush path handles sync */
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(vk->cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
}

/* per-op flush: submit + wait + re-begin after each recorded dispatch.
 * In no_flush mode this is a no-op — the generator records ALL dispatches
 * into one command buffer (with barriers between them) and submits once at
 * the end. The old comment claimed the barrier-only path "does not produce
 * correct results on this NVIDIA driver"; that predates the max-upfront
 * pool allocation (610bfa2) which fixed the freed-buffer-in-recorded-
 * dispatch bug that made the single-buffer path unsafe. The single-buffer
 * path is verified correct by the CPU A/B (corr 1.0). */
static void rec_flush(WuBuVk *vk) {
    if (vk->no_flush) return;
    vkEndCommandBuffer(vk->cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &vk->cb };
    VkResult sr;
    if (vk->pipe_submit) {
        /* pipelined: submit WITHOUT waiting. The submit boundary is the sync
         * between dispatches; the fence for this ring slot was already waited
         * (and reset) when we began re-recording it, so submitting with it
         * simply arms the wait for the NEXT time the slot wraps. */
        int slot = vk->cb_ring_n;
        sr = vkQueueSubmit(vk->q, 1, &si, vk->cb_fence[slot]);
    } else {
        sr = vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk->q);
    }
    if (getenv("WUBU_VK_DBG") && (sr != VK_SUCCESS))
        fprintf(stderr, "VKDBG flush err submit=%d\n", (int)sr);
    if (getenv("WUBU_VK_DBG") && vk->last_written >= 0 && vk->pmem[vk->last_written]) {
        float v0 = 0.0f;
        void *p = NULL;
        vkMapMemory(vk->dev, vk->pmem[vk->last_written], 0, 4, 0, &p);
        memcpy(&v0, p, 4);
        vkUnmapMemory(vk->dev, vk->pmem[vk->last_written]);
        fprintf(stderr, "VKDBG touch slot=%d v0=%.4f\n", vk->last_written, v0);
    }
    vk->last_written = -1;
    /* pipelined: use a fresh command buffer for the next dispatch. CRITICAL:
     * wait the TARGET slot's fence BEFORE re-recording it — the previous
     * submission using this ring slot may still be executing (the host races
     * ahead of the GPU). Re-beginning a cb the GPU is still reading is UB and
     * intermittently corrupts output. The wait at submit time below is too
     * late: vkBeginCommandBuffer(next) happens HERE, immediately. */
    VkCommandBuffer next = vk->cb;
    if (vk->pipe_submit) {
        int tslot = (vk->cb_ring_n + 1) % 8;
        if (vk->cb_fence[tslot] != VK_NULL_HANDLE) {
            vkWaitForFences(vk->dev, 1, &vk->cb_fence[tslot], VK_TRUE, UINT64_MAX);
            vkResetFences(vk->dev, 1, &vk->cb_fence[tslot]);
        }
        vk->cb_ring_n = tslot;
        next = vk->cb_ring[tslot];
        vk->cb = next;
    }
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(next, &cbbi);
}

/* record a conv1d dispatch (device-side chaining). in/w/b/out are pool slots.
 * is_convt selects the ConvTranspose1d pipeline; stride==1 && !is_convt routes
 * to the tiled smem pipeline (5x+ faster); epi fuses the activation epilogue
 * (0 none, 1 lrelu0.1, 2 tanh, 3 snake) into the store — the old separate
 * rec_act dispatch after the conv is then SKIPPED by the caller. */
static void rec_conv1d(WuBuVk *vk, int in_s, int w_s, int b_s, int out_s,
                       int in_ch, int n, int out_ch, int k, int stride,
                       int pad, int dil, int n_out, int is_convt, int epi) {
    rec_sync(vk);
    VkDescriptorSet ds = dspool_get(vk, vk->dsl_conv);
    if (ds == VK_NULL_HANDLE) return;
    bind4(vk, ds,
          vk->buffers[in_s], vk->buffers[w_s], vk->buffers[b_s], vk->buffers[out_s],
          (size_t)in_ch * n * 4, (size_t)(is_convt ? in_ch * out_ch : out_ch * in_ch) * k * 4,
          (size_t)out_ch * 4, (size_t)out_ch * n_out * 4);
    int use_tiled = (!is_convt && stride == 1);
    if (getenv("WUBU_VK_NO_TILED")) use_tiled = 0;   /* DA isolation: force naive path */
    if (getenv("WUBU_VK_DBG") && vk->no_flush)
        fprintf(stderr, "VKDBG rec_conv in=%d n=%d out=%d k=%d stride=%d %s\n",
                in_ch, n, out_ch, k, stride, use_tiled ? "TILED" : (is_convt ? "CONVT" : "NAIVE"));
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      is_convt ? vk->pipe_convt : (use_tiled ? vk->pipe_conv_t : vk->pipe_conv));
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            use_tiled ? vk->pl_conv_t : vk->pl_conv, 0, 1, &ds, 0, NULL);
    if (use_tiled) {
        int pc[9] = { in_ch, n, out_ch, k, stride, pad, dil, n_out, epi };
        vkCmdPushConstants(vk->cb, vk->pl_conv_t,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(vk->cb, (n_out + 127) / 128, (out_ch + 3) / 4, 1);
    } else {
        int pc[8] = { in_ch, n, out_ch, k, stride, pad, dil, n_out };
        vkCmdPushConstants(vk->cb, vk->pl_conv,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(vk->cb, (n_out + 127) / 128, (out_ch + 3) / 4, 1);
    }
    vk->last_written = out_s;
    rec_flush(vk);
}

/* record an elementwise op: mode 0 lrelu, 1 snake, 2 tanh, 3 +off per channel,
 * 4 *scalar; x is pool slot x_s, offsets slot o_s (may be -1). */
static void rec_act(WuBuVk *vk, int x_s, int o_s, int mode, int n_ch, int n, float sc) {
    if (x_s < 0) return;
    rec_sync(vk);
    VkDescriptorSet ds = dspool_get(vk, vk->dsl_act);
    if (ds == VK_NULL_HANDLE) return;
    if (o_s >= 0) {
        VkDescriptorBufferInfo dbi[2] = {
            {vk->buffers[x_s], 0, (size_t)n_ch * n * 4},
            {vk->buffers[o_s], 0, (size_t)n_ch * 4},
        };
        VkWriteDescriptorSet wds[2] = {0};
        for (int i = 0; i < 2; i++) {
            wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(vk->dev, 2, wds, 0, NULL);
    } else {
        VkDescriptorBufferInfo dbi[2] = {
            {vk->buffers[x_s], 0, (size_t)n_ch * n * 4},
            {vk->buffers[x_s], 0, 4},
        };
        VkWriteDescriptorSet wds[2] = {0};
        for (int i = 0; i < 2; i++) {
            wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(vk->dev, 2, wds, 0, NULL);
    }
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_act);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pl_act, 0, 1, &ds, 0, NULL);
    /* the shader declares {int mode; int n_ch; int n; float scalar} — push the
     * int bit patterns, NOT float-cast ints (3.0f != 3). */
    struct { int mode; int n_ch; int n; float sc; } pc = { mode, n_ch, n, sc };
    vkCmdPushConstants(vk->cb, vk->pl_act, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(vk->cb, ((size_t)n_ch * n + 255) / 256, 1, 1);
    vk->last_written = x_s;
    rec_flush(vk);
}

/* record an elt op: mode 0 a+=b, 1 a=b. a/b are pool slots; a_off/b_off are
 * element offsets within the slot (the MRF accumulator slices). */
static void rec_elt(WuBuVk *vk, int a_s, size_t a_off, int b_s, size_t b_off,
                    int mode, size_t n) {
    rec_sync(vk);
    VkDescriptorSet ds = dspool_get(vk, vk->dsl_elt);
    if (ds == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo dbi[2] = {
        {vk->buffers[a_s], a_off * 4, n * 4},
        {vk->buffers[b_s], b_off * 4, n * 4},
    };
    VkWriteDescriptorSet wds[2] = {0};
    for (int i = 0; i < 2; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(vk->dev, 2, wds, 0, NULL);
    vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_elt);
    vkCmdBindDescriptorSets(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pl_elt, 0, 1, &ds, 0, NULL);
    int pc[2] = { mode, (int)n };
    vkCmdPushConstants(vk->cb, vk->pl_elt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(vk->cb, (n + 255) / 256, 1, 1);
    vk->last_written = a_s;
    rec_flush(vk);
}

static int mk_pipe(WuBuVk *vk, const unsigned char *spv, size_t spv_len,
                   int n_binds, VkDescriptorSetLayout *dsl, VkPipelineLayout *pl,
                   VkPipeline *pipe, int pc_size) {
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = spv_len, .pCode = (const uint32_t *)spv };
    if (vkCreateShaderModule(vk->dev, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    VkDescriptorSetLayoutBinding binds[4] = {0};
    for (int i = 0; i < n_binds; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = n_binds, .pBindings = binds };
    vkCreateDescriptorSetLayout(vk->dev, &dsli, NULL, dsl);
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size };
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    if (vkCreatePipelineLayout(vk->dev, &pli, NULL, pl) != VK_SUCCESS) return -1;
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .module = sm, .pName = "main" };
    VkComputePipelineCreateInfo cp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                       .stage = ss, .layout = *pl };
    if (vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cp, NULL, pipe) != VK_SUCCESS) return -1;
    vkDestroyShaderModule(vk->dev, sm, NULL);
    return 0;
}

static VkDescriptorSet alloc_ds(WuBuVk *vk, int slot, VkDescriptorSetLayout dsl) {
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    if (vkCreateDescriptorPool(vk->dev, &dpci, NULL, &vk->dp[slot]) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = vk->dp[slot], .descriptorSetCount = 1,
                                         .pSetLayouts = &dsl };
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(vk->dev, &dsai, &ds) != VK_SUCCESS) return VK_NULL_HANDLE;
    return ds;
}

WuBuVk *wubu_vk_create(void) {
    WuBuVk *vk = (WuBuVk *)calloc(1, sizeof(WuBuVk));
    if (!vk) return NULL;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "wubumedia", .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &ai };
    if (vkCreateInstance(&ici, NULL, &vk->inst) != VK_SUCCESS) goto fail;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(vk->inst, &n, NULL);
    if (n < 1) goto fail;
    VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(n, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(vk->inst, &n, devs);
    vk->phys = devs[0];
    if (getenv("WUBU_VK_DBG")) {
        VkPhysicalDeviceProperties pdp;
        vkGetPhysicalDeviceProperties(vk->phys, &pdp);
        fprintf(stderr, "VKDBG device[0] = %s (api %d.%d.%d)\n", pdp.deviceName,
                VK_VERSION_MAJOR(pdp.apiVersion), VK_VERSION_MINOR(pdp.apiVersion),
                VK_VERSION_PATCH(pdp.apiVersion));
    }
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &nqf, NULL);
    VkQueueFamilyProperties *qfp = (VkQueueFamilyProperties *)calloc(nqf, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &nqf, qfp);
    vk->qfam = 0;
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { vk->qfam = i; break; }
    }
    free(qfp); free(devs);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                     .queueFamilyIndex = vk->qfam, .queueCount = 1,
                                     .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &dqci };
    if (vkCreateDevice(vk->phys, &dci, NULL, &vk->dev) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(vk->dev, vk->qfam, 0, &vk->q);
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                     .queueFamilyIndex = vk->qfam };
    vkCreateCommandPool(vk->dev, &cpci, NULL, &vk->pool);
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = vk->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    vkAllocateCommandBuffers(vk->dev, &cbai, &vk->cb);
    for (int i = 0; i < 8; i++) vk->cb_ring[i] = vk->cb;   /* default: all point to cb */
    VkCommandBufferAllocateInfo cbai8 = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = vk->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 8 };
    VkCommandBuffer ring8[8];
    if (vkAllocateCommandBuffers(vk->dev, &cbai8, ring8) == VK_SUCCESS)
        for (int i = 0; i < 8; i++) vk->cb_ring[i] = ring8[i];
    for (int i = 0; i < 8; i++) {
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                  .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        if (vkCreateFence(vk->dev, &fci, NULL, &vk->cb_fence[i]) != VK_SUCCESS)
            vk->cb_fence[i] = VK_NULL_HANDLE;
    }

    if (mk_pipe(vk, WUBU_VK_CONV_SPV, sizeof(WUBU_VK_CONV_SPV), 4,
                &vk->dsl_conv, &vk->pl_conv, &vk->pipe_conv, 8 * sizeof(int))) goto fail;
    if (mk_pipe(vk, WUBU_VK_CONV_TILED_SPV, sizeof(WUBU_VK_CONV_TILED_SPV), 4,
                &vk->dsl_conv_t, &vk->pl_conv_t, &vk->pipe_conv_t, 9 * sizeof(int))) goto fail;
    if (mk_pipe(vk, WUBU_VK_CONVT_SPV, sizeof(WUBU_VK_CONVT_SPV), 4,
                &vk->dsl_conv, &vk->pl_conv, &vk->pipe_convt, 8 * sizeof(int))) goto fail;
    if (mk_pipe(vk, WUBU_VK_ACT_SPV, sizeof(WUBU_VK_ACT_SPV), 2,
                &vk->dsl_act, &vk->pl_act, &vk->pipe_act, 4 * sizeof(float))) goto fail;
    if (mk_pipe(vk, WUBU_VK_ELT_SPV, sizeof(WUBU_VK_ELT_SPV), 2,
                &vk->dsl_elt, &vk->pl_elt, &vk->pipe_elt, 2 * sizeof(int))) goto fail;
    vk->ds_conv = alloc_ds(vk, 0, vk->dsl_conv);
    vk->ds_convt = alloc_ds(vk, 1, vk->dsl_conv);
    vk->ds_act = alloc_ds(vk, 2, vk->dsl_act);
    vk->ds_elt = alloc_ds(vk, 3, vk->dsl_elt);
    if (!vk->ds_conv || !vk->ds_convt || !vk->ds_act || !vk->ds_elt) goto fail;
    return vk;
fail:
    wubu_vk_destroy(vk);
    return NULL;
}

void wubu_vk_destroy(WuBuVk *vk) {
    if (!vk) return;
    if (vk->dev) {
        vkQueueWaitIdle(vk->q);
        for (int i = 0; i < WUBU_VK_POOL; i++) {
            if (vk->buffers[i]) vkDestroyBuffer(vk->dev, vk->buffers[i], NULL);
            if (vk->pmem[i]) vkFreeMemory(vk->dev, vk->pmem[i], NULL);
        }
        if (vk->pool) vkDestroyCommandPool(vk->dev, vk->pool, NULL);
        for (int i = 0; i < 8; i++)
            if (vk->cb_fence[i]) vkDestroyFence(vk->dev, vk->cb_fence[i], NULL);
        if (vk->pipe_conv) vkDestroyPipeline(vk->dev, vk->pipe_conv, NULL);
        if (vk->pipe_convt) vkDestroyPipeline(vk->dev, vk->pipe_convt, NULL);
        if (vk->pipe_conv_t) vkDestroyPipeline(vk->dev, vk->pipe_conv_t, NULL);
        if (vk->pipe_act) vkDestroyPipeline(vk->dev, vk->pipe_act, NULL);
        if (vk->pipe_elt) vkDestroyPipeline(vk->dev, vk->pipe_elt, NULL);
        if (vk->pl_conv) vkDestroyPipelineLayout(vk->dev, vk->pl_conv, NULL);
        if (vk->pl_conv_t) vkDestroyPipelineLayout(vk->dev, vk->pl_conv_t, NULL);
        if (vk->pl_act) vkDestroyPipelineLayout(vk->dev, vk->pl_act, NULL);
        if (vk->pl_elt) vkDestroyPipelineLayout(vk->dev, vk->pl_elt, NULL);
        if (vk->dsl_conv) vkDestroyDescriptorSetLayout(vk->dev, vk->dsl_conv, NULL);
        if (vk->dsl_conv_t) vkDestroyDescriptorSetLayout(vk->dev, vk->dsl_conv_t, NULL);
        if (vk->dsl_act) vkDestroyDescriptorSetLayout(vk->dev, vk->dsl_act, NULL);
        if (vk->dsl_elt) vkDestroyDescriptorSetLayout(vk->dev, vk->dsl_elt, NULL);
        for (int i = 0; i < 4; i++)
            if (vk->dp[i]) vkDestroyDescriptorPool(vk->dev, vk->dp[i], NULL);
        if (vk->dspool) vkDestroyDescriptorPool(vk->dev, vk->dspool, NULL);
        vkDestroyDevice(vk->dev, NULL);
    }
    if (vk->inst) vkDestroyInstance(vk->inst, NULL);
    free(vk);
}

/* ── public conv1d (synchronous, host buffers) ── */
int wubu_vk_conv1d(WuBuVk *vk,
                   const float *in, int in_ch, int n,
                   const float *w, const float *b,
                   int out_ch, int k, int stride, int pad, int dil,
                   float *out, int n_out) {
    if (!vk || !in || !w || !out) return -1;
    size_t in_sz = (size_t)in_ch * n * 4;
    size_t w_sz = (size_t)out_ch * in_ch * k * 4;
    size_t b_sz = (size_t)out_ch * 4;
    size_t out_sz = (size_t)out_ch * n_out * 4;
    if (pool_slot(vk, 0, in_sz) || pool_slot(vk, 1, w_sz) ||
        pool_slot(vk, 2, b_sz) || pool_slot(vk, 3, out_sz)) return -1;
    upload(vk, 0, in, in_sz);
    upload(vk, 1, w, w_sz);
    upload(vk, 2, b, b_sz);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(vk->cb, &cbbi);
    rec_conv1d(vk, 0, 1, 2, 3, in_ch, n, out_ch, k, stride, pad, dil, n_out, 0, 0);
    vkEndCommandBuffer(vk->cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &vk->cb };
    vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk->q);
    download(vk, 3, out, out_sz);
    return 0;
}

/* elt helper: mode 0 a+=b, 1 a=b — public for tests */
int wubu_vk_elt(WuBuVk *vk, float *a, const float *b, size_t n, int mode) {
    if (!vk || !a || !b) return -1;
    size_t sz = n * 4;
    if (pool_slot(vk, 0, sz) || pool_slot(vk, 1, sz)) return -1;
    upload(vk, 0, a, sz);
    upload(vk, 1, b, sz);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(vk->cb, &cbbi);
    rec_elt(vk, 0, 0, 1, 0, mode, n);
    vkEndCommandBuffer(vk->cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &vk->cb };
    vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk->q);
    download(vk, 0, a, sz);
    return 0;
}

/* ConvTranspose1d (gather) — same signature convention: w is [in_ch * out_ch * k]. */
int wubu_vk_convt1d(WuBuVk *vk,
                    const float *in, int in_ch, int n,
                    const float *w, const float *b,
                    int out_ch, int k, int stride, int pad,
                    float *out, int n_out) {
    if (!vk || !in || !w || !out) return -1;
    size_t in_sz = (size_t)in_ch * n * 4;
    size_t w_sz = (size_t)in_ch * out_ch * k * 4;
    size_t b_sz = (size_t)out_ch * 4;
    size_t out_sz = (size_t)out_ch * n_out * 4;
    if (pool_slot(vk, 0, in_sz) || pool_slot(vk, 1, w_sz) ||
        pool_slot(vk, 2, b_sz) || pool_slot(vk, 3, out_sz)) return -1;
    upload(vk, 0, in, in_sz);
    upload(vk, 1, w, w_sz);
    upload(vk, 2, b, b_sz);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(vk->cb, &cbbi);
    rec_conv1d(vk, 0, 1, 2, 3, in_ch, n, out_ch, k, stride, pad, 1, n_out, 1, 0);
    vkEndCommandBuffer(vk->cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &vk->cb };
    vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk->q);
    download(vk, 3, out, out_sz);
    return 0;
}

/* ── the full GeneratorNSF (mirrors wubu_generator_nsf_cuda) ── */
int wubu_vk_generator_nsf(WuBuVk *vk, WuBuRVCModel *model,
                          const float *z, int nF, int inter_channels,
                          const float *nsff0, const float *g,
                          float *out_audio, int max_samples,
                          int inject_noise, int use_snake) {
    (void)g;
    double tg0 = omp_get_wtime();   /* total generator wall (incl. record) */
    if (!vk || !model || !z || !out_audio) return -1;
    /* mode selection FIRST so the pipelined-drain check below sees the real
     * mode (a previous call left pipe_submit set — that's the flag we need). */
    vk->no_flush = (getenv("WUBU_VK_SINGLE_BUFFER") != NULL);  /* broken, do not ship */
    vk->pipe_submit = (getenv("WUBU_VK_WAIT_PER_OP") == NULL); /* default: pipeline */
    /* Pipelined mode: the PREVIOUS generator call's dispatches may still be
     * executing. EVERY host write to a pool buffer below (zero buffer, z,
     * sine, cond, fresh weight slots) would race the GPU reads of the prior
     * call. Drain the queue BEFORE the first host upload. */
    if (vk->pipe_submit) vkQueueWaitIdle(vk->q);
    if (getenv("WUBU_VK_DUMP"))
        fprintf(stderr, "[vkg] generator_nsf entered nF=%d inter=%d ups_total=%d\n",
                nF, inter_channels,
                model->n_upsample_layers > 0 ? model->n_upsample_layers : 4);

    int n_ups = model->n_upsample_layers > 0 ? model->n_upsample_layers : 4;
    const RVCTensor *pre_w0 = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    int init_ch = pre_w0 && pre_w0->dims[0] > 0 ? pre_w0->dims[0] : 512;
    int ups_in[8], ups_out[8], ups_rate[8], ups_k[8], ups_pad[8];
    int ups_total = 1;
    char kb[64];
    for (int L = 0; L < n_ups; L++) {
        ups_in[L] = init_ch / (1 << L);
        ups_out[L] = init_ch / (1 << (L + 1));
        ups_rate[L] = model->upsample_rates[L] > 0 ? model->upsample_rates[L] : 2;
        kfmt(kb, "dec.ups.", L, "", -1, ".weight_v");
        const RVCTensor *ut = wubu_rvc_find_tensor(model, kb);
        if (!ut) { kfmt(kb, "dec.ups.", L, "", -1, ".weight"); ut = wubu_rvc_find_tensor(model, kb); }
        ups_k[L] = ut && ut->n_dims >= 3 ? ut->dims[2] : 4;
        ups_pad[L] = (ups_k[L] - ups_rate[L]) / 2;
        if (ups_pad[L] < 0) ups_pad[L] = 0;
        ups_total *= ups_rate[L];
    }

    const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    if (!pre_w || !pre_w->data) return -1;
    int cur_ch = ups_in[0], cur_n = nF;

    /* host scratch */
    float *sine = (float *)malloc((size_t)(nF * ups_total + 8) * sizeof(float));
    float *rad = (float *)malloc((size_t)nF * sizeof(float));
    float *rad_acc = (float *)malloc((size_t)nF * sizeof(float));
    if (!sine || !rad || !rad_acc) { free(sine); free(rad); free(rad_acc); return -1; }

    /* pool slots: 0 z, 1 x, 2 sine, 3 cur, 4 tmp(stage), 5 tmp2, 6 acc, 7 stage-preserve,
     * 8 out, 9..13 weight/bias scratch (per-op uploads), 14 nc (noise out) */
    size_t z_sz = (size_t)inter_channels * nF * 4;
    size_t x_sz = (size_t)cur_ch * cur_n * 4;
    size_t sine_sz = (size_t)(nF * ups_total + 8) * 4;
    size_t out_sz = (size_t)(nF * ups_total + 8) * 4;
    size_t max_acc = 0;
    /* precompute ALL the layer sizes and allocate the working slots at their
     * MAX up front — the pool realloc between recorded dispatches is fatal
     * (the freed buffers the dispatches still reference) */
    {
        size_t max_stage = 0;
        int ich = init_ch, in_n = nF;
        for (int L = 0; L < n_ups; L++) {
            int och = init_ch / (1 << (L + 1));
            int on = (in_n - 1) * ups_rate[L] - 2 * ups_pad[L] + ups_k[L];
            if (on <= 0) return -1;
            size_t st = (size_t)och * on * 4;
            if (st > max_stage) max_stage = st;
            if (st * 3 > max_acc) max_acc = st * 3;
            ich = och; in_n = on;
        }
        if (max_stage == 0) return -1;
        if (pool_slot(vk, 3, max_stage) || pool_slot(vk, 4, max_stage) ||
            pool_slot(vk, 5, max_stage) || pool_slot(vk, 7, max_stage) ||
            pool_slot(vk, 14, max_stage) || pool_slot(vk, 6, max_acc)) return -1;
    }
    if (pool_slot(vk, 0, z_sz) || pool_slot(vk, 1, x_sz) || pool_slot(vk, 2, sine_sz) ||
        pool_slot(vk, 8, out_sz)) return -1;
    /* persistent zero buffer (slot 199) for GPU-side acc zeroing — written ONCE
     * per generator call BEFORE any dispatch, so no host/GPU race. Slot is
     * RESERVED: wslot() stops at WUBU_VK_ZERO_SLOT so no weight tensor can
     * ever overwrite it (that clobbering was a real bug — weights landed on
     * slot 19, corrupted the acc-zero copy, and silently broke the MRF). */
    if (pool_slot(vk, WUBU_VK_ZERO_SLOT, max_acc)) return -1;
    upload(vk, WUBU_VK_ZERO_SLOT, NULL, max_acc);
    upload(vk, 0, z, z_sz);

    /* fresh descriptor sets per generator call. The waitIdle at the top of
     * this function already drained the previous call (pipelined mode), so
     * the pool reset is safe in all modes. */
    if (vk->dspool) {
        vkResetDescriptorPool(vk->dev, vk->dspool, 0);
        vk->dspool_n = 0;
    }
    /* defensive: a previous generator call that errored mid-record may have
     * left no_flush set and the command buffer open. Reset both so this
     * call records cleanly. */
    vk->no_flush = 0;
    /* In pipelined mode the ring MUST restart at slot 0 — vk->cb may point at
     * an arbitrary ring slot left by the previous call, and cb_ring_n resets
     * to 0 below. Mismatch = fence guards the wrong buffer = corruption.
     * The top waitIdle already drained the previous call, so every ring
     * fence is signaled; reset slot 0 so the first submit arms it cleanly. */
    if (vk->pipe_submit) {
        vk->cb = vk->cb_ring[0];
        if (vk->cb_fence[0] != VK_NULL_HANDLE)
            vkResetFences(vk->dev, 1, &vk->cb_fence[0]);
    }
    if (getenv("WUBU_VK_SUBMIT_PER_OP") == NULL)
        vkResetCommandBuffer(vk->cb, 0);

    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(vk->cb, &cbbi);
    /* DEFAULT: per-op pipelined submits. Each dispatch is its own submit with
     * the submit boundary as sync (the ONLY sync this NVIDIA driver honors
     * between dispatches — single-buffer barriers produce garbage, verified
     * by stage dumps). With the weight cache (wslot), weight slots are never
     * rewritten mid-record, so the host can race ahead and submit the next
     * dispatch while the GPU executes the previous one — the queue orders
     * them. One vkQueueWaitIdle at the end. The old per-op (wait per
     * dispatch) is the fallback; WUBU_VK_SINGLE_BUFFER opts into the broken
     * barrier experiment. */
    vk->cb_ring_n = 0;

    /* conv_pre — weights CACHED (uploaded once, resident across chunks) */
    size_t pre_w_sz = (size_t)pre_w->dims[0] * pre_w->dims[1] * pre_w->dims[2] * 4;
    size_t pre_b_sz = (size_t)pre_w->dims[0] * 4;
    int pre_w_s = wslot(vk, "dec.conv_pre.weight", pre_w->data, pre_w_sz);
    int pre_b_s = wslot(vk, "dec.conv_pre.bias", pre_b && pre_b->data ? pre_b->data : NULL, pre_b_sz);
    if (pre_w_s < 0 || pre_b_s < 0) return -1;
    int pk = pre_w->dims[2];
    rec_conv1d(vk, 0, pre_w_s, pre_b_s, 1, inter_channels, nF, cur_ch, pk, 1, pk / 2, 1, cur_n, 0, 0);
    /* cond(g): per-channel offset = cond_b[oc] + sum_ic g[ic]*cond_w[oc,ic] */
    {
        const RVCTensor *cond_w = wubu_rvc_find_tensor(model, "dec.cond.weight");
        const RVCTensor *cond_b = wubu_rvc_find_tensor(model, "dec.cond.bias");
        if (cond_w && cond_w->data) {
            int gin = cond_w->n_dims >= 2 ? cond_w->dims[1] : 40;
            float *off = (float *)malloc((size_t)cur_ch * sizeof(float));
            for (int oc = 0; oc < cur_ch; oc++) {
                float acc = cond_b && cond_b->data ? cond_b->data[oc] : 0.0f;
                const float *wv = cond_w->data + (size_t)oc * gin;
                for (int ic = 0; ic < gin && ic < 256; ic++) acc += g[ic] * wv[ic];
                off[oc] = acc;
            }
            if (pool_slot(vk, 11, (size_t)cur_ch * 4)) return -1;
            upload(vk, 11, off, (size_t)cur_ch * 4);
            rec_act(vk, 1, 11, 3, cur_ch, cur_n, 0.0f);
            free(off);
        }
    }
    if (getenv("WUBU_VK_DUMP")) {
        float *tmp = (float *)malloc((size_t)cur_ch * cur_n * 4);
        download(vk, 1, tmp, (size_t)cur_ch * cur_n * 4);
        FILE *df = fopen("outputs/rvc_ref/vk_pre.npy", "wb");
        if (df) { fwrite(tmp, sizeof(float), (size_t)cur_ch * cur_n, df); fclose(df);
                  fprintf(stderr, "[vkg] vk_pre.npy (%d x %d)\n", cur_ch, cur_n); }
        free(tmp);
    }

    /* sine excitation (host, exact CPU formula) */
    {
        float sr = model->sample_rate > 0 ? (float)model->sample_rate : 40000.0f;
        /* CRITICAL: the real tensor is dec.m_source.l_linear.* — the old
         * "dec.sine_linear.*" name doesn't exist in RVC models, so linw
         * fell back to 1.0. For Cartman l_linear.weight = -0.817383, so the
         * sine came out UN-inverted while the CPU path (which reads the real
         * tensor) was flipped → corr -0.994, maxdiff 0.20 on the sine dump. */
        const RVCTensor *linw = wubu_rvc_find_tensor(model, "dec.m_source.l_linear.weight");
        const RVCTensor *linb = wubu_rvc_find_tensor(model, "dec.m_source.l_linear.bias");
        float linw_v = (linw && linw->data) ? linw->data[0] : 1.0f;
        float linb_v = (linb && linb->data) ? linb->data[0] : 0.0f;
        float accum = 0.0f;
        for (int t = 0; t < nF; t++) {
            float f0 = nsff0[t] > 0 ? nsff0[t] : 0.0f;
            rad[t] = f0 / sr;
            float c = rad[t] * (float)ups_total;
            float r2 = fmodf(c + 0.5f, 1.0f) - 0.5f;
            accum += r2;
            rad_acc[t] = fmodf(accum, 1.0f);
        }
        for (int j = 0; j < nF * ups_total; j++) {
            int fi = j / ups_total;
            if (fi >= nF) fi = nF - 1;
            int u = j % ups_total;
            float phase = rad[fi] * (float)(u + 1);
            if (fi > 0) phase += rad_acc[fi - 1];
            float s = sinf(2.0f * (float)M_PI * phase) * 0.1f;
            float uv = (nsff0[fi] > 0) ? 1.0f : 0.0f;
            float noise_amp = inject_noise ? (1.0f - uv) * 0.1f / 3.0f : 0.0f;
            float noise = noise_amp * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
            sine[j] = tanhf(linw_v * (s * uv + noise) + linb_v);
        }
        if (getenv("WUBU_VK_DUMP")) {
            fprintf(stderr, "[vkg] about to write vk_sine.npy (%d samples)\\n", nF * ups_total);
            FILE *df = fopen("outputs/rvc_ref/vk_sine.npy", "wb");
            if (df) { fwrite(sine, sizeof(float), (size_t)nF * ups_total, df); fclose(df); }
        }
        upload(vk, 2, sine, sine_sz);
    }
    free(sine); free(rad); free(rad_acc);

    /* d_cur = d_x */
    rec_elt(vk, 3, 0, 1, 0, 1, (size_t)cur_ch * cur_n);
    /* upsample blocks */
    for (int L = 0; L < n_ups; L++) {
        int in_ch = cur_ch, in_n = cur_n;
        if (getenv("WUBU_VK_DBG"))
            fprintf(stderr, "VKDBG L%d start cur=%dx%d\n", L, in_ch, in_n);
        int out_ch = ups_out[L], out_n = (in_n - 1) * ups_rate[L] - 2 * ups_pad[L] + ups_k[L];
        if (out_n <= 0) return -1;
        size_t stage_sz = (size_t)out_ch * out_n * 4;
        /* allocate ALL the layer's buffers BEFORE recording any dispatch that
         * uses them — the pool realloc must never free a buffer an already
         * recorded dispatch still references. */
        if (pool_slot(vk, 4, stage_sz) || pool_slot(vk, 5, stage_sz) ||
            pool_slot(vk, 6, (size_t)3 * out_ch * out_n * 4) ||
            pool_slot(vk, 7, stage_sz) ||
            pool_slot(vk, 3, (size_t)out_ch * out_n * 4) ||
            pool_slot(vk, 14, stage_sz)) return -1;

        rec_act(vk, 3, -1, use_snake ? 1 : 0, in_ch, in_n, 0.0f);

        /* convT1d: dec.ups.L */
        float *den = model->hifi_upsample_denorm[L];
        int den_len = model->hifi_upsample_denorm_len[L];
        if (!den || den_len <= 0) {
            if (getenv("WUBU_VK_DBG"))
                fprintf(stderr, "VKDBG denorm missing L=%d den=%p len=%d\n", L, (void *)den, den_len);
            return -1;
        }
        kfmt(kb, "dec.ups.", L, "", -1, ".bias");
        const RVCTensor *ub = wubu_rvc_find_tensor(model, kb);
        char wkey[96]; kfmt(wkey, "dec.ups.", L, "", -1, ".denorm");
        int up_w_s = wslot(vk, wkey, den, (size_t)den_len * 4);
        int up_b_s = wslot(vk, kb, ub && ub->data ? ub->data : NULL, (size_t)out_ch * 4);
        if (up_w_s < 0 || up_b_s < 0) return -1;
        rec_conv1d(vk, 3, up_w_s, up_b_s, 4, in_ch, in_n, out_ch, ups_k[L], ups_rate[L],
                   ups_pad[L], 1, out_n, 1, 0);
        if (L == 0 && getenv("WUBU_VK_DUMP")) {
            /* compare against CPUUP0 in wubu_rvc_real.c (stage after convT) */
            float *tmp = (float *)malloc((size_t)out_ch * out_n * 4);
            download(vk, 4, tmp, (size_t)out_ch * out_n * 4);
            fprintf(stderr, "VKUP0");
            for (int q = 0; q < 8; q++) fprintf(stderr, " %.3f", tmp[q]);
            fprintf(stderr, "\n");
            free(tmp);
        }
        if (getenv("WUBU_VK_DBG"))
            fprintf(stderr, "VKDBG L%d in=%dx%d out=%dx%d k=%d rate=%d pad=%d out_n=%d\n",
                    L, in_ch, in_n, out_ch, out_n, ups_k[L], ups_rate[L], ups_pad[L], out_n);

        /* noise conv: conv1d(sine -> out_ch, stride = product of remaining) */
        {
            int stride = 1;
            for (int j = L + 1; j < n_ups; j++) stride *= ups_rate[j];
            kfmt(kb, "dec.noise_convs.", L, "", -1, ".weight");
            const RVCTensor *ncw = wubu_rvc_find_tensor(model, kb);
            kfmt(kb, "dec.noise_convs.", L, "", -1, ".bias");
            const RVCTensor *ncb = wubu_rvc_find_tensor(model, kb);
            int kk = ncw && ncw->n_dims >= 3 ? ncw->dims[2] : ((L == n_ups - 1) ? 1 : stride * 2);
            int pad = (kk - stride) / 2;
            if (pad < 0) pad = 0;
            if (ncw && ncw->data) {
                size_t ncw_sz = (size_t)ncw->dims[0] * (size_t)ncw->dims[1] * ncw->dims[2] * 4;
                size_t ncb_sz = (size_t)out_ch * 4;
                char nk1[96]; kfmt(nk1, "dec.noise_convs.", L, "", -1, ".weight");
                char nk2[96]; kfmt(nk2, "dec.noise_convs.", L, "", -1, ".bias");
                int nw_s = wslot(vk, nk1, ncw->data, ncw_sz);
                int nb_s = wslot(vk, nk2, ncb && ncb->data ? ncb->data : NULL, ncb_sz);
                if (nw_s < 0 || nb_s < 0) return -1;
                if (pool_slot(vk, 14, stage_sz)) return -1;
                rec_conv1d(vk, 2, nw_s, nb_s, 14, 1, nF * ups_total, out_ch, kk, stride,
                           pad, 1, out_n, 0, 0);
                rec_elt(vk, 4, 0, 14, 0, 0, (size_t)out_ch * out_n);
            }
        }

        /* MRF: n_stacks x (n_pairs conv pairs) -> avg -> stage */
        {
            int ch = out_ch;
            int n_stacks = model->n_mrf_stacks;
            if (n_stacks < 1) n_stacks = 3;
            if (n_stacks > 8) n_stacks = 8;
            int n_pairs = model->n_resblock_pairs;
            if (n_pairs < 1) n_pairs = 3;
            if (n_pairs > 8) n_pairs = 8;
            size_t n2 = (size_t)ch * out_n;
            /* preserve the stage in slot 7 (the carry copies overwrite slot 4) */
            rec_elt(vk, 7, 0, 4, 0, 1, n2);
            /* zero the acc — host memset is safe in wait mode (each dispatch
             * is waited, so the previous layer's rec_elt(vk,6,...) has
             * finished). In pipelined mode the host write would race the GPU,
             * so use the GPU-side zero buffer (slot 19) there. */
            if (pool_slot(vk, 6, (size_t)n_stacks * ch * out_n * 4)) return -1;
            if (vk->pipe_submit)
                rec_elt(vk, 6, 0, WUBU_VK_ZERO_SLOT, 0, 1, (size_t)n_stacks * ch * out_n);
            else
                upload(vk, 6, NULL, (size_t)n_stacks * ch * out_n * 4);
            /* weight slots are CACHED per tensor (wslot) — no shared realloc,
             * no freed-memory bug, no re-upload per chunk */
            for (int s = 0; s < n_stacks; s++) {
                int rb = L * n_stacks + s;
                int k = model->resblock_k[s];
                if (k <= 0) k = (s == 0) ? 3 : (s == 1 ? 7 : 11);
                /* rb_in = stage; reset the residual carry to the stage */
                rec_elt(vk, 5, 0, 7, 0, 1, n2);
                rec_elt(vk, 4, 0, 7, 0, 1, n2);
                for (int cp = 0; cp < n_pairs; cp++) {
                    kfmt(kb, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                    const RVCTensor *r1v = wubu_rvc_find_tensor(model, kb);
                    kfmt(kb, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
                    const RVCTensor *r1b = wubu_rvc_find_tensor(model, kb);
                    kfmt(kb, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                    const RVCTensor *r2v = wubu_rvc_find_tensor(model, kb);
                    kfmt(kb, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
                    const RVCTensor *r2b = wubu_rvc_find_tensor(model, kb);
                    if (!r1v || !r2v || !r1v->data || !r2v->data) continue;
                    int dil = model->resblock_dil[s][cp];
                    if (dil <= 0) dil = 1 + 2 * cp;
                    int pad1 = dil * (k - 1) / 2;
                    int pad2 = k / 2;
                    size_t w1_sz = (size_t)r1v->dims[0] * (size_t)r1v->dims[1] * r1v->dims[2] * 4;
                    size_t w2_sz = (size_t)r2v->dims[0] * (size_t)r2v->dims[1] * r2v->dims[2] * 4;
                    size_t b1_sz = (size_t)ch * 4, b2_sz = (size_t)ch * 4;
                    char k1[96]; kfmt(k1, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                    char k2[96]; kfmt(k2, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                    char k3[96]; kfmt(k3, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
                    char k4[96]; kfmt(k4, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
                    int w1_s = wslot(vk, k1, r1v->data, w1_sz);
                    int w2_s = wslot(vk, k2, r2v->data, w2_sz);
                    int b1_s = wslot(vk, k3, r1b && r1b->data ? r1b->data : NULL, b1_sz);
                    int b2_s = wslot(vk, k4, r2b && r2b->data ? r2b->data : NULL, b2_sz);
                    if (w1_s < 0 || w2_s < 0 || b1_s < 0 || b2_s < 0) return -1;
                    /* x = act(rb_in); conv1; act; conv2; + residual
                     * conv1 FUSES the activation epilogue (epi 1=lrelu, 3=snake)
                     * into its store — the old separate rec_act after conv1 is
                     * skipped (C13: kernel fusion, one less dispatch + global
                     * round trip per pair). */
                    int epi1 = use_snake ? 3 : 1;
                    rec_act(vk, 5, -1, use_snake ? 1 : 0, ch, out_n, 0.0f);
                    rec_conv1d(vk, 5, w1_s, b1_s, 3, ch, out_n, ch, k, 1, pad1, dil, out_n, 0, epi1);
                    rec_conv1d(vk, 3, w2_s, b2_s, 5, ch, out_n, ch, k, 1, pad2, 1, out_n, 0, 0);
                    rec_elt(vk, 5, 0, 4, 0, 0, n2);
                    rec_elt(vk, 4, 0, 5, 0, 1, n2);   /* carry = the pair output */
                }
                /* acc[s] += rb_in */
                rec_elt(vk, 6, (size_t)s * ch * out_n, 5, 0, 0, n2);
            }
            /* stage = avg over stacks */
            rec_elt(vk, 4, 0, 6, 0, 1, n2);
            for (int s = 1; s < n_stacks; s++) rec_elt(vk, 4, 0, 6, (size_t)s * ch * out_n, 0, n2);
            rec_act(vk, 4, -1, 4, 1, (int)n2, 1.0f / (float)n_stacks);
            /* d_cur = stage */
            rec_elt(vk, 3, 0, 4, 0, 1, n2);
            cur_ch = ch; cur_n = out_n;
        }   /* end MRF */
    }   /* end upsample blocks */

    /* conv_post + tanh */
    {
        const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
        if (!post_w || !post_w->data) return -1;
        int post_in = post_w->n_dims >= 2 ? post_w->dims[1] : 32;
        if (post_in < 1) post_in = 32;
        int post_k = post_w->n_dims >= 3 ? post_w->dims[2] : 7;
        int post_pad = post_k / 2;
        int out_n = cur_n;
        if (out_n > max_samples) out_n = max_samples;
        rec_act(vk, 3, -1, use_snake ? 1 : 0, post_in, cur_n, 0.0f);
        size_t pw_sz = (size_t)post_w->dims[0] * (size_t)post_w->dims[1] * post_w->dims[2] * 4;
        int pw_s = wslot(vk, "dec.conv_post.weight", post_w->data, pw_sz);
        if (pw_s < 0) return -1;
        if (pool_slot(vk, 10, (size_t)4)) return -1;
        float zero = 0.0f;
        upload(vk, 10, &zero, 4);
        if (pool_slot(vk, 8, (size_t)out_n * 4)) return -1;
        /* conv_post FUSES the tanh epilogue (epi=2) — the old separate
         * rec_act(8, tanh) is skipped (C16). */
        rec_conv1d(vk, 3, pw_s, 10, 8, post_in, cur_n, 1, post_k, 1, post_pad, 1, out_n, 0, 2);
    }

    vkEndCommandBuffer(vk->cb);
    vk->no_flush = 0;   /* single-buffer mode ends here: submit once */
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &vk->cb };
    double t_rec = 0.0, t_sub = 0.0, t_dl = 0.0;
    if (getenv("WUBU_TIME_STAGES")) {
        double t0 = omp_get_wtime();
        VkResult sr = vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
        double t1 = omp_get_wtime();
        VkResult wr = vkQueueWaitIdle(vk->q);
        double t2 = omp_get_wtime();
        fprintf(stderr, "[vkg-time] nF=%d submit_call=%.3fs wait=%.3fs sr=%d wr=%d\n",
                nF, t1 - t0, t2 - t1, (int)sr, (int)wr);
        int out_n = cur_n;
        if (out_n > max_samples) out_n = max_samples;
        double t3 = omp_get_wtime();
        download(vk, 8, out_audio, (size_t)out_n * 4);
        double t4 = omp_get_wtime();
        fprintf(stderr, "[vkg-time] download=%.3fs TOTAL_incl_record=%.3fs\n", t4 - t3,
                omp_get_wtime() - tg0);
        return out_n;
    }
    vkQueueSubmit(vk->q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk->q);
    if (getenv("WUBU_TIME_STAGES"))
        fprintf(stderr, "[vkg-time] TOTAL generator incl record = %.3fs (nF=%d)\n",
                omp_get_wtime() - tg0, nF);
    if (getenv("WUBU_VK_DBG"))
        fprintf(stderr, "VKDBG end generator nF=%d dspool_n=%u max_acc=%zu cur_ch=%d cur_n=%d\n",
                nF, vk->dspool_n, max_acc, cur_ch, cur_n);
    int out_n = cur_n;
    if (out_n > max_samples) out_n = max_samples;
    download(vk, 8, out_audio, (size_t)out_n * 4);
    return out_n;
}
