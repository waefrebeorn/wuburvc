/*
 * wubu_ivf.c — native FAISS IndexIVFFlat reader + exact searcher (C11).
 *
 * "We make our own faiss." This parses the serialized FAISS IndexIVFFlat
 * binary format directly (no Python, no subprocess) and reproduces the
 * reference search semantics:
 *   1. coarse quantizer (IndexFlatL2 centroids) → nprobe nearest lists
 *   2. exact L2 scan inside each probed inverted list
 *   3. global top-k via a max-heap
 *
 * Serialized layout (verified byte-for-byte against faiss 1.7.4 writer):
 *   fourcc "IwFl" (IndexIVFFlat)
 *   int32  d
 *   int64  ntotal
 *   int64  dummy = 1<<20  (×2, written by faiss for compat)
 *   u8     is_trained
 *   int32  metric_type
 *   int64  nlist
 *   int32  nprobe
 *   int32  max_codes
 *   quantizer: fourcc "IxF2" (IndexFlatL2), then its own header
 *              (int32 d, int64 ntotal, dummy×2, u8 is_trained,
 *               int32 metric, int64 xb_count, then nlist*d float32 centroids)
 *   direct_map: u8 type + u64 array_len + array_len*int64  (NoMap: 0,0)
 *   invlists: fourcc "ilar"
 *             u64 nlist, u64 code_size, fourcc "full"/"sprs",
 *             u64 sizes_len + sizes_len*u64 (full) or (list,size) pairs (sprs)
 *             then per list: n*code_size codes + n*int64 ids
 *
 * All integers little-endian. Self-contained module: no god headers.
 */
#include "wubu_ivf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Little-endian readers over a byte buffer. */
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int32_t rd_i32(const uint8_t *p) { return (int32_t)rd_u32(p); }
static uint64_t rd_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static int64_t rd_i64(const uint8_t *p) { return (int64_t)rd_u64(p); }
static float rd_f32(const uint8_t *p) {
    uint32_t u = rd_u32(p);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* One inverted list: pointers into the mmap'd file buffer. */
typedef struct {
    const float  *codes;   /* [n * d] float32 */
    const int64_t *ids;    /* [n] global ids */
    int64_t n;
} WuBuIVFList;

struct WuBuIVF {
    int     d;
    int64_t ntotal;
    int64_t nlist;
    int     nprobe;        /* effective nprobe used for search */
    int     file_nprobe;   /* nprobe stored in the file */
    int     metric;        /* 1 = L2 (only L2 supported) */

    const float *centroids;  /* [nlist * d] — points into buf */
    WuBuIVFList *lists;      /* [nlist] — codes/ids point into buf */

    /* Global-id → (list, offset) lookup. Dense when ids are in [0, ntotal):
     * id_to_li[id] = list, id_to_off[id] = offset. */
    int32_t *id_to_li;
    int32_t *id_to_off;
    int     dense_ids;      /* 1 if the dense map covers all ids */

    uint8_t *buf;           /* whole file, owned */
    size_t   buf_len;
};

/* ── loader helpers ─────────────────────────────────────────────── */
static int ivf_need(WuBuIVF *ivf, size_t off, size_t n) {
    return off <= ivf->buf_len && n <= ivf->buf_len - off;
}

WuBuIVF *wubu_ivf_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || (size_t)sz > (size_t)1 << 34) { fclose(f); return NULL; }
    rewind(f);

    WuBuIVF *ivf = (WuBuIVF *)calloc(1, sizeof(WuBuIVF));
    if (!ivf) { fclose(f); return NULL; }

    ivf->buf = (uint8_t *)malloc((size_t)sz);
    if (!ivf->buf) { free(ivf); fclose(f); return NULL; }
    ivf->buf_len = (size_t)sz;

    if (fread(ivf->buf, 1, ivf->buf_len, f) != ivf->buf_len) {
        fclose(f); wubu_ivf_free(ivf); return NULL;
    }
    fclose(f);

    const uint8_t *p = ivf->buf;
    size_t off = 0;

    /* fourcc */
    if (ivf->buf_len < 4 || memcmp(p, "IwFl", 4) != 0) {
        wubu_ivf_free(ivf); return NULL;
    }
    off = 4;

    /* header */
    if (!ivf_need(ivf, off, 4 + 8 + 8 + 8 + 1 + 4 + 8 + 4 + 4)) goto fail;
    ivf->d = rd_i32(p + off); off += 4;
    ivf->ntotal = rd_i64(p + off); off += 8;
    /* two dummy int64 (1<<20) — skip */
    off += 16;
    /* is_trained u8 — skip */
    off += 1;
    ivf->metric = rd_i32(p + off); off += 4;
    ivf->nlist = rd_i64(p + off); off += 8;
    ivf->file_nprobe = rd_i32(p + off); off += 4;
    ivf->nprobe = ivf->file_nprobe > 0 ? ivf->file_nprobe : 1;
    /* max_codes int32 — skip */
    off += 4;

    if (ivf->d <= 0 || ivf->d > 4096 || ivf->ntotal <= 0 ||
        ivf->nlist <= 0 || ivf->nlist > 10000000) goto fail;
    if (ivf->metric != 1) {
        /* Only L2 indexes are supported (RVC uses METRIC_L2). */
        wubu_ivf_free(ivf); return NULL;
    }

    /* quantizer: fourcc "IxF2", then IndexFlat header + xb floats */
    if (!ivf_need(ivf, off, 4)) goto fail;
    if (memcmp(p + off, "IxF2", 4) != 0) {
        /* quantizer must be flat L2 for IVFFlat */
        wubu_ivf_free(ivf); return NULL;
    }
    off += 4;
    if (!ivf_need(ivf, off, 4 + 8 + 8 + 8 + 1 + 4 + 8)) goto fail;
    /* q d (int32), q ntotal (int64) — should match */
    off += 4 + 8;
    /* q dummy ×2 */
    off += 16;
    /* q is_trained u8, q metric int32 */
    off += 1 + 4;
    int64_t xb_count = rd_i64(p + off); off += 8;
    if (xb_count != ivf->nlist * ivf->d) goto fail;
    if (!ivf_need(ivf, off, (size_t)xb_count * 4)) goto fail;
    ivf->centroids = (const float *)(p + off);
    off += (size_t)xb_count * 4;

    /* direct_map: u8 type + u64 array_len + array_len*int64 */
    if (!ivf_need(ivf, off, 1 + 8)) goto fail;
    uint8_t dm_type = p[off]; off += 1;
    uint64_t dm_len = rd_u64(p + off); off += 8;
    if (dm_len > ivf->buf_len / 8) goto fail;
    off += (size_t)dm_len * 8;
    (void)dm_type;

    /* invlists: fourcc "ilar" */
    if (!ivf_need(ivf, off, 4)) goto fail;
    if (memcmp(p + off, "ilar", 4) != 0) goto fail;
    off += 4;
    if (!ivf_need(ivf, off, 8 + 8 + 4)) goto fail;
    int64_t il_nlist = rd_i64(p + off); off += 8;
    int64_t code_size = rd_i64(p + off); off += 8;
    if (il_nlist != ivf->nlist) goto fail;
    /* For IVFFlat, codes are raw float32 vectors: code_size == d*4 */
    if (code_size != (int64_t)ivf->d * 4) {
        /* PQ/compressed variants unsupported — we own flat only */
        wubu_ivf_free(ivf); return NULL;
    }

    /* list type: "full" (all sizes) or "sprs" ((list,size) pairs) */
    const char *lt = (const char *)(p + off); off += 4;
    int is_sprs = 0;
    if (memcmp(lt, "full", 4) == 0) is_sprs = 0;
    else if (memcmp(lt, "sprs", 4) == 0) is_sprs = 1;
    else goto fail;

    int64_t sizes_len = rd_i64(p + off); off += 8;
    if (sizes_len <= 0 || sizes_len > ivf->nlist * 2 + 8) goto fail;
    if (!ivf_need(ivf, off, (size_t)sizes_len * 8)) goto fail;

    int64_t *sizes = (int64_t *)calloc((size_t)ivf->nlist, sizeof(int64_t));
    if (!sizes) goto fail;

    if (!is_sprs) {
        if (sizes_len != ivf->nlist) { free(sizes); goto fail; }
        for (int64_t i = 0; i < ivf->nlist; i++) {
            sizes[i] = rd_i64(p + off + (size_t)i * 8);
        }
        off += (size_t)sizes_len * 8;
    } else {
        /* pairs: (list_index, size) */
        for (int64_t j = 0; j + 1 < sizes_len; j += 2) {
            int64_t li = rd_i64(p + off + (size_t)j * 8);
            int64_t n  = rd_i64(p + off + (size_t)(j + 1) * 8);
            if (li < 0 || li >= ivf->nlist || n < 0) { free(sizes); goto fail; }
            sizes[li] = n;
        }
        off += (size_t)sizes_len * 8;
    }

    /* Per-list codes + ids (contiguous in file) */
    ivf->lists = (WuBuIVFList *)calloc((size_t)ivf->nlist, sizeof(WuBuIVFList));
    if (!ivf->lists) { free(sizes); goto fail; }

    int64_t total_seen = 0;
    for (int64_t i = 0; i < ivf->nlist; i++) {
        int64_t n = sizes[i];
        ivf->lists[i].n = n;
        if (n <= 0) continue;
        if (!ivf_need(ivf, off, (size_t)n * (size_t)code_size + (size_t)n * 8)) {
            free(sizes); goto fail;
        }
        ivf->lists[i].codes = (const float *)(p + off);
        off += (size_t)n * (size_t)code_size;
        ivf->lists[i].ids = (const int64_t *)(p + off);
        off += (size_t)n * 8;
        total_seen += n;
    }
    free(sizes);

    if (total_seen != ivf->ntotal) {
        /* id list may legitimately differ from ntotal on malformed files */
        goto fail;
    }

    /* Build global-id → (list, offset) lookup.
     * Dense path: all ids in [0, ntotal) → O(1) arrays. */
    int dense = 1;
    for (int64_t i = 0; i < ivf->nlist && dense; i++) {
        const int64_t *ids = ivf->lists[i].ids;
        for (int64_t j = 0; j < ivf->lists[i].n; j++) {
            if (ids[j] < 0 || ids[j] >= ivf->ntotal) { dense = 0; break; }
        }
    }
    if (dense && ivf->ntotal <= (int64_t)INT32_MAX) {
        ivf->id_to_li = (int32_t *)malloc((size_t)ivf->ntotal * sizeof(int32_t));
        ivf->id_to_off = (int32_t *)malloc((size_t)ivf->ntotal * sizeof(int32_t));
        if (ivf->id_to_li && ivf->id_to_off) {
            for (int64_t i = 0; i < ivf->ntotal; i++) {
                ivf->id_to_li[i] = -1;
                ivf->id_to_off[i] = -1;
            }
            for (int64_t i = 0; i < ivf->nlist; i++) {
                const int64_t *ids = ivf->lists[i].ids;
                for (int64_t j = 0; j < ivf->lists[i].n; j++) {
                    int64_t gid = ids[j];
                    ivf->id_to_li[gid] = (int32_t)i;
                    ivf->id_to_off[gid] = (int32_t)j;
                }
            }
            ivf->dense_ids = 1;
        } else {
            free(ivf->id_to_li); ivf->id_to_li = NULL;
            free(ivf->id_to_off); ivf->id_to_off = NULL;
        }
    }

    return ivf;

fail:
    wubu_ivf_free(ivf);
    return NULL;
}

void wubu_ivf_free(WuBuIVF *ivf) {
    if (!ivf) return;
    free(ivf->id_to_li);
    free(ivf->id_to_off);
    free(ivf->lists);
    free(ivf->buf);
    free(ivf);
}

int wubu_ivf_dim(const WuBuIVF *ivf) { return ivf ? ivf->d : 0; }
int64_t wubu_ivf_ntotal(const WuBuIVF *ivf) { return ivf ? ivf->ntotal : 0; }
int64_t wubu_ivf_nlist(const WuBuIVF *ivf) { return ivf ? ivf->nlist : 0; }

void wubu_ivf_set_nprobe(WuBuIVF *ivf, int nprobe) {
    if (!ivf) return;
    if (nprobe <= 0) nprobe = ivf->file_nprobe > 0 ? ivf->file_nprobe : 1;
    ivf->nprobe = nprobe;
}
int wubu_ivf_get_nprobe(const WuBuIVF *ivf) { return ivf ? ivf->nprobe : 0; }

/* ── search ─────────────────────────────────────────────────────── */

/* Squared L2 distance between query and vector v (both d floats). */
static float dist_l2(const float *a, const float *b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) {
        float df = a[i] - b[i];
        s += df * df;
    }
    return s;
}

/* Top-nprobe centroid selection: keeps (dist,list) pairs in a min order
 * array; when full, evicts the worst (largest) via simple scan. nprobe is
 * small (≤64), nlist ~1k → O(nlist*nprobe) is fine and cache-friendly. */
static void select_lists(const WuBuIVF *ivf, const float *q,
                         int64_t *out_lists, int want) {
    int nlist = (int)ivf->nlist;
    int nprobe = want < nlist ? want : nlist;

    /* result arrays: worst kept at index nprobe-1 (max dist) */
    float  *rd = (float *)malloc((size_t)nprobe * sizeof(float));
    int64_t *rl = (int64_t *)malloc((size_t)nprobe * sizeof(int64_t));
    if (!rd || !rl) { free(rd); free(rl); return; }
    for (int i = 0; i < nprobe; i++) { rd[i] = 1e30f; rl[i] = -1; }

    for (int li = 0; li < nlist; li++) {
        float d = dist_l2(q, ivf->centroids + (size_t)li * ivf->d, ivf->d);
        /* find worst kept */
        int worst = 0;
        for (int i = 1; i < nprobe; i++) if (rd[i] > rd[worst]) worst = i;
        if (d < rd[worst]) {
            rd[worst] = d;
            rl[worst] = li;
        }
    }

    for (int i = 0; i < nprobe; i++) out_lists[i] = rl[i];
    free(rd); free(rl);
}

/* Max-heap of size k for the global top-k. Root = worst (largest dist). */
typedef struct { float dist; int64_t id; } HeapItem;
static void heap_sift(HeapItem *h, int k, int i) {
    for (;;) {
        int l = i * 2 + 1, r = l + 1, m = i;
        if (l < k && h[l].dist > h[m].dist) m = l;
        if (r < k && h[r].dist > h[m].dist) m = r;
        if (m == i) break;
        HeapItem t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static void heap_push(HeapItem *h, int k, float dist, int64_t id) {
    if (k <= 0) return;
    if (dist < h[0].dist) {
        h[0].dist = dist;
        h[0].id = id;
        heap_sift(h, k, 0);
    }
}

int wubu_ivf_search(const WuBuIVF *ivf,
                    const float *queries, int n_queries, int k,
                    int64_t *out_ids, float *out_dists) {
    if (!ivf || !queries || !out_ids || !out_dists || n_queries < 1 || k < 1)
        return -1;
    if (ivf->nprobe < 1) return -1;

    int nprobe = ivf->nprobe < ivf->nlist ? ivf->nprobe : (int)ivf->nlist;
    int64_t *lists = (int64_t *)malloc((size_t)nprobe * sizeof(int64_t));
    HeapItem *heap = (HeapItem *)malloc((size_t)k * sizeof(HeapItem));
    if (!lists || !heap) { free(lists); free(heap); return -1; }

    for (int f = 0; f < n_queries; f++) {
        const float *q = queries + (size_t)f * ivf->d;

        for (int i = 0; i < k; i++) {
            heap[i].dist = 1e30f;
            heap[i].id = -1;
        }

        select_lists(ivf, q, lists, nprobe);
        for (int pi = 0; pi < nprobe; pi++) {
            int64_t li = lists[pi];
            if (li < 0) continue;
            const WuBuIVFList *L = &ivf->lists[li];
            for (int64_t j = 0; j < L->n; j++) {
                float d = dist_l2(q, L->codes + (size_t)j * ivf->d, ivf->d);
                heap_push(heap, k, d, L->ids[j]);
            }
        }

        /* heap root is worst; emit sorted best-first via reverse pop */
        for (int i = k - 1; i >= 0; i--) {
            out_dists[(size_t)f * k + i] = heap[0].dist;
            out_ids[(size_t)f * k + i] = heap[0].id;
            heap[0] = heap[i];
            heap_sift(heap, i, 0);
        }
    }

    free(lists);
    free(heap);
    return 0;
}

int wubu_ivf_get_vector(const WuBuIVF *ivf, int64_t id, float *out_vec) {
    if (!ivf || !out_vec) return -1;
    if (ivf->dense_ids) {
        if (id < 0 || id >= ivf->ntotal) return -1;
        int32_t li = ivf->id_to_li[id];
        int32_t oj = ivf->id_to_off[id];
        if (li < 0 || oj < 0) return -1;
        memcpy(out_vec, ivf->lists[li].codes + (size_t)oj * ivf->d,
               (size_t)ivf->d * sizeof(float));
        return 0;
    }
    /* fallback: linear scan (rare; ids non-dense) */
    for (int64_t i = 0; i < ivf->nlist; i++) {
        const WuBuIVFList *L = &ivf->lists[i];
        for (int64_t j = 0; j < L->n; j++) {
            if (L->ids[j] == id) {
                memcpy(out_vec, L->codes + (size_t)j * ivf->d,
                       (size_t)ivf->d * sizeof(float));
                return 0;
            }
        }
    }
    return -1;
}

int wubu_ivf_search_vectors(const WuBuIVF *ivf,
                            const float *queries, int n_queries, int k,
                            float *out_vectors, float *out_dists) {
    if (!ivf || !queries || !out_vectors || !out_dists) return -1;
    int64_t *ids = (int64_t *)malloc((size_t)n_queries * k * sizeof(int64_t));
    if (!ids) return -1;
    if (wubu_ivf_search(ivf, queries, n_queries, k, ids, out_dists) != 0) {
        free(ids); return -1;
    }
    for (int f = 0; f < n_queries; f++) {
        for (int i = 0; i < k; i++) {
            int64_t gid = ids[(size_t)f * k + i];
            float *dst = out_vectors + ((size_t)f * k + i) * ivf->d;
            if (gid < 0 || wubu_ivf_get_vector(ivf, gid, dst) != 0) {
                memset(dst, 0, (size_t)ivf->d * sizeof(float));
            }
        }
    }
    free(ids);
    return 0;
}
