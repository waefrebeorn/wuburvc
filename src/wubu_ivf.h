/*
 * wubu_ivf.h — native FAISS IndexIVFFlat reader + exact searcher (C11).
 *
 * We make our own FAISS. No Python, no subprocess, no third-party parsing.
 * This module reads the serialized FAISS IndexIVFFlat binary format
 * (fourcc "IwFl", quantizer "IxF2", invlists "ilar") and performs the
 * same coarse-quantize → nprobe-list → exact-L2 top-k search the real
 * library does, plus global-ID vector retrieval for the RVC blend.
 *
 * Self-contained: no god headers, opaque struct, C11 only.
 */
#ifndef WUBU_IVF_H
#define WUBU_IVF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WuBuIVF WuBuIVF;

/* Load a FAISS IndexIVFFlat binary .index file. Returns NULL on any
 * parse error (file missing, wrong magic, unsupported variant). */
WuBuIVF *wubu_ivf_load(const char *path);

/* Free the index and all owned memory. */
void wubu_ivf_free(WuBuIVF *ivf);

/* Accessors */
int      wubu_ivf_dim(const WuBuIVF *ivf);
int64_t  wubu_ivf_ntotal(const WuBuIVF *ivf);
int64_t  wubu_ivf_nlist(const WuBuIVF *ivf);

/* Override the number of coarse lists scanned per query.
 * 0 = keep the file's stored nprobe (or the load default). */
void wubu_ivf_set_nprobe(WuBuIVF *ivf, int nprobe);
int  wubu_ivf_get_nprobe(const WuBuIVF *ivf);

/* Exact IVF search.
 * queries:      [n_queries * d] float32 row-major
 * out_ids:      [n_queries * k] global vector ids (int64), best-first
 * out_dists:    [n_queries * k] squared L2 distances (float32)
 * Returns 0 on success, -1 on bad args.
 * If fewer than k vectors exist in the probed lists, remaining slots get
 * id -1 and distance +inf. */
int wubu_ivf_search(const WuBuIVF *ivf,
                    const float *queries, int n_queries, int k,
                    int64_t *out_ids, float *out_dists);

/* Retrieve one vector by global id. out_vec must hold d floats.
 * Returns 0 on success, -1 if id is unknown. */
int wubu_ivf_get_vector(const WuBuIVF *ivf, int64_t id, float *out_vec);

/* Search + gather neighbor vectors in one call (the RVC retrieval path).
 * out_vectors: [n_queries * k * d] float32 (row-major per neighbor)
 * out_dists:   [n_queries * k] float32
 * Returns 0 on success. */
int wubu_ivf_search_vectors(const WuBuIVF *ivf,
                            const float *queries, int n_queries, int k,
                            float *out_vectors, float *out_dists);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_IVF_H */
