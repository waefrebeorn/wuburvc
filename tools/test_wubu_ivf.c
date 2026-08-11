/*
 * test_wubu_ivf.c — Triple-DA verification of the native C11 IVF reader.
 *
 * Loads a real .index file, runs wubu_ivf_search with the same queries and
 * nprobe the Python helper used, and prints the top-k ids/distances so the
 * Python side can compare against faiss ground truth (16/16 exact match).
 *
 * Usage: test_wubu_ivf <index> <queries.bin> <n_queries> <k> <nprobe>
 *        queries.bin = n_queries * d float32 (d read from the index).
 * Prints: "d=.. ntotal=.. nlist=.. nprobe=.."
 *         per query: "f:<id>:<dist>" for each of k results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_ivf.h"

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <index> <queries.bin> <n_queries> <k> <nprobe>\n", argv[0]);
        return 2;
    }
    const char *index_path = argv[1];
    const char *qpath = argv[2];
    int n_queries = atoi(argv[3]);
    int k = atoi(argv[4]);
    int nprobe = atoi(argv[5]);

    WuBuIVF *ivf = wubu_ivf_load(index_path);
    if (!ivf) {
        fprintf(stderr, "FAIL: wubu_ivf_load(%s) returned NULL\n", index_path);
        return 1;
    }

    int d = wubu_ivf_dim(ivf);
    fprintf(stderr, "d=%d ntotal=%lld nlist=%lld\n",
            d, (long long)wubu_ivf_ntotal(ivf), (long long)wubu_ivf_nlist(ivf));
    fprintf(stderr, "nprobe file=%d effective=%d\n",
            wubu_ivf_get_nprobe(ivf), nprobe);

    FILE *qf = fopen(qpath, "rb");
    if (!qf) { fprintf(stderr, "FAIL: cannot open %s\n", qpath); return 1; }
    float *queries = (float *)malloc((size_t)n_queries * d * sizeof(float));
    if (fread(queries, sizeof(float), (size_t)n_queries * d, qf) !=
        (size_t)n_queries * d) {
        fprintf(stderr, "FAIL: short read from %s\n", qpath);
        return 1;
    }
    fclose(qf);

    wubu_ivf_set_nprobe(ivf, nprobe);

    int64_t *ids = (int64_t *)malloc((size_t)n_queries * k * sizeof(int64_t));
    float *dists = (float *)malloc((size_t)n_queries * k * sizeof(float));
    if (wubu_ivf_search(ivf, queries, n_queries, k, ids, dists) != 0) {
        fprintf(stderr, "FAIL: search returned error\n");
        return 1;
    }

    /* Also verify get_vector round-trips on the retrieved ids */
    float *vec = (float *)malloc((size_t)d * sizeof(float));
    for (int f = 0; f < n_queries; f++) {
        for (int i = 0; i < k; i++) {
            int64_t gid = ids[(size_t)f * k + i];
            float ddist = dists[(size_t)f * k + i];
            if (gid >= 0 && wubu_ivf_get_vector(ivf, gid, vec) == 0) {
                /* recompute distance from the retrieved vector */
                const float *q = queries + (size_t)f * d;
                float s = 0.0f;
                for (int j = 0; j < d; j++) {
                    float df = q[j] - vec[j];
                    s += df * df;
                }
                fprintf(stderr, "verify f=%d i=%d id=%lld dist=%.4f recompute=%.4f %s\n",
                        f, i, (long long)gid, ddist, s,
                        (s - ddist) * (s - ddist) < 1e-3f * (1.0f + ddist) ? "OK" : "BAD");
            } else {
                fprintf(stderr, "verify f=%d i=%d id=%lld (unretrievable)\n",
                        f, i, (long long)gid);
            }
        }
    }

    /* Machine-readable results for the Python comparator */
    printf("d=%d ntotal=%lld nlist=%lld nprobe=%d\n",
           d, (long long)wubu_ivf_ntotal(ivf), (long long)wubu_ivf_nlist(ivf), nprobe);
    for (int f = 0; f < n_queries; f++) {
        for (int i = 0; i < k; i++) {
            printf("r %d %d %lld %.6f\n", f, i,
                   (long long)ids[(size_t)f * k + i], dists[(size_t)f * k + i]);
        }
    }

    free(vec);
    free(ids);
    free(dists);
    free(queries);
    wubu_ivf_free(ivf);
    return 0;
}
