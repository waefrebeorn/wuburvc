#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
    int d = 768;
    int n_queries = 4;
    float *queries = (float *)malloc((size_t)n_queries * d * sizeof(float));
    if (!queries) {
        fprintf(stderr, "Failed to allocate queries\n");
        return 1;
    }
    
    // Generate simple test data - all 1.0f
    for (int i = 0; i < n_queries * d; i++) {
        queries[i] = 1.0f;
    }
    
    FILE *f = fopen("C:/Users/eman5/wuburvc/build/tmp/tiny_q.bin", "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file\n");
        free(queries);
        return 1;
    }
    
    size_t written = fwrite(queries, sizeof(float), (size_t)n_queries * d, f);
    fclose(f);
    free(queries);
    
    if (written != (size_t)n_queries * d) {
        fprintf(stderr, "Failed to write all query data\n");
        return 1;
    }
    
    printf("Generated %d queries of dimension %d\n", n_queries, d);
    return 0;
}