// conv_openmp.c
#include <stdlib.h>
#include <omp.h>

void conv_openmp(int* M, int w, int* K, int k, int* C) {
    int pad = k / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < w; ++j) {
            int sum = 0;
            for (int u = 0; u < k; ++u) {
                int mi = i + (u - pad);
                if (mi < 0 || mi >= w) {
                    // skip contributions where row is out of bounds
                    continue;
                }
                for (int v = 0; v < k; ++v) {
                    int mj = j + (v - pad);
                    if (mj < 0 || mj >= w) continue;
                    int mval = M[mi * w + mj];
                    int kval = K[u * k + v];
                    sum += mval * kval;
                }
            }
            C[i * w + j] = sum;
        }
    }
}
