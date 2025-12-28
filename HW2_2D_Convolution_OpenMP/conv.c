// conv.c
#include <stdlib.h>

void conv(int* M, int w, int* K, int k, int* C) {
    int pad = k / 2;

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < w; ++j) {
            int sum = 0;
            for (int u = 0; u < k; ++u) {
                int mi = i + (u - pad);
                if (mi < 0 || mi >= w) {
                    // entire row of kernel here maps outside in vertical direction: still need to check v loop but M is zero
                    for (int v = 0; v < k; ++v) {
                        // contribution is zero because M out of bounds
                        (void)v; // no-op to avoid warnings
                    }
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
