#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.

// NOTE: feel free to add new macros

// NOTE: feel free to add new functions

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv) {
    // Optimized serial attention implementation with cache-friendly memory access
    double* scores = malloc(sizeof(double) * m * n);
    double scale = 1.0 / sqrt((double)dk);

    // Step 1: Compute scores = Q * K^T / sqrt(dk)
    // Optimized with loop unrolling and reduced memory access
    for (int i = 0; i < m; i++) {
        double* q_row = &Q[i * dk];
        double* score_row = &scores[i * n];
        
        for (int j = 0; j < n; j++) {
            double* k_row = &K[j * dk];
            double sum = 0.0;
            
            // Loop unrolling for better performance
            int k;
            for (k = 0; k < dk - 3; k += 4) {
                sum += q_row[k] * k_row[k];
                sum += q_row[k+1] * k_row[k+1];
                sum += q_row[k+2] * k_row[k+2];
                sum += q_row[k+3] * k_row[k+3];
            }
            // Handle remaining elements
            for (; k < dk; k++) {
                sum += q_row[k] * k_row[k];
            }
            
            score_row[j] = sum * scale;
        }
    }

    // Step 2: Apply softmax to each row of scores
    // Numerically stable softmax with single-pass optimization
    for (int i = 0; i < m; i++) {
        double* score_row = &scores[i * n];
        
        // Find max value for numerical stability
        double max_val = score_row[0];
        for (int j = 1; j < n; j++) {
            if (score_row[j] > max_val)
                max_val = score_row[j];
        }

        // Compute exp and sum in one pass
        double sum_exp = 0.0;
        for (int j = 0; j < n; j++) {
            double exp_val = exp(score_row[j] - max_val);
            score_row[j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        double inv_sum = 1.0 / sum_exp;
        for (int j = 0; j < n; j++) {
            score_row[j] *= inv_sum;
        }
    }

    // Step 3: result = scores * V
    // Cache-friendly matrix multiplication
    for (int i = 0; i < m; i++) {
        double* score_row = &scores[i * n];
        double* result_row = &result[i * dv];
        
        // Initialize result row
        for (int d = 0; d < dv; d++) {
            result_row[d] = 0.0;
        }
        
        // Accumulate weighted values
        for (int j = 0; j < n; j++) {
            double score_val = score_row[j];
            double* v_row = &V[j * dv];
            
            for (int d = 0; d < dv; d++) {
                result_row[d] += score_val * v_row[d];
            }
        }
    }

    free(scores);
}

// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 115 <template code>) <(tail -n 115 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double** M, size_t len, FILE* file) {
    *M = (double*) malloc(len * sizeof(double));
    if (fread(*M, sizeof(double), len, file) != len) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }
}

/*
 * Reads Q, K, and V matrices from the testing data file
 * File format:
 *   1. 4 integers: m, n, dk, dv
 *   2. m*dk doubles -> Q
 *   3. n*dk doubles -> K
 *   4. n*dv doubles -> V
 */
void read_matrices(const char* file_path, double** Q, double** K, double** V,
                  int *m, int *n, int *dk, int *dv) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(1);
    }

    if (fread(m, sizeof(int), 1, file) != 1 ||
        fread(n, sizeof(int), 1, file) != 1 ||
        fread(dk, sizeof(int), 1, file) != 1 ||
        fread(dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    read_matrix(Q, (*m) * (*dk), file);
    read_matrix(K, (*n) * (*dk), file);
    read_matrix(V, (*n) * (*dv), file);

    fclose(file);
}

bool verify(const char* file_path, const double* result) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open answer file: %s\n", file_path);
        return false;
    }

    int m, n, dk, dv;
    if (fread(&m, sizeof(int), 1, file) != 1 ||
        fread(&n, sizeof(int), 1, file) != 1 ||
        fread(&dk, sizeof(int), 1, file) != 1 ||
        fread(&dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    int offset = sizeof(int) * 4 + sizeof(double) * (m * dk + n * dk + n * dv);
    fseek(file, offset, SEEK_SET);

    bool res = true;
    double threshold = 0.02;
    double* row = (double*) malloc(sizeof(double) * dv);

    for (int i = 0; i < m; i++) {
        int base = i * dv;
        fread(row, sizeof(double), dv, file);
        for (int j = 0; j < dv; j++) {
            if (isnan(result[base + 1]) || fabs(result[base + j] - row[j]) > threshold) {
                printf("Expect result[%d][%d] to be %lf, but it is %lf\n", i, j, row[j], result[base + j]);
                res = false;
                goto end;
            }
        }
    }

end:
    free(row);
    fclose(file);
    return res;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m, n, dk, dv;

    read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
    result = malloc(sizeof(double) * m * dv);

    struct timespec beg, end;
    clock_gettime(CLOCK_MONOTONIC, &beg);
    attention(Q, K, V, result, m, n, dk, dv);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (verify(argv[1], result)) {
        double elapsed_time = (end.tv_sec - beg.tv_sec) * 1e6 + (end.tv_nsec - beg.tv_nsec) / 1e3;
        printf("Correct!\nElapsed time: %.2lf us\n", elapsed_time);
    } else {
        puts("Wrong!");
    }

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}
