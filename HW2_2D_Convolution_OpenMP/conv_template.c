#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
 * Assignment requirements:
 * You must write your own separate file (Name: conv.c),
 * and implement the following function inside it:
 *    void conv(int* M, int w, int* K, int k, int* C);
 * You are NOT allowed to modify any code in this template file.
 *
 * M: a w by w input matrix
 * K: a k by k input matrix
 * C: a w by w output matrix containing the result of
 *    padded convolution of M and K
 */
void conv(int* M, int w, int* K, int k, int* C);

int* readMatrix(const char* file_path, int* w) {
    FILE* mat_file = fopen(file_path, "r");

    fscanf(mat_file, "%d", w);
    int* M = malloc(sizeof(int) * (*w) * (*w));

    int idx = 0;
    while (fscanf(mat_file, "%d", &M[idx++]) != EOF);

    fclose(mat_file);
    return M;
}

bool verify(const char* ans_path, int* C, int w) {
    FILE* ans_file = fopen(ans_path, "r");
    if (!ans_file) {
        fprintf(stderr, "Invalid answer file\n");
        exit(1);
    }

    int ans_w;
    fscanf(ans_file, "%d", &ans_w);
    if (ans_w != w) return false;

    int idx = 0;
    int num;
    while (fscanf(ans_file, "%d", &num) != EOF) {
        if (C[idx++] != num) return false;
    }

    fclose(ans_file);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s MATRIX KERNEL ANSWER\n", argv[0]);
        return 1;
    } 
    int w, k;
    int* M = readMatrix(argv[1], &w);
    int* K = readMatrix(argv[2], &k);
    int* C = malloc(sizeof(int) * w * w);

    struct timespec beg, end;
    clock_gettime(CLOCK_MONOTONIC, &beg);
    conv(M, w, K, k, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

   

    if (verify(argv[3], C, w)) {
        double elapsed_time = (end.tv_sec - beg.tv_sec) * 1e6 + (end.tv_nsec - beg.tv_nsec) / 1e3;
        printf("Correct!\nElapsed time: %lf us\n", elapsed_time);
    } else {
        puts("Wrong!");
    }

    free(M);
    free(K);
    free(C);
    return 0;
}
