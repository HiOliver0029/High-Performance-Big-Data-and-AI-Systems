#include <math.h>
#include <mpi.h>
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
               int m, int n, int dk, int dv,
               int mpi_rank, int mpi_size) {
    // Optimized Open MPI attention implementation with load balancing
    // and non-blocking communication
    
    // Step 1: Broadcast dimensions to all processes
    int dims[4] = {m, n, dk, dv};
    MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        m = dims[0]; n = dims[1]; dk = dims[2]; dv = dims[3];
    }
    
    // Step 2: Calculate load distribution (balanced across processes)
    int rows_per_proc = m / mpi_size;
    int remainder = m % mpi_size;
    int local_m = rows_per_proc + (mpi_rank < remainder ? 1 : 0);
    int start_row = mpi_rank * rows_per_proc + (mpi_rank < remainder ? mpi_rank : remainder);

    // Step 3: Allocate local buffers
    double* local_Q = malloc(sizeof(double) * local_m * dk);
    double* local_result = malloc(sizeof(double) * local_m * dv);

    // Step 4: Broadcast K and V to all processes (non-blocking)
    if (mpi_rank != 0) {
        K = malloc(sizeof(double) * n * dk);
        V = malloc(sizeof(double) * n * dv);
    }
    
    MPI_Request req[3];
    MPI_Ibcast(K, n * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Ibcast(V, n * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD, &req[1]);

    // Step 5: Scatter Q rows to processes
    int* sendcounts = NULL;
    int* displs = NULL;
    if (mpi_rank == 0) {
        sendcounts = malloc(sizeof(int) * mpi_size);
        displs = malloc(sizeof(int) * mpi_size);
        int offset = 0;
        for (int r = 0; r < mpi_size; r++) {
            int count = rows_per_proc + (r < remainder ? 1 : 0);
            sendcounts[r] = count * dk;
            displs[r] = offset * dk;
            offset += count;
        }
    }
    
    MPI_Iscatterv(Q, sendcounts, displs, MPI_DOUBLE,
                  local_Q, local_m * dk, MPI_DOUBLE,
                  0, MPI_COMM_WORLD, &req[2]);

    // Wait for all data distribution to complete
    MPI_Waitall(3, req, MPI_STATUSES_IGNORE);

    // Step 6: Local Attention Computation (optimized)
    double scale = 1.0 / sqrt((double)dk);
    double* scores = malloc(sizeof(double) * local_m * n);

    // Compute attention scores for local rows
    for (int i = 0; i < local_m; i++) {
        double* q_row = &local_Q[i * dk];
        double* score_row = &scores[i * n];
        
        // Step 6a: QK^T / sqrt(dk) with loop unrolling
        for (int j = 0; j < n; j++) {
            double* k_row = &K[j * dk];
            double sum = 0.0;
            
            int k;
            for (k = 0; k < dk - 3; k += 4) {
                sum += q_row[k] * k_row[k];
                sum += q_row[k+1] * k_row[k+1];
                sum += q_row[k+2] * k_row[k+2];
                sum += q_row[k+3] * k_row[k+3];
            }
            for (; k < dk; k++) {
                sum += q_row[k] * k_row[k];
            }
            
            score_row[j] = sum * scale;
        }

        // Step 6b: Numerically stable softmax
        double max_val = score_row[0];
        for (int j = 1; j < n; j++) {
            if (score_row[j] > max_val) 
                max_val = score_row[j];
        }

        double sum_exp = 0.0;
        for (int j = 0; j < n; j++) {
            double exp_val = exp(score_row[j] - max_val);
            score_row[j] = exp_val;
            sum_exp += exp_val;
        }
        
        double inv_sum = 1.0 / sum_exp;
        for (int j = 0; j < n; j++) {
            score_row[j] *= inv_sum;
        }

        // Step 6c: Multiply by V (cache-friendly)
        double* result_row = &local_result[i * dv];
        for (int d = 0; d < dv; d++) {
            result_row[d] = 0.0;
        }
        
        for (int j = 0; j < n; j++) {
            double score_val = score_row[j];
            double* v_row = &V[j * dv];
            
            for (int d = 0; d < dv; d++) {
                result_row[d] += score_val * v_row[d];
            }
        }
    }

    // Step 7: Gather results to rank 0
    if (mpi_rank == 0) {
        for (int r = 0; r < mpi_size; r++) {
            int count = rows_per_proc + (r < remainder ? 1 : 0);
            sendcounts[r] = count * dv;
            displs[r] = (r * rows_per_proc + (r < remainder ? r : remainder)) * dv;
        }
    }
    
    MPI_Gatherv(local_result, local_m * dv, MPI_DOUBLE,
                result, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Step 8: Cleanup
    free(local_Q);
    free(local_result);
    free(scores);
    
    if (mpi_rank != 0) {
        free(K);
        free(V);
    }
    
    if (mpi_rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 127 <template code>) <(tail -n 127 <your code>)`

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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m, n, dk, dv;

    if (rank == 0) {
        read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
        result = malloc(sizeof(double) * m * dv);
    }

    double beg, duration, duration_max;
    beg = MPI_Wtime();
    attention(Q, K, V, result, m, n, dk, dv, rank, size);
    duration = MPI_Wtime() - beg;

    MPI_Reduce(&duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (verify(argv[1], result)) {
            printf("Correct!\nElapsed time: %.2lf us\n", duration_max * 1e6);
        } else {
            puts("Wrong!");
        }
    }

    MPI_Finalize();

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}
