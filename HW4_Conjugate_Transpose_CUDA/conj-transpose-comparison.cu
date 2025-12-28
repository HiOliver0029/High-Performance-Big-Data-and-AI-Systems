#include <ctime>
#include <iostream>
#include <random>
#include <string>

#define TILE_DIM 32

using namespace std;

struct __align__(8) Complex {
    float real;
    float imag;
};

__global__ void transpose_naive(Complex M_d_in[], Complex M_d_out[], int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        Complex val = M_d_in[y * N + x];
        val.imag = -val.imag;
        M_d_out[x * N + y] = val; // Uncoalesced write
    }
}

__global__ void transpose_optimized(Complex M_d_in[], Complex M_d_out[], int N) {
    __shared__ Complex tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = M_d_in[y * N + x];
    }

    __syncthreads();

    int new_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int new_y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (new_x < N && new_y < N) {
        Complex val = tile[threadIdx.x][threadIdx.y];
        val.imag = -val.imag;
        M_d_out[new_y * N + new_x] = val;
    }
}

void initMatrix(Complex M[], int N) {
    mt19937 gen;
    gen.seed(time(nullptr));
    uniform_real_distribution<float> dis(-1e6, 1e6);

    int count = N * N;
    for (int i = 0; i < count; i++) {
        M[i].real = dis(gen);
        M[i].imag = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    int N = 4096; // Larger size for better comparison
    int times = 100;

    if (argc >= 3) times = stoi(argv[2]);
    if (argc >= 2) N = stoi(argv[1]);

    size_t size = N * N * sizeof(Complex);
    Complex* M_in = new Complex[N * N];
    Complex* M_d_in;
    Complex* M_d_out;
    cudaMalloc(&M_d_in, size);
    cudaMalloc(&M_d_out, size);

    initMatrix(M_in, N);
    cudaMemcpy(M_d_in, M_in, size, cudaMemcpyHostToDevice);

    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    float duration;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Warmup
    transpose_naive<<<grid_dim, block_dim>>>(M_d_in, M_d_out, N);
    transpose_optimized<<<grid_dim, block_dim>>>(M_d_in, M_d_out, N);
    cudaDeviceSynchronize();

    // Measure Naive
    float total_duration_naive = 0;
    for (int i = 0; i < times; i++) {
        cudaEventRecord(beg);
        transpose_naive<<<grid_dim, block_dim>>>(M_d_in, M_d_out, N);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&duration, beg, end);
        total_duration_naive += duration;
    }

    // Measure Optimized
    float total_duration_opt = 0;
    for (int i = 0; i < times; i++) {
        cudaEventRecord(beg);
        transpose_optimized<<<grid_dim, block_dim>>>(M_d_in, M_d_out, N);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&duration, beg, end);
        total_duration_opt += duration;
    }

    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "Naive Average Time: " << (total_duration_naive / times) << " ms" << endl;
    cout << "Optimized Average Time: " << (total_duration_opt / times) << " ms" << endl;
    cout << "Speedup: " << (total_duration_naive / total_duration_opt) << "x" << endl;

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(M_d_in);
    cudaFree(M_d_out);
    delete[] M_in;
    return 0;
}
