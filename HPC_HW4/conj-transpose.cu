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

// WARN: you are only allowed to modify this function
__global__ void transpose(Complex M_d_in[], Complex M_d_out[], int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        Complex val = M_d_in[y * N + x];
        val.imag = -val.imag;
        M_d_out[x * N + y] = val;
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

bool verify(Complex M_in[], Complex M_out[], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (M_out[i * N + j].real != M_in[j * N + i].real ||
                M_out[i * N + j].imag != -M_in[j * N + i].imag) {
                cout << "Expected M_out[" << i << "][" << j << "] to be "
                     << M_in[j * N + i].real << showpos << -M_in[j * N + i].imag << "i, but it is "
                     << M_out[i * N + j].real << showpos << M_out[i * N + j].imag << 'i' << endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int N = 1024;
    int times = 100;

    if (argc >= 3) times = stoi(argv[2]);
    if (argc >= 2) N = stoi(argv[1]);

    size_t size = N * N * sizeof(Complex);
    Complex* M_in = new Complex[N * N];
    Complex* M_out = new Complex[N * N];
    Complex* M_d_in;
    Complex* M_d_out;
    cudaMalloc(&M_d_in, size);
    cudaMalloc(&M_d_out, size);

    initMatrix(M_in, N);
    cudaMemcpy(M_d_in, M_in, size, cudaMemcpyHostToDevice);

    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    float duration;
    float total_duration = 0;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    for (int i = 0; i < times; i++) {
        // clear matrix
        cudaMemset(M_d_out, size, 0);

        cudaEventRecord(beg);
        transpose<<<grid_dim, block_dim>>>(M_d_in, M_d_out, N);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        cudaEventElapsedTime(&duration, beg, end);
        total_duration += duration;
    }
    cudaMemcpy(M_out, M_d_out, size, cudaMemcpyDeviceToHost);

    if (verify(M_in, M_out, N)) {
        cout << "Correct!" << endl;
        cout << "Average conjugate transpose time: " << (total_duration * 1000 / times) << " us" << endl;
    } else {
        cout << "Wrong!" << endl;
    }

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(M_d_in);
    cudaFree(M_d_out);
    delete[] M_in;
    delete[] M_out;
    return 0;
}
