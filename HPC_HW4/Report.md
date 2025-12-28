# Homework 4: CUDA Conjugate Transpose Report
Name: 林席葦  
Department: CSIE, Master 1st  
ID: R14922092  
Date: 30/11/2025  

## 1. Introduction
The objective of this assignment is to implement an efficient CUDA kernel for computing the conjugate transpose of a complex matrix. The conjugate transpose of a matrix $A$, denoted as $A^*$, is obtained by taking the transpose of $A$ and then taking the complex conjugate of each entry. That is, $(A^*)_{ij} = \overline{A_{ji}}$.

## 2. Methodology

### 2.1 Naive Approach
A naive implementation would directly read from the input matrix and write to the output matrix.
- **Reading**: `in[y * N + x]` (Coalesced if `x` is the fast dimension).
- **Writing**: `out[x * N + y]` (Strided/Uncoalesced).
Because the write operation accesses global memory with a stride of $N$, it results in poor memory bandwidth utilization.

### 2.2 Optimized Approach: Shared Memory Tiling
To optimize memory access, we use Shared Memory as a buffer.
1.  **Coalesced Read**: Threads in a block read a tile of the input matrix from global memory into shared memory. The access pattern `in[y * N + x]` is coalesced.
2.  **Shared Memory Exchange**: Threads synchronize, and then we transpose the coordinates within the shared memory tile.
3.  **Coalesced Write**: Threads write the data from shared memory to the output matrix. By calculating the transposed global indices `new_x` and `new_y` such that `new_x` varies with `threadIdx.x`, we ensure the write operation `out[new_y * N + new_x]` is also coalesced.

### 2.3 Bank Conflict Avoidance
When transposing data using shared memory, threads in a warp may access the same memory bank if the stride of the access matches the number of banks (usually 32).
- **Problem**: Accessing `tile[threadIdx.x][threadIdx.y]` (where `threadIdx.x` varies) causes bank conflicts if the row size is a multiple of 32.
- **Solution**: We pad the shared memory array by adding an extra column: `__shared__ Complex tile[TILE_DIM][TILE_DIM + 1];`. This shifts the elements such that column accesses fall into distinct banks, eliminating conflicts.

## 3. Implementation Details

The kernel `transpose` was implemented with the following logic:

```cpp
__global__ void transpose(Complex M_d_in[], Complex M_d_out[], int N) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ Complex tile[TILE_DIM][TILE_DIM + 1];

    // 1. Load data into shared memory (Coalesced Read)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = M_d_in[y * N + x];
    }

    __syncthreads();

    // 2. Calculate transposed indices for writing
    // We swap the block indices (blockIdx.y, blockIdx.x) to target the transposed tile
    int new_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int new_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 3. Write data to global memory (Coalesced Write)
    if (new_x < N && new_y < N) {
        // Read from shared memory with swapped indices
        Complex val = tile[threadIdx.x][threadIdx.y];
        val.imag = -val.imag; // Apply conjugate
        M_d_out[new_y * N + new_x] = val;
    }
}
```

## 4. Performance Results

### 4.1 Environment
- **GPU**: NVIDIA A100-PCIE-40GB
- **CUDA Version**: 12.2 (Driver), 11.8 (Runtime)
- **Matrix Size**: 4096 x 4096
- **Block Size**: 32 x 32

### 4.2 Comparison
We compared the performance of the Naive implementation versus the Optimized (Shared Memory Tiling + Padding) implementation.

| Implementation | Average Execution Time (ms) |
| :--- | :--- |
| **Naive** | 1.117 ms |
| **Optimized** | 0.435 ms |

**Speedup**: The optimized version is approximately **2.57x** faster than the naive version.

### 4.3 Analysis
The naive approach suffers from uncoalesced global memory writes, where adjacent threads write to memory addresses separated by a large stride ($N$). This drastically reduces the effective memory bandwidth.

The optimized approach uses shared memory to reorder the data.
1.  **Read**: Threads read from global memory into shared memory in a coalesced manner.
2.  **Transpose**: Data is transposed within the fast on-chip shared memory.
3.  **Write**: Threads write from shared memory to global memory in a coalesced manner.

Although there is overhead associated with using shared memory and synchronization (`__syncthreads()`), the gain from maximizing global memory bandwidth far outweighs the cost, resulting in the observed speedup.

## 5. Conclusion
By utilizing shared memory tiling and padding, we successfully converted the uncoalesced global memory writes into coalesced accesses and eliminated shared memory bank conflicts. This results in a significantly higher effective memory bandwidth and improved performance compared to a naive implementation.
