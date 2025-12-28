# Homework 4 Report: CUDA Conjugate Transpose

**Student ID**: R14922092  
**Name**: 林席葦  
**Department**: CSIE, Master 1st   
**Date**: December 11, 2025  

---

## 1. Introduction
The objective of this assignment is to implement the conjugate transpose of an $N \times N$ complex matrix using CUDA. The conjugate transpose $A^*$ involves transposing the matrix and taking the complex conjugate of each element. We explore three implementations:
1.  **Naive**: Using only global memory.
2.  **Shared Memory**: Using shared memory tiling to ensure coalesced global memory accesses.
3.  **Bank Conflict Avoidance**: Optimizing the shared memory implementation to minimize bank conflicts.

## 2. Experiment Environment

-   **OS**: Linux (Kernel 5.15.0-122-generic)
-   **CPU**: Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz
-   **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
-   **CUDA Version**: 12.4 (Driver), 11.8 (Runtime)
-   **Compiler**: nvcc 11.8

## 3. Implementation Details

### 3.1 Naive Implementation (`conj-transpose.cu`)
This implementation reads directly from global memory and writes directly to global memory.
-   **Read**: `in[y * N + x]` (Coalesced if `x` is the fast dimension).
-   **Write**: `out[x * N + y]` (Uncoalesced).
The write operation has a stride of $N$, which breaks memory coalescing and results in poor bandwidth utilization.

### 3.2 Shared Memory Implementation (`conj-transpose-shmem.cu`)
To fix the uncoalesced write, we use shared memory as a buffer.
-   **Step 1**: Threads read a tile from global memory into `__shared__ Complex tile[TILE_DIM][TILE_DIM]`. This read is coalesced.
-   **Step 2**: Threads synchronize (`__syncthreads()`).
-   **Step 3**: Threads write from shared memory to global memory using transposed indices. By calculating `new_x` and `new_y` based on the transposed block coordinates, we ensure the write to `out[new_y * N + new_x]` is also coalesced.
-   **Issue**: Accessing `tile[threadIdx.x][threadIdx.y]` in the write phase causes bank conflicts because `TILE_DIM` (32) is a multiple of the number of memory banks (32). All threads in a warp access the same bank simultaneously.

### 3.3 Bank Conflict Avoidance (`conj-transpose-shmem-bc-avoid.cu`)
To resolve bank conflicts, we add padding to the shared memory array.
-   **Declaration**: `__shared__ Complex tile[TILE_DIM][TILE_DIM + 1];`
-   **Effect**: The padding shifts the elements such that elements in the same column (which are accessed simultaneously by a warp during the transpose) fall into different memory banks. This serializes the access only if necessary, but in this case, it completely parallelizes the access.

## 4. Performance Analysis

### 4.1 Execution Time Results

We measured the average execution time (in microseconds) over 100 iterations for varying matrix sizes $N$.

| Matrix Size ($N$) | Naive (us) | Shared Memory (us) | Shared Memory + BC Avoid (us) | Speedup (BC vs Naive) |
| :--- | :--- | :--- | :--- | :--- |
| **1024** | 65.26 | 47.02 | 44.83 | **1.46x** |
| **2048** | 247.33 | 174.06 | 166.89 | **1.48x** |
| **4096** | 973.21 | 681.31 | 655.87 | **1.48x** |

### 4.2 Observations
1.  **Coalescing Impact**: Across all sizes, the Shared Memory implementation significantly outperforms the Naive one. This confirms that global memory bandwidth is the primary bottleneck. The uncoalesced writes in the Naive version waste a significant portion of the memory bandwidth.
2.  **Bank Conflict Impact**: The "BC Avoid" version consistently outperforms the standard Shared Memory version. The improvement is stable across different matrix sizes, demonstrating that eliminating shared memory serialization provides a consistent latency reduction.
3.  **Scalability**: The speedup remains consistent (~1.48x) as $N$ increases, indicating that the optimization is robust for large datasets.

### 4.3 Profiling Analysis (Nsight Compute)

We analyzed the memory behavior using Nsight Compute metrics for $N=4096$.

#### Memory Coalescing Metrics
-   `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` (Load Ratio)
-   `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio` (Store Ratio)
-   `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` (Total Store Sectors)

| Implementation | Load Ratio | Store Ratio | Total Store Sectors | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Naive** | 8 | **32** | **16,777,216** | The high store ratio (32) and 4x more total store sectors confirm uncoalesced writes. Each request transfers 32 sectors, wasting bandwidth. |
| **Shared Memory** | 8 | **8** | **4,194,304** | Both load and store ratios are 8, and total store sectors dropped by 4x. This indicates perfect coalescing. |
| **BC Avoid** | 8 | **8** | **4,194,304** | Maintains the same efficient global memory access pattern. |

#### Bank Conflict Metrics
-   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (Shared Load Conflicts)

| Implementation | Shared Load Conflicts | Analysis |
| :--- | :--- | :--- |
| **Naive** | 0 | Does not use shared memory. |
| **Shared Memory** | **15,806,242** | High number of conflicts due to stride-32 access pattern in shared memory. |
| **BC Avoid** | **73,101** | Conflicts are drastically reduced (by >99.5%). The remaining small number might be due to non-ideal alignment or other overheads, but the primary stride conflict is eliminated. |

## 5. Question Answering

### Q1: What is "warp" in GPU? How a 2-D thread block divided into warps?
A **warp** is a group of 32 consecutive threads in a CUDA block that are executed together in a SIMT (Single Instruction, Multiple Thread) fashion. They share the same program counter and execute the same instruction at the same time.

For a 2-D thread block of size $(D_x, D_y)$, threads are indexed linearly as $tid = threadIdx.y \times D_x + threadIdx.x$.
-   The dimension `threadIdx.x` increases faster.
-   Warps are formed by grouping these linear IDs. For example, threads with linear IDs 0-31 form the first warp, 32-63 the second, and so on.
-   Therefore, rows of the block (along x) are packed into warps first.

### Q2: What is the difference between regular CUDA memory (allocated with `cudaMalloc`) and CUDA shared memory (allocated with `__shared__` qualifier)?
-   **Regular CUDA Memory (Global Memory)**:
    -   **Location**: Off-chip DRAM (VRAM).
    -   **Scope**: Accessible by all threads in all blocks and the host (CPU).
    -   **Lifetime**: Persists until explicitly freed (`cudaFree`) or the application ends.
    -   **Speed**: High latency (hundreds of cycles), high bandwidth but lower than on-chip memory.
-   **Shared Memory**:
    -   **Location**: On-chip (L1 cache/Shared Memory configurable split).
    -   **Scope**: Accessible only by threads within the same thread block.
    -   **Lifetime**: Exists only for the duration of the thread block execution.
    -   **Speed**: Very low latency (similar to registers), extremely high bandwidth.

### Q3: What is memory coalescing and when does it happen? Did you apply any technique to ensure memory coalescing?
**Memory Coalescing** is the technique where the GPU memory controller combines multiple memory accesses from threads in a warp into a single transaction. It happens when threads in a warp access a contiguous block of aligned memory.

**Technique Applied**:
Yes, in the `conj-transpose-shmem.cu` and `conj-transpose-shmem-bc-avoid.cu` implementations:
1.  **Read**: Threads read `M_d_in[y * N + x]`. Since `x` corresponds to `threadIdx.x` (consecutive threads), the addresses are contiguous.
2.  **Write**: Instead of writing directly to `M_d_out[x * N + y]` (strided), we used shared memory to transpose the data tile. We then calculated new indices `new_x` and `new_y` such that `new_x` varies with `threadIdx.x`. The write `M_d_out[new_y * N + new_x]` then becomes contiguous, ensuring coalescing.

### Q4: What is bank conflict and when does it happen? Did you avoid bank conflict in your code?
**Bank Conflict** happens in shared memory when multiple threads in the same warp access different addresses that map to the same memory bank. Shared memory is divided into 32 banks (4-byte wide). If $k$ threads access the same bank, the access is serialized, taking $k$ times longer.

**Technique Applied**:
Yes, in `conj-transpose-shmem-bc-avoid.cu`:
-   **Problem**: In the shared memory implementation, we declared `tile[32][32]`. When reading a column `tile[threadIdx.x][threadIdx.y]` (where `threadIdx.x` varies), threads 0 to 31 access `tile[0][y]`, `tile[1][y]`, etc. Since the row size is 32, `tile[0][y]` and `tile[1][y]` map to the same bank (Bank 0 and Bank 0) if the elements are 32-bit words. For 64-bit `Complex`, it's slightly different but conflicts still occur with stride 32.
-   **Solution**: We declared `__shared__ Complex tile[32][33]`. The extra column (padding) changes the stride. Now, `tile[0][y]` and `tile[1][y]` are separated by 33 elements, mapping them to different banks. This eliminates the conflict.
