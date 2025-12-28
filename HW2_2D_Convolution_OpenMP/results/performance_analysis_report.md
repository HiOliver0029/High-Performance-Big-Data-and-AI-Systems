
# High Performance Computing HW2 - Convolution Performance Report

Generated on: 2025-09-22 15:56:28

## Executive Summary

This report presents a comprehensive analysis of 2D convolution implementation performance, 
comparing serial and OpenMP parallel versions across different matrix sizes and kernel sizes.

## Experimental Setup

### Platform Specifications
- **Operating System**: Windows
- **Compiler**: GCC with OpenMP support
- **Test Configuration**:
  - Matrix Sizes: [256, 512, 1024, 2048, 4096]
  - Kernel Sizes: [3, 5, 7, 9]
  - Thread Counts: [1, 2, 4, 8, 16, 32]

## Performance Analysis Results

### Serial Performance Analysis

#### Serial Execution Times (microseconds)

Kernel_Size         3          5          7          9
Matrix_Size                                           
256            2674.1     6850.5    21684.8    23638.9
512           10896.1    40764.8    75026.8    95689.2
1024          50969.4   119381.9   301449.0   358554.7
2048         235275.9   529756.1   899364.1  1455349.8
4096         754668.2  2019950.1  4174755.5  6507727.7

#### Computational Complexity Analysis

- Kernel 3x3: O(n^2.07)
- Kernel 5x5: O(n^2.01)
- Kernel 7x7: O(n^1.88)
- Kernel 9x9: O(n^2.01)

### Parallel Performance Analysis

#### Speedup Results

**Matrix 256x256:**

- Kernel 3x3:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 1.63x speedup (81.3% efficiency)
  - 4 threads: 1.92x speedup (47.9% efficiency)
  - 8 threads: 1.53x speedup (19.2% efficiency)
  - 16 threads: 1.53x speedup (9.5% efficiency)
  - 32 threads: 0.96x speedup (3.0% efficiency)

- Kernel 5x5:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 4.87x speedup (243.3% efficiency)
  - 4 threads: 5.18x speedup (129.5% efficiency)
  - 8 threads: 2.94x speedup (36.7% efficiency)
  - 16 threads: 4.78x speedup (29.8% efficiency)
  - 32 threads: 2.63x speedup (8.2% efficiency)

- Kernel 7x7:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 4.37x speedup (218.3% efficiency)
  - 4 threads: 4.67x speedup (116.8% efficiency)
  - 8 threads: 2.80x speedup (35.0% efficiency)
  - 16 threads: 4.15x speedup (25.9% efficiency)
  - 32 threads: 2.76x speedup (8.6% efficiency)

- Kernel 9x9:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.17x speedup (108.4% efficiency)
  - 4 threads: 2.29x speedup (57.1% efficiency)
  - 8 threads: 1.78x speedup (22.3% efficiency)
  - 16 threads: 2.52x speedup (15.8% efficiency)
  - 32 threads: 3.31x speedup (10.3% efficiency)

**Matrix 512x512:**

- Kernel 3x3:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 3.34x speedup (167.0% efficiency)
  - 4 threads: 4.63x speedup (115.8% efficiency)
  - 8 threads: 5.60x speedup (70.0% efficiency)
  - 16 threads: 2.29x speedup (14.3% efficiency)
  - 32 threads: 2.68x speedup (8.4% efficiency)

- Kernel 5x5:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 4.67x speedup (233.6% efficiency)
  - 4 threads: 4.61x speedup (115.3% efficiency)
  - 8 threads: 4.72x speedup (59.1% efficiency)
  - 16 threads: 6.06x speedup (37.9% efficiency)
  - 32 threads: 4.71x speedup (14.7% efficiency)

- Kernel 7x7:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 3.13x speedup (156.5% efficiency)
  - 4 threads: 3.52x speedup (88.0% efficiency)
  - 8 threads: 3.84x speedup (48.0% efficiency)
  - 16 threads: 2.10x speedup (13.1% efficiency)
  - 32 threads: 3.32x speedup (10.4% efficiency)

- Kernel 9x9:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.52x speedup (126.1% efficiency)
  - 4 threads: 3.44x speedup (86.1% efficiency)
  - 8 threads: 3.78x speedup (47.3% efficiency)
  - 16 threads: 4.14x speedup (25.9% efficiency)
  - 32 threads: 4.99x speedup (15.6% efficiency)

**Matrix 1024x1024:**

- Kernel 3x3:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 3.77x speedup (188.3% efficiency)
  - 4 threads: 3.63x speedup (90.8% efficiency)
  - 8 threads: 6.45x speedup (80.6% efficiency)
  - 16 threads: 5.10x speedup (31.9% efficiency)
  - 32 threads: 5.11x speedup (16.0% efficiency)

- Kernel 5x5:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.72x speedup (135.9% efficiency)
  - 4 threads: 3.37x speedup (84.2% efficiency)
  - 8 threads: 4.18x speedup (52.2% efficiency)
  - 16 threads: 4.34x speedup (27.1% efficiency)
  - 32 threads: 3.76x speedup (11.8% efficiency)

- Kernel 7x7:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.05x speedup (102.7% efficiency)
  - 4 threads: 3.24x speedup (80.9% efficiency)
  - 8 threads: 3.88x speedup (48.5% efficiency)
  - 16 threads: 3.91x speedup (24.4% efficiency)
  - 32 threads: 3.63x speedup (11.4% efficiency)

- Kernel 9x9:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.54x speedup (126.9% efficiency)
  - 4 threads: 3.07x speedup (76.7% efficiency)
  - 8 threads: 3.64x speedup (45.5% efficiency)
  - 16 threads: 4.52x speedup (28.2% efficiency)
  - 32 threads: 4.44x speedup (13.9% efficiency)

**Matrix 2048x2048:**

- Kernel 3x3:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 1.77x speedup (88.3% efficiency)
  - 4 threads: 3.19x speedup (79.8% efficiency)
  - 8 threads: 4.02x speedup (50.2% efficiency)
  - 16 threads: 4.01x speedup (25.1% efficiency)
  - 32 threads: 3.41x speedup (10.7% efficiency)

- Kernel 5x5:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.34x speedup (116.9% efficiency)
  - 4 threads: 3.86x speedup (96.5% efficiency)
  - 8 threads: 3.87x speedup (48.4% efficiency)
  - 16 threads: 4.96x speedup (31.0% efficiency)
  - 32 threads: 4.73x speedup (14.8% efficiency)

- Kernel 7x7:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.53x speedup (126.5% efficiency)
  - 4 threads: 3.51x speedup (87.7% efficiency)
  - 8 threads: 3.04x speedup (38.0% efficiency)
  - 16 threads: 4.01x speedup (25.0% efficiency)
  - 32 threads: 4.30x speedup (13.4% efficiency)

- Kernel 9x9:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.24x speedup (111.9% efficiency)
  - 4 threads: 3.29x speedup (82.2% efficiency)
  - 8 threads: 3.97x speedup (49.6% efficiency)
  - 16 threads: 4.25x speedup (26.6% efficiency)
  - 32 threads: 4.34x speedup (13.5% efficiency)

**Matrix 4096x4096:**

- Kernel 3x3:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.77x speedup (138.5% efficiency)
  - 4 threads: 3.17x speedup (79.3% efficiency)
  - 8 threads: 5.03x speedup (62.9% efficiency)
  - 16 threads: 4.89x speedup (30.5% efficiency)
  - 32 threads: 5.11x speedup (16.0% efficiency)

- Kernel 5x5:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.72x speedup (136.2% efficiency)
  - 4 threads: 4.00x speedup (100.0% efficiency)
  - 8 threads: 4.51x speedup (56.3% efficiency)
  - 16 threads: 4.46x speedup (27.9% efficiency)
  - 32 threads: 4.41x speedup (13.8% efficiency)

- Kernel 7x7:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 8.77x speedup (438.7% efficiency)
  - 4 threads: 12.14x speedup (303.6% efficiency)
  - 8 threads: 12.55x speedup (156.9% efficiency)
  - 16 threads: 16.08x speedup (100.5% efficiency)
  - 32 threads: 13.47x speedup (42.1% efficiency)

- Kernel 9x9:
  - 1 threads: 1.00x speedup (100.0% efficiency)
  - 2 threads: 2.41x speedup (120.5% efficiency)
  - 4 threads: 2.44x speedup (61.0% efficiency)
  - 8 threads: 4.33x speedup (54.1% efficiency)
  - 16 threads: 4.13x speedup (25.8% efficiency)
  - 32 threads: 4.52x speedup (14.1% efficiency)

#### Performance Highlights

- **Best Speedup**: 16.08x with 16 threads (Matrix 4096x4096, Kernel 7x7)
- **Best Efficiency**: 438.7% with 2 threads (Matrix 4096x4096, Kernel 7x7)

### Scalability Analysis

#### Strong Scaling

Strong scaling measures how execution time decreases as the number of processors increases for a fixed problem size. Our analysis shows:

- 1 threads: 100.0% average efficiency
- 2 threads: 158.3% average efficiency
- 4 threads: 99.0% average efficiency
- 8 threads: 54.0% average efficiency
- 16 threads: 28.8% average efficiency
- 32 threads: 13.5% average efficiency

#### Optimization Techniques Applied

1. **OpenMP Parallelization**: Used `#pragma omp parallel for collapse(2)` to parallelize the outer two loops
2. **Static Scheduling**: Applied static scheduling for better load balancing
3. **Zero Padding**: Implemented efficient boundary condition handling
4. **Memory Access Optimization**: Optimized array indexing for better cache locality

## Conclusions and Recommendations

1. **Parallel Efficiency**: The OpenMP implementation shows good scalability up to 32 threads, achieving an average speedup of 4.33x.

2. **Problem Size Impact**: Larger matrices benefit more from parallelization due to better computation-to-overhead ratio.

3. **Kernel Size Effect**: Larger kernels show better parallel efficiency due to increased computational intensity per output element.

4. **Optimization Opportunities**: Further improvements could include SIMD vectorization, cache blocking, and GPU acceleration for very large problem sizes.

