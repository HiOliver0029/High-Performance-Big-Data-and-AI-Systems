# High Performance Big Data and AI Systems

This repository contains the coursework and assignments for the "High Performance Big Data and AI Systems" course. It explores various high-performance computing (HPC) techniques including OpenMP, MPI, and CUDA to optimize algorithms commonly used in big data and AI applications.

## Projects

### [HW2: 2D Convolution (OpenMP)](HW2_2D_Convolution_OpenMP)
**Objective:** Implement and optimize 2D convolution using OpenMP for shared-memory parallelization.

*   **Description:** This project compares serial and OpenMP parallel versions of 2D convolution across various matrix and kernel sizes.
*   **Key Features:**
    *   Parallel implementation using OpenMP.
    *   Comprehensive performance analysis including strong scaling and efficiency measurements.
    *   Achieved significant speedup on multi-core systems.

### [HW3: Attention Mechanism (MPI)](HW3_Attention_Mechanism_MPI)
**Objective:** Implement the Scaled Dot-Product Attention mechanism using MPI for distributed memory systems.

*   **Description:** This project focuses on the parallelization of the attention mechanism, a core component of Transformer models. It compares a serial implementation with an MPI-based parallel version.
*   **Key Features:**
    *   Distributed computing using MPI.
    *   Optimization techniques such as loop unrolling, non-blocking communication, and cache-friendly memory access.
    *   Detailed performance analysis and scalability testing.

### [HW4: Conjugate Transpose (CUDA)](HW4_Conjugate_Transpose_CUDA)
**Objective:** Implement and optimize the conjugate transpose of a complex matrix using CUDA.

*   **Description:** This project explores GPU acceleration for matrix operations. It implements three versions of the conjugate transpose algorithm to demonstrate memory optimization techniques.
*   **Key Features:**
    *   **Naive Implementation:** Baseline using global memory.
    *   **Shared Memory:** Uses tiling to ensure coalesced global memory accesses.
    *   **Bank Conflict Avoidance:** Optimizes shared memory usage to minimize bank conflicts and maximize throughput.
    *   Performance profiling on NVIDIA GPUs (e.g., RTX 3090).

## Research Reports

### [Midterm: Deep Research Agent for Large Systems](HPC-Midterm%20Deep%20Research%20Agent%20for%20Large%20Systems.pdf)
**Topic:** Deep Research Agent for Large Systems

### [Final: Optimizing Distributed Large Model Inference](HPC_Final_Optimizing%20Distributed%20Large%20Model%20Inference.pdf)
**Topic:** Optimizing Distributed Large Model Inference
