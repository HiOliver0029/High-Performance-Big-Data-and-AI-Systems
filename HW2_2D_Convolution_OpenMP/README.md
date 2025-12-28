# High Performance Computing HW2 - 2D Convolution

[![Language](https://img.shields.io/badge/Language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](https://www.openmp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)](https://github.com)

A high-performance 2D convolution implementation with both serial and OpenMP parallel versions, featuring comprehensive performance analysis and scalability evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

This project implements efficient 2D convolution algorithms for high-performance computing applications. It includes:

- **Serial Implementation**: Optimized single-threaded 2D convolution
- **Parallel Implementation**: OpenMP-accelerated multi-threaded version
- **Performance Analysis**: Comprehensive benchmarking and scalability evaluation
- **Automated Testing**: Batch scripts for systematic performance measurement

### Key Features

- ✅ **Zero Padding**: Proper boundary condition handling
- ✅ **OpenMP Parallelization**: Multi-threaded acceleration
- ✅ **Comprehensive Testing**: Multiple matrix and kernel sizes
- ✅ **Performance Visualization**: Automated chart generation
- ✅ **Scalability Analysis**: Strong scaling evaluation
- ✅ **Cross-Platform**: Windows and Linux support

## 🔧 Requirements

### Software Dependencies

- **Compiler**: GCC with OpenMP support
- **Python**: 3.7+ (for analysis scripts)
- **Libraries**: 
  - pandas
  - matplotlib
  - numpy
  - seaborn

### Hardware Recommendations

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB+ for large matrix operations
- **Storage**: 1GB free space for results

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HiOliver0029/HPC_HW2_Convolution.git
cd HPC_HW2_Convolution
```

### 2. Install Python Dependencies

```bash
pip install pandas matplotlib numpy seaborn
```

### 3. Verify Compiler

```bash
gcc --version
gcc -fopenmp --version
```

## 📖 Usage

### Quick Start

1. **Compile Programs**:
   ```bash
   gcc conv.c conv_template.c -o conv
   gcc -fopenmp conv_openmp.c conv_openmp_template.c -o conv_openmp
   ```

2. **Run Single Test**:
   ```bash
   ./conv testing_data/mat-256.txt testing_data/ker-3.txt testing_data/ans-256-3.txt
   ```

3. **Run OpenMP Test**:
   ```bash
   export OMP_NUM_THREADS=4
   ./conv_openmp testing_data/mat-512.txt testing_data/ker-5.txt testing_data/ans-512-5.txt
   ```

### Automated Testing

#### Windows
```cmd
run_tests_report_fixed.bat
```

#### Linux/MacOS
```bash
chmod +x run_tests.sh
./run_tests.sh
```

### Performance Analysis

1. **Run Analysis Script**:
   ```bash
   cd results
   python analyze_results_enhanced.py
   ```

2. **View Generated Reports**:
   - `performance_analysis_report.md` - Detailed analysis
   - `*.png` - Performance charts
   - `*.csv` - Raw performance data

## 📊 Performance Analysis

### Test Configurations

| Parameter | Values |
|-----------|--------|
| **Matrix Sizes** | 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096 |
| **Kernel Sizes** | 3×3, 5×5, 7×7, 9×9 |
| **Thread Counts** | 1, 2, 4, 8, 16, 32 |

### Analysis Features

- **Speedup Calculation**: Performance improvement with multiple threads
- **Efficiency Metrics**: Resource utilization analysis
- **Scalability Evaluation**: Strong scaling characteristics
- **Complexity Analysis**: Computational complexity measurement

### Sample Results

```
Matrix 1024×1024, Kernel 3×3:
  1 thread:  1.00x speedup (100.0% efficiency)
  2 threads: 1.94x speedup (97.0% efficiency)
  4 threads: 3.78x speedup (94.5% efficiency)
  8 threads: 6.85x speedup (85.6% efficiency)
```

## 📁 Project Structure

```
HPC_HW2_Convolution/
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
├── conv.c                        # Serial implementation
├── conv_openmp.c                 # OpenMP parallel implementation
├── conv_template.c               # Test framework (serial)
├── conv_openmp_template.c        # Test framework (parallel)
├── run_tests_report_fixed.bat    # Windows test script
├── run_tests.sh                  # Linux test script
├── fix_and_analyze.py           # Data recovery script
├── testing_data/                # Test matrices and kernels
│   ├── mat-256.txt              # Input matrices
│   ├── ker-3.txt                # Convolution kernels
│   └── ans-256-3.txt            # Expected results
└── results/                     # Generated results
    ├── analyze_results_enhanced.py
    ├── performance_analysis_report.md
    ├── *.csv                    # Performance data
    └── *.png                    # Performance charts
```

## 🔬 Implementation Details

### Serial Version (`conv.c`)

- **Algorithm**: Direct 2D convolution with zero padding
- **Optimization**: Efficient boundary checking and array indexing
- **Complexity**: O(w² × k²) where w is matrix width, k is kernel size

### Parallel Version (`conv_openmp.c`)

- **Parallelization**: OpenMP `parallel for` with `collapse(2)`
- **Scheduling**: Static scheduling for load balancing
- **Thread Safety**: No shared variables requiring synchronization

### Key Optimizations

1. **Memory Access Pattern**: Sequential access for cache efficiency
2. **Loop Collapse**: Better work distribution across threads
3. **Static Scheduling**: Predictable workload distribution
4. **Boundary Optimization**: Efficient zero-padding implementation

## 📈 Results Summary

### Performance Highlights

- **Maximum Speedup**: Up to 7.2x with 8 threads
- **Best Efficiency**: 97% with 2 threads
- **Scalability**: Good strong scaling up to 8 threads
- **Correctness**: 100% test pass rate for all configurations

### Platform-Specific Results

Results may vary based on:
- CPU architecture and core count
- Memory bandwidth and cache size
- Operating system and compiler optimizations
- System load and background processes
