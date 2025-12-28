#!/bin/bash

# 設定 Metrics
METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"

echo "========================================"
echo "1. Compiling CUDA codes..."
echo "========================================"
nvcc conj-transpose.cu -o conj-transpose -O3
nvcc conj-transpose-shmem.cu -o conj-transpose-shmem -O3
nvcc conj-transpose-shmem-bc-avoid.cu -o conj-transpose-shmem-bc-avoid -O3

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful."
echo ""

echo "========================================"
echo "2. Profiling Naive Implementation..."
echo "========================================"
# 只跑 1 次迭代以節省 Profiling 時間
ncu --csv --metrics $METRICS ./conj-transpose 4096 1

echo ""
echo "========================================"
echo "3. Profiling Shared Memory Implementation..."
echo "========================================"
ncu --csv --metrics $METRICS ./conj-transpose-shmem 4096 1

echo ""
echo "========================================"
echo "4. Profiling Shared Memory + BC Avoid..."
echo "========================================"
ncu --csv --metrics $METRICS ./conj-transpose-shmem-bc-avoid 4096 1

echo ""
echo "Done."
