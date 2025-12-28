# Attention Mechanism Optimization with MPI

本專案實作了 **Scaled Dot-Product Attention** 機制的兩個版本，並在真實 HPC 叢集上進行部署與效能測試。

## 專案概述

- ✅ **串行版本** (`attention.c`) - 單執行緒基準實作
- ✅ **平行版本** (`attention-mpi.c`) - MPI 多節點平行實作
- ✅ **Docker 環境測試** - 理想條件下的演算法驗證
- ✅ **叢集部署** - 真實 HPC 環境的生產級測試
- ✅ **完整除錯指南** - 多節點通訊疑難排解

## 目錄結構

| 檔案 | 說明 |
|------|------|
| `attention.c` | 串行實作（基準） |
| `attention-mpi.c` | MPI 平行實作 |
| `test_data_generator.py` | 測試資料生成工具 |
| `REPORT_CLUSTER.md` | **🎯 叢集效能報告 & 部署指南** |
| `TROUBLESHOOTING.md` | **🔧 多節點部署疑難排解** |
| `REPORT.md` | Docker 測試結果 & 演算法驗證 |

---

## 快速開始

### 開發環境 (Docker)

```bash
# 生成測試資料
python3 test_data_generator.py

# 編譯
gcc -O3 -o attention attention.c -lm
mpicc -O3 -o attention-mpi attention-mpi.c -lm

# 測試串行版本
./attention test_xlarge.bin

# 測試 MPI 版本（單節點）
mpirun -np 8 ./attention-mpi test_xlarge.bin
```

---

## 生產叢集部署

### 1. 連線至叢集

```bash
# ProxyJump 連線
ssh -J hpcai_course_student@140.112.90.37:9037 Team10@172.16.179.50
```

### 2. 上傳檔案

```bash
# 從本機上傳
scp -J hpcai_course_student@140.112.90.37:9037 \
    attention.c attention-mpi.c test_*.bin \
    Team10@172.16.179.50:~/HP_HW3/
```

### 3. 編譯

```bash
# 登入叢集後
gcc -O3 -o attention attention.c -lm
mpicc -O3 -o attention-mpi attention-mpi.c -lm
```

### 4. 同步至所有節點

```bash
# 將執行檔與測試資料同步到其他節點
for node in rdma5 rdma6 rdma7; do
    ssh $node "mkdir -p ~/HP_HW3"
    scp attention-mpi test_*.bin $node:~/HP_HW3/
done
```

### 5. 執行測試

#### 單節點測試（推薦 m=1024）

```bash
mpirun -np 8 --mca btl ^openib ~/HP_HW3/attention-mpi ~/HP_HW3/test_xlarge.bin
```

#### 多節點測試（建議 m ≥ 4096）

**建立 Hostfile**:
```bash
cat > hosts <<EOF
172.16.179.50 slots=16
172.16.179.55 slots=16
172.16.179.56 slots=16
172.16.179.57 slots=16
EOF
```

**執行 2 節點測試**:
```bash
mpirun -H 172.16.179.50:8,172.16.179.55:8 \
    --mca btl ^openib \
    ~/HP_HW3/attention-mpi ~/HP_HW3/test_xlarge.bin
```

**⚠️ 關鍵參數**: `--mca btl ^openib` 為**必要**參數，用於禁用 InfiniBand 並強制使用 TCP/IP 通訊。

---

## 叢集環境資訊

**硬體配置**:
- **節點**: 4 台（rdma4, rdma5, rdma6, rdma7）
  - 實際主機名: inventec-0, inventec-5, inventec-6, inventec-7
- **核心數**: 每節點 16 核心（Intel Xeon）
- **網路**: Ethernet ens81np0（172.16.179.50/55/56/57）

**軟體環境**:
- **作業系統**: Ubuntu 20.04 LTS
- **MPI 版本**: Open MPI 4.1.4
- **編譯器**: GCC 9.4.0

---

## 效能總結

### Docker 環境（理想條件）

| Processes | 加速比 | 效率 |
|-----------|--------|------|
| 1 (serial) | 1.00x | 100% |
| 2 | 1.93x | 96.5% |
| 4 | 3.78x | 94.5% |
| 8 | 7.32x | 91.5% |
| 16 | 10.61x | 66.3% |
| 32 | **11.08x** | 34.6% |

**最佳表現**: 32 processes 達 11.08x 加速（理論效率 94.2%）

### HPC 叢集（真實部署）

#### 單節點擴展性（m=1024）

| Processes | 時間 (μs) | 加速比 | 效率 |
|-----------|-----------|--------|------|
| 1 (serial) | 111,252.95 | 1.00x | 100% |
| 2 | 61,687.35 | 1.80x | 90.1% |
| 4 | 40,802.02 | 2.73x | 68.2% |
| 8 | 29,490.85 | 3.77x | 47.1% |
| 16 | 21,820.35 | **5.10x** | 31.9% |

#### 多節點擴展性（2-4 節點 × 8 processes）

| 節點數 | 時間 (μs) | 加速比 | 跨節點效率 |
|--------|-----------|--------|------------|
| 1 | 29,490.85 | 3.77x | - |
| 2 | 39,104.18 | 2.85x | **17.8%** ⚠️ |
| 3 | 39,575.06 | 2.81x | **11.7%** ⚠️ |
| 4 | 40,076.07 | 2.78x | **8.7%** ⚠️ |

**關鍵發現**: 對於 m=1024，增加節點**反而降低**效能，原因是網路通訊開銷過高。

---

## 部署建議

### ✅ 適合多節點的情境

- 問題規模 **m ≥ 4096**（16 倍資料量）
- 計算/通訊比 **> 10:1**
- 需處理超大規模資料集

### ❌ 不適合多節點的情境

- 小規模問題（**m < 2048**）
- 高頻率同步需求
- 使用 TCP/IP 而非 InfiniBand

### 🎯 最佳配置（m=1024）

- **單節點**: 2-4 processes（效率 68-90%）
- **多節點**: 不推薦（負擴展）

---

## 常見問題與解決方案

### 1. 跨節點通訊超時

```bash
# ✗ 錯誤: 缺少關鍵參數
mpirun --hostfile hosts -np 16 ./attention-mpi test.bin

# ✓ 正確: 加入 --mca btl ^openib
mpirun --hostfile hosts -np 16 --mca btl ^openib ./attention-mpi test.bin
```

### 2. 主機名無法解析

```bash
# ✗ 錯誤: 使用主機名
rdma4 slots=16
inventec-5 slots=16

# ✓ 正確: 使用 IP 地址
172.16.179.50 slots=16
172.16.179.55 slots=16
```

### 3. 遠端節點找不到執行檔

```bash
# 執行前必須同步到所有節點
for node in rdma5 rdma6 rdma7; do
    scp attention-mpi test_*.bin $node:~/HP_HW3/
done
```

### 4. Python 不可用

```bash
# 在本機生成測試資料後上傳
python3 test_data_generator.py
scp test_*.bin Team10@172.16.179.50:~/HP_HW3/
```

**👉 完整診斷流程請參閱 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)**

---

## Attention 機制說明

Scaled Dot-Product Attention 計算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 計算步驟

1. **計算注意力分數**: $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
2. **應用 Softmax**: $\text{weights} = \text{softmax}(\text{scores})$
3. **加權求和**: $\text{output} = \text{weights} \times V$

---

## 相關文件

- **[`REPORT_CLUSTER.md`](REPORT_CLUSTER.md)**: 完整叢集部署指南 & 效能分析
- **[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)**: 多節點通訊除錯（hostfile、SSH、MPI 參數、檔案同步）
- **[`REPORT.md`](REPORT.md)**: Docker 測試結果 & 演算法驗證

---

## 環境需求

**開發環境**:
- GCC 編譯器
- Open MPI 4.1+
- Python 3（用於測試資料生成）

**叢集部署**:
- SSH 存取權（支援 ProxyJump）
- 節點間無密碼 SSH
- 共享或同步的 home 目錄
- 所有節點已安裝 MPI

---

## 編譯

```bash
# 自動編譯
./compile.sh

# 手動編譯
gcc -O3 -o attention attention.c -lm
mpicc -O3 -o attention-mpi attention-mpi.c -lm
```

---

## 測試

```bash
# 生成測試資料（多種規模）
python3 test_data_generator.py

# 快速測試（串行基準）
./attention test_xlarge.bin

# MPI 測試（單節點，4 processes）
mpirun -np 4 --mca btl ^openib ./attention-mpi test_xlarge.bin

# 叢集測試（2 節點，每節點 8 processes）
mpirun -H 172.16.179.50:8,172.16.179.55:8 \
    --mca btl ^openib \
    ~/HP_HW3/attention-mpi ~/HP_HW3/test_xlarge.bin
```

---

## 授權

學術專案 - 請參閱課程資料了解使用條款

## 作者

Team10 - 高效能計算課程

---

**最後更新**: 2025-10-26  
**叢集測試環境**: CSIE HPC Cluster (rdma4-rdma7)  
**MPI 版本**: Open MPI 4.1.4
