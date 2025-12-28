# MPI 叢集部署疑難排解指南

## 目錄

1. [問題清單](#問題清單)
2. [Hostfile 主機名問題](#1-hostfile-主機名問題)
3. [跨節點通訊超時](#2-跨節點通訊超時)
4. [檔案同步問題](#3-檔案同步問題)
5. [網路介面配置](#4-網路介面配置)
6. [Python 依賴問題](#5-python-依賴問題)
7. [多節點效率低下](#6-多節點效率低下)
8. [診斷工具與技巧](#診斷工具與技巧)

---

## 問題清單

| 問題 | 嚴重性 | 狀態 | 解決方案 |
|------|--------|------|----------|
| Hostfile 主機名無法解析 | 🔴 高 | ✅ 已解決 | 使用 IP 地址 |
| 跨節點 MPI 通訊超時 | 🔴 高 | ✅ 已解決 | `--mca btl ^openib` |
| 檔案路徑找不到 | 🟡 中 | ✅ 已解決 | 使用絕對路徑 + 同步 |
| 網路介面配置錯誤 | 🟡 中 | ✅ 已解決 | 移除 `eth0` 參數 |
| Python 不可用 | 🟢 低 | ⚠️ 繞過 | 預先生成測試資料 |
| 多節點效率極低 | 🟡 中 | ⚠️ 設計限制 | 增大問題規模 |

---

## 1. Hostfile 主機名問題

### 1.1 問題描述

**症狀**:
```bash
$ mpirun --hostfile hosts -np 8 hostname
ssh: Could not resolve hostname inventec-1: Temporary failure in name resolution
ssh: Could not resolve hostname inventec-5: Temporary failure in name resolution
ORTE was unable to reliably start one or more daemons.
```

**原因**:
- Hostfile 使用的主機名 (`rdma4`, `rdma5`, `inventec-1`, 等) 無法被 DNS 解析
- 實際主機名與別名不一致:
  - `rdma4` → `inventec-0` (不是 `inventec-4`)
  - `rdma5` → `inventec-5`
  - `rdma6` → `inventec-6`
  - `rdma7` → `inventec-7`

### 1.2 診斷步驟

**Step 1: 檢查實際主機名**

```bash
# 在各節點執行
hostname

# 輸出:
# rdma4: inventec-0
# rdma5: inventec-5
# rdma6: inventec-6
# rdma7: inventec-7
```

**Step 2: 測試 SSH 連線**

```bash
# 測試別名
ssh rdma5 hostname  # ✅ 成功: inventec-5
ssh rdma6 hostname  # ✅ 成功: inventec-6

# 測試主機名
ssh inventec-5 hostname  # ❌ 失敗: Cannot resolve
ssh inventec-1 hostname  # ❌ 失敗: Cannot resolve
```

**Step 3: 檢查 IP 地址**

```bash
# 查看本機 IP
ip addr show ens81np0 | grep "inet "
# 輸出: 172.16.179.50/16

# 查看其他節點
ssh rdma5 "ip addr show ens81np0 | grep 'inet '"
# 輸出: 172.16.179.55/16
```

### 1.3 解決方案

**方案 1: 使用 IP 地址** (✅ 推薦)

```bash
cat > hosts <<EOF
172.16.179.50 slots=16
172.16.179.55 slots=16
172.16.179.56 slots=16
172.16.179.57 slots=16
EOF
```

**測試**:
```bash
mpirun --hostfile hosts -np 8 --mca btl ^openib hostname
# ✅ 成功: 輸出 inventec-0, inventec-5, inventec-6, inventec-7
```

**方案 2: 使用別名** (需 SSH 配置)

```bash
cat > hosts <<EOF
rdma4 slots=16
rdma5 slots=16
rdma6 slots=16
rdma7 slots=16
EOF
```

但需確保 SSH 可解析（通常叢集管理員已配置 `/etc/hosts`）。

### 1.4 經驗教訓

- ✅ **永遠使用 IP 地址**最可靠
- ⚠️ **不要假設主機名規則**（inventec-0 ≠ rdma4 的數字對應）
- 🔍 **先測試 SSH 連線**再執行 MPI 程式

---

## 2. 跨節點通訊超時

### 2.1 問題描述

**症狀**:
```bash
$ mpirun --hostfile hosts -npernode 16 ./attention-mpi test_xlarge.bin
# 程式卡住，沒有任何輸出
# 180 秒後超時
TIMEOUT (>180s)
```

**初步診斷**:
```bash
# 單節點測試
$ mpirun -np 16 ./attention-mpi test_xlarge.bin
Correct! Elapsed time: 27300.63 us
# ✅ 成功

# 跨節點測試
$ mpirun -H 172.16.179.50:16,172.16.179.55:16 ./attention-mpi test_xlarge.bin
# ❌ 卡住
```

### 2.2 原因分析

**問題根源**: MPI 嘗試使用 InfiniBand (OpenFabrics) 驅動，但叢集上未安裝或未配置。

**錯誤訊息** (verbose 模式):
```bash
$ mpirun --hostfile hosts -np 8 --mca btl_base_verbose 10 hostname
[inventec-0:1174236] mca: base: components_register: found loaded component openib
BTL openib: No active ports found
```

### 2.3 解決方案

**核心參數**: `--mca btl ^openib`

```bash
# 禁用 OpenIB (InfiniBand)，強制使用 TCP
mpirun --hostfile hosts -np 8 --mca btl ^openib hostname
```

**完整執行指令**:

```bash
# 2 節點測試
mpirun -H 172.16.179.50:8,172.16.179.55:8 \
    --mca btl ^openib \
    /home/Team10/HP_HW3/attention-mpi \
    /home/Team10/HP_HW3/test_xlarge.bin
```

**驗證成功**:
```bash
$ mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib hostname
inventec-0
inventec-0
inventec-5
inventec-5
# ✅ 成功看到兩個節點
```

### 2.4 其他可能參數

如果 `--mca btl ^openib` 仍有問題，可嘗試:

```bash
# 明確指定使用 self + tcp
--mca btl self,tcp

# 指定網路介面
--mca btl_tcp_if_include ens81np0

# 禁用 tree spawn
--mca plm_rsh_no_tree_spawn 1

# 組合使用
mpirun --mca btl ^openib --mca btl_tcp_if_include ens81np0 ...
```

### 2.5 診斷技巧

**1. 測試基本跨節點通訊**:
```bash
# 最簡單的測試
mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib hostname
```

**2. 使用 verbose 模式**:
```bash
mpirun --mca btl_base_verbose 10 --mca plm_base_verbose 10 ...
```

**3. 測試 SSH 連線**:
```bash
# 確保無密碼 SSH
ssh 172.16.179.55 echo OK
ssh rdma5 echo OK
```

---

## 3. 檔案同步問題

### 3.1 問題描述

**症狀**:
```bash
$ mpirun --hostfile hosts -npernode 8 ./attention-mpi test_xlarge.bin
--------------------------------------------------------------------------
mpirun was unable to launch the specified application as it could not access
or execute an executable:

Executable: ./attention-mpi
Node: rdma5

while attempting to start process rank 8.
--------------------------------------------------------------------------
```

**原因**: `attention-mpi` 只在 rdma4 (inventec-0) 上，其他節點找不到。

### 3.2 診斷步驟

**Step 1: 檢查檔案是否存在**

```bash
# 本機 (rdma4)
ls -lh ~/HP_HW3/attention-mpi
# -rwxrwxr-x 1 Team10 Team10 25K Oct 26 12:01 attention-mpi

# 其他節點
ssh rdma5 "ls -lh ~/HP_HW3/attention-mpi"
# ls: cannot access ~/HP_HW3/attention-mpi: No such file or directory
# ❌ 不存在！
```

**Step 2: 檢查目錄結構**

```bash
ssh rdma5 "ls ~/HP_HW3/"
# ls: cannot access ~/HP_HW3/: No such file or directory
# ❌ 連目錄都不存在！
```

### 3.3 解決方案

**方案 1: 手動同步** (快速)

```bash
# 創建目錄
for node in rdma5 rdma6 rdma7; do
    ssh $node "mkdir -p ~/HP_HW3"
done

# 同步執行檔與測試資料
for node in rdma5 rdma6 rdma7; do
    scp attention-mpi test_*.bin $node:~/HP_HW3/
done
```

**方案 2: 自動化腳本**

```bash
#!/bin/bash
# sync_to_all_nodes.sh

NODES="rdma5 rdma6 rdma7"
FILES="attention-mpi test_small.bin test_medium.bin test_large.bin test_xlarge.bin"

for node in $NODES; do
    echo "Syncing to $node..."
    ssh $node "mkdir -p ~/HP_HW3"
    scp $FILES $node:~/HP_HW3/
    
    # 驗證
    ssh $node "ls -lh ~/HP_HW3/attention-mpi"
done

echo "Sync complete!"
```

**方案 3: 使用絕對路徑** (避免相對路徑問題)

```bash
# 不要用 ./attention-mpi
# 使用完整路徑
mpirun --hostfile hosts -npernode 8 \
    /home/Team10/HP_HW3/attention-mpi \
    /home/Team10/HP_HW3/test_xlarge.bin
```

### 3.4 驗證同步

```bash
# 驗證腳本
for node in rdma4 rdma5 rdma6 rdma7; do
    echo "=== $node ==="
    ssh $node "ls -lh ~/HP_HW3/attention-mpi ~/HP_HW3/test_xlarge.bin"
done
```

**預期輸出**:
```
=== rdma4 ===
-rwxrwxr-x 1 Team10 Team10  25K Oct 26 12:01 attention-mpi
-rw-r--r-- 1 Team10 Team10 2.1M Oct 26 12:01 test_xlarge.bin

=== rdma5 ===
-rwxrwxr-x 1 Team10 Team10  25K Oct 26 12:01 attention-mpi
-rw-r--r-- 1 Team10 Team10 2.1M Oct 26 12:01 test_xlarge.bin
...
```

---

## 4. 網路介面配置

### 4.1 問題描述

**症狀**:
```bash
$ mpirun --hostfile hosts --mca btl_tcp_if_include eth0 -np 8 hostname
--------------------------------------------------------------------------
None of the TCP networks specified to be included for out-of-band communications
could be found:

  Value given: eth0

Please revise the specification and try again.
--------------------------------------------------------------------------
```

**原因**: 叢集網路介面是 `ens81np0`，不是 `eth0`。

### 4.2 診斷

**Step 1: 檢查網路介面**

```bash
ip addr show | grep -E "^[0-9]+:|inet "

# 輸出:
# 1: lo: <LOOPBACK,UP,LOWER_UP>
# 2: enx7accc64d6358: <BROADCAST,MULTICAST>
# 3: ens81np0: <BROADCAST,MULTICAST,UP,LOWER_UP>
#     inet 172.16.179.50/16 brd 172.16.179.255 scope global ens81np0
```

**Step 2: 確認活躍介面**

```bash
ip -o -4 addr show | grep -v "127.0.0.1"
# 3: ens81np0    inet 172.16.179.50/16 ...
```

### 4.3 解決方案

**移除 `eth0` 參數** 或 **使用正確介面名稱**:

```bash
# 方案 1: 不指定介面 (推薦)
mpirun --hostfile hosts --mca btl ^openib -np 8 hostname

# 方案 2: 指定正確介面
mpirun --hostfile hosts --mca btl ^openib --mca btl_tcp_if_include ens81np0 -np 8 hostname
```

### 4.4 常見網路介面名稱

| 傳統命名 | 新命名 (Predictable Network Names) |
|----------|-------------------------------------|
| `eth0` | `ens33`, `ens81np0`, `enp0s3` |
| `eth1` | `ens34`, `enp0s8` |
| `wlan0` | `wlp3s0` |

**查詢方法**:
```bash
# 列出所有網路介面
ip link show

# 只看活躍的
ip -o link show | grep "state UP"
```

---

## 5. Python 依賴問題

### 5.1 問題描述

**症狀**:
```bash
$ python3 test_data_generator.py
bash: python3: command not found

$ which python
# (無輸出)
```

**影響**: 無法在叢集上動態生成測試資料。

### 5.2 解決方案

**方案 1: 預先生成測試資料** (✅ 推薦)

```bash
# 在本機 (有 Python 的環境) 生成
python3 test_data_generator.py

# 上傳到叢集
scp -J hpcai_course_student@140.112.90.37:9037 \
    test_*.bin Team10@172.16.179.50:~/HP_HW3/
```

**方案 2: 使用 module 系統**

```bash
# 查詢可用模組
module avail python

# 載入 Python 模組
module load python/3.8

# 驗證
python3 --version
```

**方案 3: 使用 Conda/Mamba** (如果可用)

```bash
# 檢查是否有 conda
which conda
which mamba

# 如果有，啟用環境
conda activate base
```

### 5.3 替代方案

如果完全無法使用 Python，可以用 C 程式生成測試資料：

```c
// generate_test_data.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_random_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// ... (實作 attention 計算)
```

---

## 6. 多節點效率低下

### 6.1 問題描述

**症狀**:
```
單節點 8 procs:  29,490.85 μs (3.77x speedup, 47.1% efficiency)
2 節點 × 8 procs: 39,104.18 μs (2.85x speedup, 17.8% efficiency)
4 節點 × 8 procs: 40,076.07 μs (2.78x speedup, 8.7% efficiency)
```

**現象**: 增加節點反而變慢！

### 6.2 原因分析

**通訊開銷主導**:

| 階段 | 單節點 (μs) | 2 節點 (μs) | 增加 |
|------|-------------|-------------|------|
| 通訊 | ~1,000 | ~20,000 | 20x |
| 計算 | ~28,000 | ~18,000 | -36% |
| **總計** | **29,490** | **39,104** | **+33%** |

**關鍵問題**:
1. **TCP/IP 延遲**: 每次通訊 100-200 μs latency
2. **資料量大**: 需廣播 1 MB (K + V 矩陣)
3. **同步次數多**: Bcast × 2 + Scatterv + Gatherv = 4 次同步
4. **問題規模小**: m=1024 不足以分攤通訊成本

### 6.3 解決方案

**方案 1: 增大問題規模** (✅ 推薦)

```bash
# 生成更大的測試資料
# m = 4096, n = 4096 (16x 資料量)
python3 test_data_generator.py --m 4096 --n 4096 --output test_huge.bin

# 測試
mpirun -H 172.16.179.50:16,172.16.179.55:16 --mca btl ^openib \
    ./attention-mpi test_huge.bin
```

**預期效果**: 計算/通訊 比例 > 10:1，效率提升至 30-50%

**方案 2: 優化通訊模式**

```c
// 改進: 減少廣播次數
// 將 K, V 合併為一次廣播
double* KV_combined = malloc((n*dk + n*dv) * sizeof(double));
// ... 組合 K, V
MPI_Bcast(KV_combined, n*dk + n*dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

**方案 3: 改用 InfiniBand** (需硬體支援)

如果叢集有 InfiniBand 但未配置:
```bash
# 聯絡管理員啟用
# 延遲可降至 1-5 μs (vs TCP 100-200 μs)
```

### 6.4 何時使用多節點

**經驗法則**:

```
計算/通訊比 > 10:1

計算時間 ≈ O(m × n × d)
通訊時間 ≈ 4 × (latency + bandwidth × data_size)

對於 TCP/IP:
latency ≈ 150 μs
bandwidth ≈ 1 GB/s

m ≥ 4096 才適合 2-4 節點
m ≥ 8192 才適合 4+ 節點
```

---

## 診斷工具與技巧

### 通用診斷流程

```bash
# 1. 測試 SSH 連線
for node in rdma4 rdma5 rdma6 rdma7; do
    ssh $node hostname || echo "$node FAILED"
done

# 2. 測試 MPI 基本通訊
mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib hostname

# 3. 檢查檔案同步
for node in rdma4 rdma5 rdma6 rdma7; do
    echo "=== $node ==="
    ssh $node "ls -lh ~/HP_HW3/attention-mpi"
done

# 4. 小規模測試
mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib \
    ~/HP_HW3/attention-mpi ~/HP_HW3/test_small.bin

# 5. 逐步增加規模
# 2 nodes → 3 nodes → 4 nodes
# 2 procs/node → 4 → 8 → 16
```

### 診斷腳本

**完整診斷腳本** (`diagnose_cluster.sh`):

```bash
#!/bin/bash

echo "=========================================="
echo "MPI 叢集診斷工具"
echo "=========================================="
echo ""

# 1. SSH 連線測試
echo "[1] 測試 SSH 連線"
for node in rdma4 rdma5 rdma6 rdma7; do
    echo -n "  $node: "
    ssh -o ConnectTimeout=5 $node "hostname" 2>/dev/null && echo "✓" || echo "✗"
done
echo ""

# 2. IP 連線測試
echo "[2] 測試 IP 連線"
for ip in 172.16.179.50 172.16.179.55 172.16.179.56 172.16.179.57; do
    echo -n "  $ip: "
    ssh -o ConnectTimeout=5 $ip "hostname" 2>/dev/null && echo "✓" || echo "✗"
done
echo ""

# 3. 檔案同步檢查
echo "[3] 檢查檔案同步"
for node in rdma4 rdma5 rdma6 rdma7; do
    echo "  $node:"
    ssh $node "ls -lh ~/HP_HW3/attention-mpi ~/HP_HW3/test_xlarge.bin 2>/dev/null" | sed 's/^/    /'
done
echo ""

# 4. MPI 通訊測試
echo "[4] 測試 MPI 通訊 (hostname)"
mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib hostname 2>&1 | head -10
echo ""

# 5. 小規模程式測試
echo "[5] 測試 MPI 程式 (test_small.bin)"
timeout 60 mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib \
    ~/HP_HW3/attention-mpi ~/HP_HW3/test_small.bin
echo ""

echo "=========================================="
echo "診斷完成"
echo "=========================================="
```

### 效能 Profiling

**使用 MPI 內建工具**:

```bash
# 啟用 profiling
mpirun --mca btl ^openib --mca ompi_display_comm on ...

# 查看通訊統計
mpirun --mca btl_base_verbose 10 ...
```

**時間測量**:

```bash
# 分解時間
time mpirun --mca btl ^openib ... > /dev/null

# 輸出:
# real    0m15.234s  (總時間)
# user    0m8.145s   (CPU 時間)
# sys     0m2.456s   (系統時間)
```

---

## 快速參考

### 常用指令

```bash
# 測試 SSH
ssh rdma5 hostname

# 測試 MPI (單節點)
mpirun -np 4 --mca btl ^openib ./attention-mpi test_xlarge.bin

# 測試 MPI (多節點)
mpirun -H 172.16.179.50:2,172.16.179.55:2 --mca btl ^openib hostname

# 同步檔案
scp attention-mpi test_*.bin rdma5:~/HP_HW3/

# 檢查網路
ip addr show ens81np0
```

### 除錯 Checklist

- [ ] SSH 無密碼登入正常
- [ ] Hostfile 使用 IP 地址
- [ ] 所有節點已同步執行檔與資料
- [ ] 使用 `--mca btl ^openib` 參數
- [ ] 使用絕對路徑
- [ ] 問題規模足夠大 (m ≥ 4096 for multi-node)
- [ ] 測試過 `hostname` 指令
- [ ] 檢查網路介面名稱 (ens81np0, 不是 eth0)

---

**文件版本**: 1.0  
**更新日期**: 2025年10月26日  
**作者**: Team10 - MPI Troubleshooting Guide
