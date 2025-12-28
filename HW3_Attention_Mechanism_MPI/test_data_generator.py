#!/usr/bin/env python3
"""
Test Data Generator for Attention Mechanism
Generates binary test files with Q, K, V matrices and expected results
"""

import numpy as np
import struct
import sys
import os

def scaled_dot_product_attention(Q, K, V):
    """
    計算 Scaled Dot-Product Attention
    
    Args:
        Q: Query matrix (m, dk)
        K: Key matrix (n, dk)
        V: Value matrix (n, dv)
    
    Returns:
        Attention output (m, dv)
    """
    dk = Q.shape[1]
    
    # Step 1: 計算注意力分數 scores = Q @ K.T / sqrt(dk)
    scores = np.matmul(Q, K.T) / np.sqrt(dk)
    
    # Step 2: 應用 softmax (數值穩定版本)
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Step 3: 加權求和 output = attention_weights @ V
    output = np.matmul(attention_weights, V)
    
    return output

def generate_test_data(m, n, dk, dv, filename, seed=42):
    """
    生成測試資料並儲存為二進位檔案
    
    Args:
        m: Query 矩陣行數
        n: Key/Value 矩陣行數
        dk: Key 維度
        dv: Value 維度
        filename: 輸出檔案名稱
        seed: 隨機種子
    """
    np.random.seed(seed)
    
    # 生成隨機矩陣 (標準常態分佈)
    Q = np.random.randn(m, dk)
    K = np.random.randn(n, dk)
    V = np.random.randn(n, dv)
    
    # 計算正確答案
    result = scaled_dot_product_attention(Q, K, V)
    
    # 寫入二進位檔案
    with open(filename, 'wb') as f:
        # 寫入維度 (4 個整數)
        f.write(struct.pack('i', m))
        f.write(struct.pack('i', n))
        f.write(struct.pack('i', dk))
        f.write(struct.pack('i', dv))
        
        # 寫入矩陣 (row-major order, double precision)
        f.write(Q.astype(np.float64).tobytes())
        f.write(K.astype(np.float64).tobytes())
        f.write(V.astype(np.float64).tobytes())
        f.write(result.astype(np.float64).tobytes())
    
    file_size = os.path.getsize(filename)
    
    print(f"Generated {filename}:")
    print(f"  Dimensions: m={m}, n={n}, dk={dk}, dv={dv}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Result shape: {result.shape}")
    print(f"  Random seed: {seed}")
    print()

def read_test_data(filename):
    """讀取測試資料並返回矩陣"""
    with open(filename, 'rb') as f:
        # 讀取維度
        m, n, dk, dv = struct.unpack('iiii', f.read(16))
        
        # 讀取矩陣
        Q = np.frombuffer(f.read(m*dk*8), dtype=np.float64).reshape(m, dk)
        K = np.frombuffer(f.read(n*dk*8), dtype=np.float64).reshape(n, dk)
        V = np.frombuffer(f.read(n*dv*8), dtype=np.float64).reshape(n, dv)
        expected = np.frombuffer(f.read(m*dv*8), dtype=np.float64).reshape(m, dv)
    
    return Q, K, V, expected, (m, n, dk, dv)

def verify_test_data(filename):
    """驗證測試資料的正確性"""
    Q, K, V, expected, dims = read_test_data(filename)
    
    # 重新計算
    computed = scaled_dot_product_attention(Q, K, V)
    
    # 比較結果
    diff = np.abs(computed - expected)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"Verification for {filename}:")
    print(f"  Dimensions: m={dims[0]}, n={dims[1]}, dk={dims[2]}, dv={dims[3]}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Avg difference: {avg_diff:.2e}")
    print(f"  Status: {'PASS ✓' if max_diff < 1e-10 else 'FAIL ✗'}")
    print()
    
    return max_diff < 1e-10

def main():
    """主函數：生成所有測試資料"""
    
    print("=" * 60)
    print("Attention Mechanism Test Data Generator")
    print("=" * 60)
    print()
    
    # 測試案例 1: 極小測試 (除錯用)
    generate_test_data(4, 4, 2, 2, "test_tiny.bin", seed=1)
    
    # 測試案例 2: 小型測試 (快速驗證)
    generate_test_data(8, 8, 4, 4, "test_small.bin", seed=42)
    
    # 測試案例 3: 中型測試
    generate_test_data(64, 64, 32, 32, "test_medium.bin", seed=123)
    
    # 測試案例 4: 大型測試 (效能測試)
    generate_test_data(256, 256, 64, 64, "test_large.bin", seed=456)
    
    # 測試案例 5: 超大型測試 (壓力測試)
    generate_test_data(1024, 1024, 64, 64, "test_xlarge.bin", seed=789)
    
    # 測試案例 6: 非對稱矩陣
    generate_test_data(100, 200, 50, 80, "test_asymmetric.bin", seed=999)
    
    # 測試案例 7: 超級大型 (多節點測試)
    generate_test_data(2048, 2048, 128, 128, "test_xxlarge.bin", seed=2024)
    
    print("=" * 60)
    print("All test data generated successfully!")
    print("=" * 60)
    print()
    
    # 驗證所有測試資料
    print("Verifying generated test data...")
    print("-" * 60)
    
    test_files = [
        "test_tiny.bin",
        "test_small.bin",
        "test_medium.bin",
        "test_large.bin",
        "test_xlarge.bin",
        "test_asymmetric.bin",
        "test_xxlarge.bin"
    ]
    
    all_passed = True
    for f in test_files:
        if os.path.exists(f):
            passed = verify_test_data(f)
            all_passed = all_passed and passed
    
    print("-" * 60)
    if all_passed:
        print("All verifications PASSED ✓")
    else:
        print("Some verifications FAILED ✗")
    print()

if __name__ == "__main__":
    main()
