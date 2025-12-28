import pandas as pd
import os
import re

def extract_data_from_txt_files():
    """從現有的 txt 檔案中提取數據"""
    results_dir = "results"
    
    # 找到最新的結果檔案
    txt_files = [f for f in os.listdir(results_dir) if f.endswith('.txt') and 'results' in f]
    
    if not txt_files:
        print("沒有找到結果檔案")
        return
    
    # 處理 OpenMP 結果
    openmp_file = [f for f in txt_files if 'openmp' in f]
    if openmp_file:
        print(f"處理檔案: {openmp_file[0]}")
        
        with open(os.path.join(results_dir, openmp_file[0]), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析數據
        openmp_data = []
        lines = content.split('\n')
        
        current_threads = None
        for i, line in enumerate(lines):
            # 找到線程數
            if 'threads, Matrix:' in line:
                match = re.search(r'(\d+) threads, Matrix: (\d+) x (\d+), Kernel: (\d+) x (\d+)', line)
                if match:
                    current_threads = int(match.group(1))
                    matrix_size = int(match.group(2))
                    kernel_size = int(match.group(4))
                    
                    # 查找下一行的時間
                    if i + 2 < len(lines) and 'Elapsed time:' in lines[i + 2]:
                        time_match = re.search(r'Elapsed time: ([\d.]+) us', lines[i + 2])
                        if time_match:
                            exec_time = float(time_match.group(1))
                            openmp_data.append({
                                'Threads': current_threads,
                                'Matrix_Size': matrix_size,
                                'Kernel_Size': kernel_size,
                                'Execution_Time_us': exec_time,
                                'Status': 'PASS'
                            })
        
        # 創建 DataFrame 並保存
        if openmp_data:
            df = pd.DataFrame(openmp_data)
            timestamp = "2025-09-22_fixed"
            output_file = os.path.join(results_dir, f'openmp_performance_{timestamp}.csv')
            df.to_csv(output_file, index=False)
            print(f"已創建修復的 OpenMP 數據: {output_file}")
            print(f"提取了 {len(openmp_data)} 筆數據")
            
            # 顯示數據摘要
            print("\n數據摘要:")
            print(df.groupby(['Threads', 'Matrix_Size'])['Execution_Time_us'].count())
            
            return df
    
    return None

def analyze_performance(df):
    """分析性能數據"""
    if df is None:
        return
    
    print("\n" + "="*60)
    print("性能分析報告")
    print("="*60)
    
    # 計算加速比
    print("\n加速比分析:")
    for matrix in sorted(df['Matrix_Size'].unique()):
        for kernel in sorted(df['Kernel_Size'].unique()):
            subset = df[(df['Matrix_Size'] == matrix) & (df['Kernel_Size'] == kernel)]
            baseline = subset[subset['Threads'] == 1]['Execution_Time_us']
            
            if len(baseline) > 0:
                baseline_time = baseline.iloc[0]
                print(f"\nMatrix {matrix}x{matrix}, Kernel {kernel}x{kernel}:")
                
                for threads in sorted(subset['Threads'].unique()):
                    thread_data = subset[subset['Threads'] == threads]['Execution_Time_us']
                    if len(thread_data) > 0:
                        thread_time = thread_data.iloc[0]
                        speedup = baseline_time / thread_time
                        efficiency = speedup / threads * 100
                        print(f"  {threads:2d} threads: {speedup:5.2f}x speedup, {efficiency:5.1f}% efficiency")

def create_analysis_script():
    """創建增強的分析腳本"""
    script_content = '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_analyze():
    # 找到最新的 CSV 檔案
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    openmp_files = [f for f in csv_files if 'openmp_performance' in f]
    
    if not openmp_files:
        print("找不到 OpenMP 數據檔案")
        return
    
    # 載入最新的檔案
    latest_file = sorted(openmp_files)[-1]
    df = pd.read_csv(latest_file)
    
    print(f"載入檔案: {latest_file}")
    print(f"數據筆數: {len(df)}")
    
    # 基本統計
    print("\\n基本統計:")
    print(df.groupby(['Threads', 'Matrix_Size'])['Execution_Time_us'].describe())
    
    # 繪製加速比圖
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange']
    matrices = sorted(df['Matrix_Size'].unique())
    
    for i, matrix in enumerate(matrices):
        plt.subplot(2, 2, i+1)
        
        for kernel in sorted(df['Kernel_Size'].unique()):
            subset = df[(df['Matrix_Size'] == matrix) & (df['Kernel_Size'] == kernel)]
            baseline = subset[subset['Threads'] == 1]['Execution_Time_us']
            
            if len(baseline) > 0:
                baseline_time = baseline.iloc[0]
                threads = []
                speedups = []
                
                for thread_count in sorted(subset['Threads'].unique()):
                    thread_data = subset[subset['Threads'] == thread_count]['Execution_Time_us']
                    if len(thread_data) > 0:
                        threads.append(thread_count)
                        speedups.append(baseline_time / thread_data.iloc[0])
                
                plt.plot(threads, speedups, 'o-', label=f'Kernel {kernel}x{kernel}')
        
        # 理想加速比線
        max_threads = max(df['Threads'])
        plt.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='理想加速比')
        
        plt.title(f'Matrix {matrix}x{matrix}')
        plt.xlabel('線程數')
        plt.ylabel('加速比')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n圖表已保存為 speedup_analysis.png")

if __name__ == "__main__":
    load_and_analyze()
'''
    
    with open('results/analyze_fixed.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("已創建分析腳本: results/analyze_fixed.py")

def main():
    print("修復和分析 HPC HW2 數據")
    print("="*40)
    
    # 提取數據
    df = extract_data_from_txt_files()
    
    # 分析性能
    analyze_performance(df)
    
    # 創建分析腳本
    create_analysis_script()
    
    print("\n使用方法:")
    print("cd results")
    print("python analyze_fixed.py")

if __name__ == "__main__":
    main()